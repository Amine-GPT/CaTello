from djitellopy import Tello
import cv2
import time
import numpy as np

# Connect to drone and enable video stream
drone = Tello()
drone.connect()
drone.streamon()

# Load YOLO
net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")
layer_names = net.getLayerNames()  # Get the names of all the layers in the network
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]  # Get the output layers of the network

# Initialize parameters for cat face tracking
threshold = 20  # Threshold value for determining movement error
scaling_factor = 0.6  # Scaling factor for the movement error in the y direction
fb_range = [70000, 100000]  # Range of cat face area for steady forward-backward movement
pid = [0.4, 0.4, 0]  # PID constants for drone control [proportional, integral, derivative]
pError = 0  # Previous error for PID controller
last_known_face_position = (0, 0, time.time())  # Last known cat face position (x, y, timestamp)
last_known_box = None  # Last detected cat face bounding box
no_face_count = 0  # Number of frames without a detected cat face

manual_control = False  # Flag indicating manual control mode
cooldown_time = 3  # Time in seconds for which manual control stays active after a key press
last_keypress_time = 0  # The time of the last key press
lr, fb, ud, yv = 0, 0, 0, 0  # Drone control variables (left-right, forward-backward, up-down, yaw)


# Function to detect cat face in the image
def find_face(img):
    """
        Detects a cat face in the given image using YOLO.

        Args:
            img (numpy.ndarray): The input image.

        Returns:
            tuple: A tuple containing the modified image with bounding box and center coordinates of the detected cat face.
        """
    global last_known_box, no_face_count

    height, width, channels = img.shape  # Get the dimensions of the image
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)  # Create a blob from the image
    net.setInput(blob)  # Set the blob as the input to the network
    outs = net.forward(output_layers)  # Forward pass through the network

    class_ids = []  # Initialize empty lists to store class IDs, confidences, and bounding boxes
    confidences = []
    boxes = []

    max_area = 0  # Initialize the maximum area and center coordinates
    cx, cy = 0, 0

    for out in outs:
        for detection in out:
            scores = detection[5:]  # Extract the confidence scores for the classes
            class_id = np.argmax(scores)  # Get the index of the class with the highest score
            confidence = scores[class_id]  # Get the confidence of the detected class
            if class_id == 15 and confidence > 0.5:  # Confidence threshold
                # Object detected
                center_x = int(detection[0] * width)  # Calculate the center coordinates of the bounding box
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)   # Calculate the width and height of the bounding box
                h = int(detection[3] * height)
                area = w * h  # Calculate the area of the bounding box
                #  print(f'Face area: {area}')

                if area > max_area:  # Update the maximum area and center coordinates if a larger face is found
                    max_area = area
                    cx = center_x
                    cy = center_y

                # Rectangle coordinates
                x = int(center_x - w / 2)  # Calculate the top-left corner coordinates of the bounding box
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])  # Add the bounding box coordinates to the list
                confidences.append(float(confidence))  # Add the confidence to the list
                class_ids.append(class_id)  # Add the class ID to the list

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4) # Apply non-maximum suppression to remove overlapping boxes

    if len(indexes) == 0:  # If no face is detected
        no_face_count += 1  # Increment the count of frames without a detected face
        if last_known_box is not None and no_face_count <= 10:  # Check if there is a last known box and the count is within the threshold
            x, y, w, h, _, _ = last_known_box
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Draw the last known box on the image
            cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)  # Draw a circle at the center of the last known face.
    else:
        for i in range(len(boxes)):  # Iterate over the bounding boxes
            if i in indexes:  # Check if the bounding box index is selected by non-maximum suppression
                x, y, w, h = boxes[i]  # Get the coordinates and size of the selected bounding box
                last_known_box = (x, y, w, h, cx, cy)  # Update the last known box with the current box and center coordinates
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Draw the bounding box on the image
                cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)  # Draw a circle at the center of the detected face
        no_face_count = 0  # Reset the count when a face is detected

    return img, (cx, cy, max_area)  # Return the modified image and the coordinates and area of the detected face


def drone_control(cx, cy, img_width, img_height, area, pError, pid, last_known_pos):
    """
          Controls the movement of the drone based on the position of the cat face.

          Args:
              cx (int): X-coordinate of the cat face.
              cy (int): Y-coordinate of the cat face.
              img_width (int): Width of the image.
              img_height (int): Height of the image.
              area (int): Area of the cat face.
              pError (int): Previous error for PID controller.
              pid (list): PID constants for drone control [proportional, integral, derivative].
              last_known_pos (tuple): Last known cat face position (x, y, timestamp).

          Returns:
              int: Error in the x direction.
          """
    global last_known_face_position
    yaw_timer = 5  # Set the desired delay for constant yaw velocity (in seconds)

    if max_area == 0:  # If no face is detected
        cx, cy = last_known_pos[0], last_known_pos[1]  # Use the last known face position if no face is detected
    else:
        last_known_face_position = (cx, cy, time.time())  # Update the timer when a cat face is detected

    error_x = cx - img_width // 2  # Calculate the error in the x direction
    error_y = cy - img_height // 2  # Calculate the error in the y direction

    lr, fb, ud, yv = 0, 0, 0, 0  # Initialize the drone control variables
    lr_scaling_factor = 1  # Adjust this value if needed

    if max_area == 0:  # If no face is detected
        error_x = 0  # Set the error in the x direction to zero
        error_y = 0  # Set the error in the x direction to zero

        # Delay the constant yaw velocity using a timer
        if time.time() - last_known_face_position[2] > yaw_timer:
            yv = 20  # Reduced constant yaw velocity when no cat face is detected
        else:
            yv = 0
    else:
        if abs(error_x) > threshold:  # Check if the error in the x direction exceeds the threshold
            lr = int(max(-20, min(20, error_x * lr_scaling_factor)))  # Calculate the left-right movement
        else:
            lr = 0

        if abs(error_y) > threshold:  # Check if the error in the y direction exceeds the threshold
            fb = int(max(-20, min(20, error_y * scaling_factor)))  # Calculate the forward-backward movement
        else:
            fb = 0

        if area > fb_range[0] and area < fb_range[1]:  # Check if the area is within the desired range
            fb = 0
        elif area > fb_range[1]:  # Check if the area is larger than the upper range
            fb = -10
        elif area < fb_range[0] and area != 0:  # Check if the area is smaller than the lower range and not zero
            fb = 10

        p_x = error_x * pid[0]  # Calculate the proportional term for the PID controller
        d_x = (error_x - pError) * pid[1]  # Calculate the derivative term for the PID controller
        pError = error_x  # Update the previous error with the current error

        yv_raw = int(p_x + d_x)  # Calculate the raw yaw velocity based on the PID controller
        yv = max(-20, min(20, yv_raw))  # Limit the maximum yaw velocity

    drone.send_rc_control(lr, fb, ud, yv)  # Send the drone control commands
    return error_x  # Return the error in the x direction

# Main loop
while True:
    # Step 1: Get the drone's video feed
    img = drone.get_frame_read().frame

    # Step 2: Detect cat face in the image
    img, (cx, cy, max_area) = find_face(img)

    # Step 3: Resize the image to a higher size while maintaining aspect ratio
    scale_percent = 120  # Increase this value to resize the window to a higher size
    width = int(img.shape[1] * scale_percent / 100)  # Calculate the new width based on the scale percent
    height = int(img.shape[0] * scale_percent / 100)  # Calculate the new height based on the scale percent
    dim = (width, height)  # New dimensions for resizing
    resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    # Step 4: Control drone movement based on cat face position
    if not manual_control:
        pError = drone_control(cx, cy, resized_img.shape[1], resized_img.shape[0], max_area, pError,
                               pid, last_known_face_position)

    # Add text on the resized image based on cat face detection status
    if max_area > 0:
        cv2.putText(resized_img, "Target acquired...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(resized_img, "Searching for cats...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    battery_level = drone.get_battery()
    cv2.putText(resized_img, f'Battery: {battery_level}%', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                cv2.LINE_AA)
    cv2.imshow('Tello Drone Cat Face Tracking', resized_img)

    #  Handle keyboard input for drone control
    key = cv2.waitKey(1)

    if key != -1:
        last_keypress_time = time.time()  # Update the last key press time
        manual_control = True  # Enable manual control when a key is pressed

    if time.time() - last_keypress_time > cooldown_time:
        manual_control = False  # Switch off manual control after the cooldown time

    if key == ord('z'):
        drone.takeoff()
    elif key == ord('r'):
        drone.send_rc_control(0, 0, 50, 0)
        time.sleep(0.1)
        drone.send_rc_control(0, 0, 0, 0)
    elif key == ord('f'):
        drone.send_rc_control(0, 0, -50, 0)
        time.sleep(0.1)
        drone.send_rc_control(0, 0, 0, 0)
    elif key == ord('w'):
        drone.send_rc_control(0, 50, 0, 0)
        time.sleep(0.1)
        drone.send_rc_control(0, 0, 0, 0)
    elif key == ord('s'):
        drone.send_rc_control(0, -50, 0, 0)
        time.sleep(0.1)
        drone.send_rc_control(0, 0, 0, 0)
    elif key == ord('a'):
        drone.send_rc_control(-50, 0, 0, 0)
        time.sleep(0.1)
        drone.send_rc_control(0, 0, 0, 0)
    elif key == ord('d'):
        drone.send_rc_control(50, 0, 0, 0)
        time.sleep(0.1)
        drone.send_rc_control(0, 0, 0, 0)
    elif key == ord('e'):
        drone.send_rc_control(0, 0, 0, 50)
        time.sleep(0.5)
        drone.send_rc_control(0, 0, 0, 0)
    elif key == ord('q'):
        drone.send_rc_control(0, 0, 0, -50)
        time.sleep(0.5)
        drone.send_rc_control(0, 0, 0, 0)

    elif key == ord('x'):
        drone.land()
    elif key == ord(' '):
        drone.land()
        break

# Cleanup
cv2.destroyAllWindows()
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
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                area = w * h
                #print(f'Face area: {area}')
                if area > max_area:
                    max_area = area
                    cx = center_x
                    cy = center_y

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)  # Non-max suppression

    if len(indexes) == 0:  # If no face is detected
        no_face_count += 1
        if last_known_box is not None and no_face_count <= 10:
            x, y, w, h, _, _ = last_known_box
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
    else:
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                last_known_box = (x, y, w, h, cx, cy)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
        no_face_count = 0  # Reset the count when a face is detected

    return img, (cx, cy, max_area)


def drone_control(cx, cy, img_width, img_height, area, pError, pid, last_known_pos):
    global last_known_face_position  # Make sure to use the global variable
    yaw_timer = 5  # Set the desired delay for constant yaw velocity (in seconds)

    if max_area == 0:
        cx, cy = last_known_pos[0], last_known_pos[1]  # Use the last known face position if no face is detected
    else:
        last_known_face_position = (cx, cy, time.time())  # Update the timer when a cat face is detected

    error_x = cx - img_width // 2
    error_y = cy - img_height // 2

    lr, fb, ud, yv = 0, 0, 0, 0
    lr_scaling_factor = 1  # Adjust this value if needed

    if max_area == 0:
        error_x = 0
        error_y = 0

        # Delay the constant yaw velocity using a timer
        if time.time() - last_known_face_position[2] > yaw_timer:
            yv = 20  # Reduced constant yaw velocity when no cat face is detected
        else:
            yv = 0
    else:
        if abs(error_x) > threshold:
            lr = int(max(-25, min(25, error_x * lr_scaling_factor)))
        else:
            lr = 0

        if abs(error_y) > threshold:
            fb = int(max(-25, min(25, error_y * scaling_factor)))
        else:
            fb = 0

        if area > fb_range[0] and area < fb_range[1]:
            fb = 0
        elif area > fb_range[1]:
            fb = -10
        elif area < fb_range[0] and area != 0:
            fb = 10

        p_x = error_x * pid[0]
        d_x = (error_x - pError) * pid[1]
        pError = error_x

        yv_raw = int(p_x + d_x)
        yv = max(-25, min(25, yv_raw))  # Limit the maximum yaw velocity

    drone.send_rc_control(lr, fb, ud, yv)
    return error_x


# Main loop
while True:
    # Step 1: Get the drone's video feed
    img = drone.get_frame_read().frame

    # Step 2: Detect cat face in the image
    img, (cx, cy, max_area) = find_face(img)

    # Step 3: Resize the image to a higher size while maintaining aspect ratio
    scale_percent = 120  # Increase this value to resize the window to a higher size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    # Step 4: Control drone movement based on cat face position
    if not manual_control:
        pError = drone_control(cx, cy, resized_img.shape[1], resized_img.shape[0], max_area, pError,
                               pid, last_known_face_position)

    # Add text on the resized image based on cat face detection status
    if max_area > 0:
        cv2.putText(resized_img, "Target acquired...", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
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

    if key == ord('z'):
        drone.takeoff()
    elif key == ord('r'):
        drone.send_rc_control(0, 0, 20, 0)
        drone.send_rc_control(0, 0, 0, 0)
    elif key == ord('f'):
        drone.send_rc_control(0, 0, -20, 0)
        drone.send_rc_control(0, 0, 0, 0)
    elif key == ord('w'):
        drone.send_rc_control(0, 20, 0, 0)
        drone.send_rc_control(0, 0, 0, 0)
    elif key == ord('s'):
        drone.send_rc_control(0, -20, 0, 0)
        drone.send_rc_control(0, 0, 0, 0)
    elif key == ord('a'):
        drone.send_rc_control(-20, 0, 0, 0)
        drone.send_rc_control(0, 0, 0, 0)
    elif key == ord('d'):
        drone.send_rc_control(20, 0, 0, 0)
        drone.send_rc_control(0, 0, 0, 0)
    elif key == ord('q'):
        drone.send_rc_control(0, 0, 0, 20)
        drone.send_rc_control(0, 0, 0, 0)
    elif key == ord('e'):
        drone.send_rc_control(0, 0, 0, 0)
        drone.send_rc_control(0, 0, 0, -20)

    elif key == ord('x'):
        drone.land()
    elif key == ord(' '):
        drone.land()
        break

# Cleanup
cv2.destroyAllWindows()
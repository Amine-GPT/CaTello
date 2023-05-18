# CaTello
Meet CaTello - the cat-chasing drone designed to engage your cats.
This project marries DJI Ryze Tello drone with the AI-driven YOLOv4-tiny machine learning model, to create an  interactive pet companion.

<div align="center">
    <img src="https://github.com/Amine-GPT/CaTello/assets/131181870/d11a62a1-88b4-4e27-bd99-63070982b156" alt="Cat Image" width="500" height="500"/>
</div>

## Disclaimer :warning: :warning: :warning: 
Please note, this is my initial endeavor in Python, and as such, the drone's behavior may not always be as predictable as one might expect. This makes exercising caution and maintaining constant control over the drone extremely important for the safety of your pet and the drone.

We strongly recommend you to exercise the utmost caution when operating the drone, and to always keep it within a safe and open environment. Keep your hand on the keyboard, ready to press the 'SPACE' key, which will immediately shut down all processes in case of any unforeseen occurrences.

It's crucial that you familiarize yourself with and follow all the safety instructions provided by Tello Drone.

## Installation
Follow these instructions to get CaTello up and running:

1. Clone the repository: 
    ```bash
    git clone https://github.com/yourusername/Catello.git
    ```
2. Install the necessary Python libraries:
    ```python
    pip install djitellopy opencv-contrib-python
    ```
## Usage
Here's how to use catello:

1. Make sure your Tello drone is connected to Wi-Fi.
2. Run `catello.py`.
3. An OpenCV window should appear, displaying the drone's video feed and battery life.
4. Press `z` on your keyboard for the drone to take off, `x` to land, and the `spacebar` for an emergency landing or to stop the script.
5. Once airborne, the drone starts rotating clockwise to find and detect the cat. When detected, the drone will follow the cat, maintaining a safe distance.

https://github.com/Amine-GPT/CaTello/assets/131181870/bf307d00-b4d6-4d45-9bf1-9ca119c0cd03

## Configurations
Improve your drone's performance by updating these parameters in `catello.py`:

### Cat Face Detection
- Range of cat face area for steady forward-backward movement: `fb_range = [70000, 100000]`
- PID constants for drone control [proportional, integral, derivative]: `pid = [0.4, 0.4, 0]`
- Confidence threshold: `confidence > 0.5`

 <img src="https://github.com/Amine-GPT/CaTello/assets/131181870/be974f45-6faa-40a2-a8d0-36cc433a4737" alt="Cat Detection" width="250" height="250"/>

### Drone Velocity
- Reduced constant yaw velocity when no cat face is detected: `yv = 20`
- Min/Max left, right velocity: `lr = int(max(-20, min(20, error_x * lr_scaling_factor)))`
- Min/Max forward, backward velocity: `fb = int(max(-20, min(20, error_x * lr_scaling_factor)))`
- Min/Max yaw velocity: `yv = max(-20, min(20, yv_raw))`

## Manual Control
Override the automatic controls with these manual controls:

- Use `wasd` for forward, left, backward, and right respectively.
- Use `r,f` to go up and down.
- Use `q,e` for rotation counterclockwise and clockwise.
- Adjust `cooldown_time = 3` in `catello.py` to change the time it takes for the drone to switch from manual to automatic mode.

## Customization
Want to track something else? Change `class_id == 15` in `catello.py` to any of the YOLO4-tiny pretrained class_ids (you can find the list under `coco.names`).

## Credits
[Murtaza's Workshop](https://www.youtube.com/watch?v=LmEcyQnfpDA)

GPT4


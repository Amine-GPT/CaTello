# CaTello
Meet CaTello - the cat-chasing drone designed to stimulate and engage your cats.
This project marries DJI's Tello Ryze drone with the AI-driven YOLOv4-tiny machine learning model, to create an  interactive pet companion.

<div align="center">
    <img src="doc/images/cat1.png" alt="Cat Image" width="500" height="500"/>
</div>

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

## Manual Control
Override the automatic controls with these manual controls:

- Use `wasd` for forward, left, backward, and right respectively.
- Use `r,f` to go up and down.
- Use `q,e` for rotation counterclockwise and clockwise.
- Adjust `cooldown_time = 3` in `catello.py` to change the time it takes for the drone to switch from manual to automatic mode.

## Configurations
Improve your drone's performance by updating these parameters in `catello.py`:

### Cat Face Detection
- Range of cat face area for steady forward-backward movement: `fb_range = [70000, 100000]`
- PID constants for drone control [proportional, integral, derivative]: `pid = [0.4, 0.4, 0]`
- Confidence threshold: `confidence > 0.5`

### Drone Velocity
- Reduced constant yaw velocity when no cat face is detected: `yv = 20`
- Min/Max left, right velocity: `lr = int(max(-25, min(25, error_x * lr_scaling_factor)))`
- Min/Max forward, backward velocity: `fb = int(max(-25, min(25, error_x * lr_scaling_factor)))`
- Min/Max yaw velocity: `yv = max(-25, min(25, yv_raw))`

## Customization
Want to track something else? Change `class_id == 15` in `catello.py` to any of the YOLO4-tiny pretrained class_ids (you can find the list under `coco.names`).

## Contributing
We welcome contributions! If you're interested in improving Catello, please submit a pull request. For major changes, please open an issue first to discuss your ideas.

## Credits
This project was made possible by [insert sources here, including tutorials, videos, and OpenAI's GPT-4].

## License
This project is licensed under the [insert license here].

<div align="center">
    <img src="doc/images/cat2.png" alt="Cat Image" width="500" height="500"/>
    </div>

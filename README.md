# cs593-project
Project for CS593 at Purdue

## Requirements

### Equipment
1. Furhat robot, virtual or physical
2. Internet connection
3. Webcam stream

### Software
1. Python 3
2. Packages in `requirements.txt`
3. Ability to run `yolo-v8` without problems at 15FPS





## Running the code
```
usage: gui.py [-h] [--camera CAMERA] [--fps FPS]

Webcam viewer

options:
  -h, --help       show this help message and exit
  --camera CAMERA  Camera index
  --fps FPS        Target FPS
```

### UI Guide
- Start / Stop button: starts and stops the camera
- Furhat IP field: used to enter the Furhat IP. Enter the IP when first starting the program to connect to Furhat.
- Connect Furhat button: connects to Furhat with the given IP 
- Enable Tracking button: click to enable tracking
- Enable Greeting button: click to enable greeting
- Enable Gesture Attention button: click to enable gesture attention

### Basic Modes
1. Tracking: Tracking mode will have Furhat turn its head to follow the selected person at 0.5 second intervals. The following will continue until a new target is selected or if tracking is disabled. 
2. Greeting: In greeting mode, Furhat will attempt to "greet" and learn a person's name. When a new person enters the frame, Furhat will turn and face the persons and ask for their name. If Furhat receives a response, it will now mark the bounding box around the participant with the name instead of an ID. 
3. Gesture attention: In this mode, Furhat will respond to a particular gesture. When someone waves their hand in front of the camera (similar to getting one's attention), Furhat will turn to the person and greet them. 


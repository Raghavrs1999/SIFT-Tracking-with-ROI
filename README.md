# SIFT Tracking

This project implements a real-time object tracking system using the Scale-Invariant Feature Transform (SIFT) algorithm. It allows users to select a Region of Interest (ROI) in a live video stream and then tracks that object as it moves.

## Features

- Live video capture from a webcam
- Interactive ROI selection
- Real-time object tracking using SIFT features
- Visual feedback with bounding box around the tracked object

## Requirements

- Python 3.6+
- OpenCV (cv2) 4.5.0+
- NumPy 1.19.0+

## Installation

1. Clone this repository:
   ```
  https://github.com/Raghavrs1999/SIFT-Tracking-with-ROI.git
   cd SIFT-Tracking-with-ROI
   ```

2. (Optional) Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the script with:

```
python SIFT Tracking.py
```

1. When the live stream window opens, use your mouse to draw a rectangle around the object you want to track.
2. Release the mouse button to start tracking.
3. The selected object will be tracked in real-time, with a green bounding box indicating its position.
4. Press 'Esc' to exit the program.

## How it works

1. The script captures video from the default webcam.
2. The user selects a Region of Interest (ROI) by drawing a rectangle on the video feed.
3. SIFT features are extracted from the selected ROI.
4. In each subsequent frame, SIFT features are extracted and matched against the original ROI features.
5. A homography is computed to determine the new position and orientation of the tracked object.
6. A bounding box is drawn around the tracked object in each frame.

## Limitations

- The tracking may lose accuracy if the object undergoes significant changes in appearance, lighting, or if it's occluded.
- Performance may vary depending on the complexity of the scene and the power of your computer.

## Contributing

Contributions, issues, and feature requests are welcome. Feel free to check [issues page](https://github.com/yourusername/sift-tracking/issues) if you want to contribute.


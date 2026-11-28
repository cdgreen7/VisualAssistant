# Visual Assistant

A visual assistant for blind and low-vision users. 

## Hardware

- Laptop or USB webcam

## Pipeline

camera → YOLO object detection → spatial position → OCR / barcode → audio feedback

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python main.py # or python3 main.py
```

Press `q` to quit.

## Camera selection and permissions

`main.py` uses camera 0 unless `VA_CAMERA_INDEX` names another one. It prints the
camera it uses ("Using camera 0.").

```bash
VA_CAMERA_INDEX=1 python main.py           # macOS / Linux
$env:VA_CAMERA_INDEX=1; python main.py     # Windows PowerShell
```

A Mac can list several cameras: the built-in camera, iPhones through Continuity
Camera, and virtual cameras. Index 0 is not always the built-in one, so if the wrong
camera (or a phone) opens, try 1 or 2. `system_profiler SPCameraDataType` lists the
cameras macOS sees, though not in OpenCV's order.

If the camera doesn't start, `main.py` says why instead of closing silently:

- **"Couldn't open camera N"** usually means the app running Python has no camera
  permission. On macOS, open System Settings > Privacy & Security > Camera, turn on
  the app you run `main.py` from (Terminal, iTerm or VS Code), then quit and reopen
  that app. If it isn't listed, or access was refused earlier, run
  `tccutil reset Camera com.apple.Terminal` (use your app's bundle ID) and start
  `main.py` again to get the permission prompt. It can also mean the index doesn't
  exist: try another `VA_CAMERA_INDEX`.
- **"Camera N opened but delivered no frame"**: the camera is in use by another
  app (video calls), asleep, or a Continuity Camera iPhone that is locked or out of
  range. Close the other app, or choose another index.
- **"The camera stopped delivering frames"** while running: the camera was
  disconnected or taken by another app. Brief dropouts (under about a second) are
  retried automatically.

## Development documentation

Phase-by-phase notes (roadmap, changes, tests, findings and limitations) are in [docs/README.md](docs/README.md).

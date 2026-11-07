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

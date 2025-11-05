# Shoplifting Detection using YOLO Pose Estimation

This project implements a shoplifting detection system using YOLO pose estimation and XGBoost classification.

## Features

- Real-time pose detection using YOLOv8
- Behavior classification using XGBoost
- Support for video file processing
- Real-time visualization with OpenCV

## Requirements

- Python 3.11+
- OpenCV
- Ultralytics YOLO
- XGBoost
- Pandas
- NumPy
- cvzone

## Project Structure

- `main.py`: Main application for real-time detection
- `datset.py`: Dataset creation and preprocessing
- `model.py`: XGBoost model training
- `Normal.py`: Normal behavior data collection
- `Suspicious.py`: Suspicious behavior data collection

## Installation

1. Clone the repository:
```bash
git clone [your-repo-url]
cd [repo-name]
```

2. Create and activate virtual environment:
```bash
python -m venv venv
# Windows
.\venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Train the model:
```bash
python datset.py
python model.py
```

2. Run detection:
```bash
python main.py
```

Press 'q' to quit the application.

## License

[Your chosen license]

## Contributing

Feel free to open issues and pull requests!
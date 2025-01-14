# Turbine Flow Rate Tracker

A Python application for tracking turbine rotation and calculating flow rates from video recordings. The application uses simple pixel tracking to detect blade passages and compute real-time flow rates **using the turbine radius and blade area**.

## Features

- Video loading and automatic repair of corrupted files
- Interactive pixel selection for tracking
- Real-time flow rate calculation
- Flow rate visualization with live plotting
- Data export to CSV
- Spike filtering and measurement validation
- Support for slow-motion video analysis

## Installation

1. Clone this repository
2. Install required dependencies:
```bash
pip install -r requirements.txt
```

Note: For the video repair functionality to work, GPAC's MP4Box needs to be installed:
- Windows: Install GPAC from https://gpac.wp.imt.fr/downloads/
- Linux: `sudo apt-get install gpac`
- macOS: `brew install gpac`

## Usage

1. Run the application:
```bash
python main.py
```

2. Use the interface to:
   - Load a video file
   - Select a tracking point on the turbine
   - Adjust slow motion ratio if needed
   - Start analysis
   - Export data and graphs

## Configuration

Key settings can be adjusted in `config.py`:
- Turbine physical parameters (blade area, radius)
- Video analysis thresholds
- Spike filtering parameters
- Graph display settings

## Data Output

- CSV file with timestamps and flow rates
- PNG graph of flow rate over time
- Statistical analysis including mean and standard deviation

## Requirements

- Python 3.7+
- PyQt6
- OpenCV
- NumPy
- Matplotlib
- GPAC (for video repair functionality)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

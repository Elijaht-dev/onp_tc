"""Configuration settings for the Turbine Tracker application."""

import os

# Paths
MP4BOX_PATH = r"C:\Program Files\GPAC\mp4box.exe" if os.name == 'nt' else "MP4Box"

# Turbine physical parameters
BLADE_AREA = 0.65 / 10000.0  # Convert from cm² to m²
BLADE_RADIUS = 0.019  # Radius in meters (1.9 cm)

# Unit conversion constants
MS_TO_S = 1000.0  # Milliseconds to seconds
M3S_TO_MLS = 1000000.0  # m³/s to ml/s

# Default UI values
DEFAULT_SLOW_MOTION = 8.5  # 1s real = 8.5s video
WINDOW_SIZE = (1024, 768)
WINDOW_POSITION = (100, 100)

# Video frame analysis
DARK_THRESHOLD = 100  # Threshold for black detection
MAX_COLOR_VALUE = 150  # Maximum color value for black detection
TARGET_SELECTION_WIDTH = 800  # Width for selection window

# Spike filtering parameters
FILTER_WINDOW_SIZE = 7
OUTLIER_THRESHOLD = 2.5
MIN_FLOW_RATE = 0.0001  # ml/s
MAX_FLOW_RATE = 10000.0  # ml/s
MAX_RATE_CHANGE = 100.0  # ml/s

# Graph settings
PLOT_BUFFER_SIZE = 1000  # Maximum number of points to show on plot
MARKER_SIZE = 2

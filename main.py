import sys
import cv2
from PyQt6.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, 
                            QWidget, QLabel, QFileDialog, QDoubleSpinBox)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from collections import deque
from config import *

class TurbineAnalyzer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setGeometry(*WINDOW_POSITION, *WINDOW_SIZE)
        self.setWindowTitle("Turbine Flow Analyzer")
        
        self.video_path = None
        self.selected_point = None
        self.is_analyzing = False
        self.revolution_times = deque()
        self.flow_rates = []
        self.time_points = []
        self.total_volume = 0  # Add total volume tracker
        self.total_volume_label = None  # Add label reference
        self.rpm_values = []  # Add RPM tracking
        
        self.setup_ui()
        
    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Video selection
        self.select_button = QPushButton("Select Video")
        self.select_button.clicked.connect(self.select_video)
        layout.addWidget(self.select_button)
        
        # Slow motion factor input
        self.slow_motion_input = QDoubleSpinBox()
        self.slow_motion_input.setValue(DEFAULT_SLOW_MOTION)
        self.slow_motion_input.setRange(1, 100)
        layout.addWidget(QLabel("Slow Motion Factor:"))
        layout.addWidget(self.slow_motion_input)
        
        # Add total volume display before the graph
        self.total_volume_label = QLabel("Total Volume: 0.00 ml")
        layout.addWidget(self.total_volume_label)
        
        # Graph
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        # Start button
        self.start_button = QPushButton("Start Analysis")
        self.start_button.clicked.connect(self.start_analysis)
        layout.addWidget(self.start_button)
        
        # Add save button after the start button
        self.save_button = QPushButton("Save Data")
        self.save_button.clicked.connect(self.save_data)
        layout.addWidget(self.save_button)
        
    def select_video(self):
        self.video_path = QFileDialog.getOpenFileName(self, "Select Video")[0]
        if self.video_path:
            self.select_tracking_point()
    
    def select_tracking_point(self):
        cap = cv2.VideoCapture(self.video_path)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (TARGET_SELECTION_WIDTH, 
                             int(TARGET_SELECTION_WIDTH * frame.shape[0] / frame.shape[1])))
            self.selected_point = None
            
            def mouse_callback(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN:
                    self.selected_point = (x, y)
                    cv2.destroyWindow("Select Point")
            
            cv2.imshow("Select Point", frame)
            cv2.setMouseCallback("Select Point", mouse_callback)
            cv2.waitKey(0)
            cap.release()
    
    def start_analysis(self):
        if not self.video_path or not self.selected_point:
            return
            
        self.is_analyzing = True
        self.revolution_times.clear()
        self.flow_rates = []
        self.time_points = []
        self.total_volume = 0  # Reset total volume at start
        self.total_volume_label.setText("Total Volume: 0.00 ml")
        self.rpm_values = []  # Reset RPM values
        
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_time = 1000.0 / fps  # ms per frame
        current_time = 0
        last_black_start = None
        is_black = False
        revolutions = 0
        
        while cap.isOpened() and self.is_analyzing:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame = cv2.resize(frame, (TARGET_SELECTION_WIDTH, 
                             int(TARGET_SELECTION_WIDTH * frame.shape[0] / frame.shape[1])))
            pixel_color = frame[self.selected_point[1], self.selected_point[0]]
            
            # Detect black regions
            is_current_black = all(c <= MAX_COLOR_VALUE for c in pixel_color)
            
            if is_current_black and not is_black:
                if last_black_start is not None:
                    revolution_time = current_time - last_black_start
                    self.revolution_times.append(revolution_time)
                    revolutions += 1
                last_black_start = current_time
            
            is_black = is_current_black
            current_time += frame_time
            
            # Calculate metrics every real second
            real_time = current_time / (MS_TO_S * self.slow_motion_input.value())
            if len(self.time_points) == 0 or real_time - self.time_points[-1] >= 1.0:
                if self.revolution_times:
                    avg_revolution_time = sum(self.revolution_times) / len(self.revolution_times)
                    
                    # Debug information
                    print(f"Average time between black detections: {avg_revolution_time:.2f} ms")
                    print(f"Number of detections in this second: {len(self.revolution_times)}")
                    
                    # Nouvelle méthode de calcul RPM :
                    # 1. avg_revolution_time est en ms pour 1/4 de tour
                    # 2. Un tour complet = 4 * avg_revolution_time ms
                    # 3. Nombre de tours par seconde = 1000 / (4 * avg_revolution_time)
                    # 4. RPM = tours par seconde * 60 * facteur slow motion
                    rpm = (1000.0 / (1.25 * avg_revolution_time)) * 60.0 * self.slow_motion_input.value()
                    
                    print(f"Calculated RPM: {rpm:.2f}")
                    
                    flow_rate = self.calculate_flow_rate(rpm)
                    print(f"Calculated flow rate: {flow_rate:.2f} ml/s")
                    
                    self.flow_rates.append(flow_rate)
                    self.time_points.append(real_time)
                    self.rpm_values.append(rpm)  # Store RPM values
                    self.update_graph()
                    
                    # Calculate volume for this second and add to total
                    volume_this_second = flow_rate  # ml/s * 1s = ml
                    self.total_volume += volume_this_second
                    self.total_volume_label.setText(f"Total Volume: {self.total_volume:.2f} ml")
                    
                    self.revolution_times.clear()
            
            cv2.imshow("Analysis", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
        self.is_analyzing = False
    
    def calculate_flow_rate(self, rpm):
        """Calculates fluid flow rate based on turbine RPM."""
        # Convert RPM to radians per second (2π rad/rev × rev/min × 1min/60sec)
        omega = (2 * np.pi * rpm) / 60.0
        
        # Calculate fluid velocity (m/s)
        fluid_velocity = omega * BLADE_RADIUS
        
        # Calculate flow rate (m³/s) and convert to ml/s
        flow_rate = fluid_velocity * BLADE_AREA * M3S_TO_MLS
        
        return flow_rate
    
    def update_graph(self):
        self.ax.clear()
        self.ax.plot(self.time_points, self.flow_rates, 'b-', marker='o', markersize=MARKER_SIZE)
        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('Flow Rate (ml/s)')
        self.ax.set_title(f'Turbine Flow Rate Over Time\nTotal Volume: {self.total_volume:.2f} ml')
        self.canvas.draw()
    
    def save_data(self):
        """Save the analysis data to a CSV file."""
        if not self.flow_rates or not self.time_points:
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Data",
            "",
            "CSV Files (*.csv)"
        )
        
        if file_path:
            if not file_path.endswith('.csv'):
                file_path += '.csv'
                
            with open(file_path, 'w', newline='') as csvfile:
                # Write header
                csvfile.write('Time (s);Flow Rate (ml/s);RPM\n')
                
                # Write data with ; as delimiter and , for decimals
                for t, f, r in zip(self.time_points, self.flow_rates, self.rpm_values):
                    time_str = f'{t:.2f}'.replace('.', ',')
                    flow_str = f'{f:.2f}'.replace('.', ',')
                    rpm_str = f'{r:.2f}'.replace('.', ',')
                    csvfile.write(f'{time_str};{flow_str};{rpm_str}\n')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = TurbineAnalyzer()
    window.show()
    sys.exit(app.exec())
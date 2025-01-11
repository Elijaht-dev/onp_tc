import sys
import cv2
import subprocess
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, 
                            QWidget, QLabel, QFileDialog, QDoubleSpinBox)
from PyQt6.QtCore import QTimer
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from collections import deque

class TurbineTracker(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Suivi de Vitesse de Turbine")
        self.tracking = False
        self.last_color = None
        self.revolution_count = 0
        self.last_time = 0
        self.selected_pos = None
        self.video_cap = None
        self.current_frame = None
        self.frame_count = 0
        self.current_frame_pos = 0
        self.selecting = False
        self.start_time = None  # Add this after other instance variables
        self.mp4box_path = r"C:\Program Files\GPAC\mp4box.exe"  # Add path to MP4Box executable if needed
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Create UI elements
        self.load_button = QPushButton("Charger Vidéo")
        self.load_button.clicked.connect(self.load_video)
        
        self.select_button = QPushButton("Sélectionner Pixel")
        self.select_button.clicked.connect(self.start_selection)
        self.select_button.setEnabled(False)
        
        self.track_button = QPushButton("Démarrer le Suivi")
        self.track_button.setEnabled(False)
        self.track_button.clicked.connect(self.toggle_tracking)
        
        self.info_label = QLabel("Chargez d'abord une vidéo")
        self.speed_label = QLabel("Vitesse: 0 tr/min")
        
        # Add widgets to layout
        layout.addWidget(self.load_button)
        layout.addWidget(self.select_button)
        layout.addWidget(self.track_button)
        layout.addWidget(self.info_label)
        layout.addWidget(self.speed_label)
        
        # Setup timer for tracking
        self.timer = QTimer()
        self.timer.timeout.connect(self.track_pixel)
        self.timer.setInterval(50)  # 20 fps
        
        self.setGeometry(100, 100, 300, 200)
        
        # Add configuration parameters
        self.slow_motion_factor = 1.0  # e.g., 8x slow motion = 8.0
        self.blade_area = 0.0  # m²
        self.flow_coefficient = 0.0  # dimensionless
        
        # Data for plotting
        self.time_points = deque(maxlen=100)
        self.flow_points = deque(maxlen=100)
        
        # Create figure for plotting
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        
        # Add configuration inputs
        self.slow_motion_input = QDoubleSpinBox()
        self.slow_motion_input.setRange(1, 1000)
        self.slow_motion_input.setValue(1.0)
        self.slow_motion_input.setPrefix("Facteur Ralenti: ")
        
        self.area_input = QDoubleSpinBox()
        self.area_input.setRange(0, 100)
        self.area_input.setValue(0.0)
        self.area_input.setPrefix("Surface Pale (m²): ")
        
        self.coef_input = QDoubleSpinBox()
        self.coef_input.setRange(0, 1)
        self.coef_input.setValue(0.0)
        self.coef_input.setPrefix("Coefficient de Débit: ")
        
        self.flow_label = QLabel("Débit: 0.0 m³/s")
        
        # Add new widgets to layout
        layout.addWidget(self.slow_motion_input)
        layout.addWidget(self.area_input)
        layout.addWidget(self.coef_input)
        layout.addWidget(self.flow_label)
        layout.addWidget(self.canvas)
    
    def fix_video(self, video_path):
        """Fix problematic MP4 files using MP4Box"""
        try:
            fixed_path = f"{os.path.splitext(video_path)[0]}_fixed.mp4"
            subprocess.run([self.mp4box_path, '-add', video_path, '-new', fixed_path], 
                         check=True, capture_output=True)
            return fixed_path
        except Exception as e:
            self.info_label.setText(f"Erreur de conversion: {str(e)}")
            return None

    def load_video(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Sélectionner un Fichier Vidéo", "", 
                                                "Fichiers Vidéo (*.mp4 *.avi *.mov *.mkv *.wmv)")
        if filename:
            if self.video_cap is not None:
                self.video_cap.release()

            # First try to open the original video
            self.video_cap = cv2.VideoCapture(filename)
            if not self.video_cap.isOpened():
                self.info_label.setText("Tentative de réparation du fichier vidéo...")
                # Try to fix the video
                fixed_file = self.fix_video(filename)
                if fixed_file and os.path.exists(fixed_file):
                    self.video_cap = cv2.VideoCapture(fixed_file)
                    if not self.video_cap.isOpened():
                        self.info_label.setText("Erreur: Échec de la réparation de la vidéo")
                        self.video_cap = None
                        return
                else:
                    self.info_label.setText("Erreur: Échec de la conversion de la vidéo")
                    self.video_cap = None
                    return

            # Continue with normal video loading
            self.frame_count = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if self.frame_count <= 0:
                self.info_label.setText("Erreur: Impossible de déterminer la longueur de la vidéo")
                self.video_cap.release()
                self.video_cap = None
                return

            # Verify video can be read
            ret, frame = self.video_cap.read()
            if ret and frame is not None:
                self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.select_button.setEnabled(True)
                self.info_label.setText("Vidéo chargée. Sélectionnez un pixel à suivre")
                self.current_frame_pos = 0
                return

            self.info_label.setText("Erreur: Impossible de lire la vidéo")
            self.video_cap.release()
            self.video_cap = None
    
    @staticmethod
    def mouse_callback(event, x, y, flags, param):
        self = param
        if event == cv2.EVENT_LBUTTONDOWN and self.selecting:
            self.selected_pos = (x, y)
            self.selecting = False
            self.track_button.setEnabled(True)
            self.info_label.setText(f"Position sélectionnée: {self.selected_pos}")
            
            # Draw marker on the frame and show it
            marked_frame = self.current_frame.copy()
            cv2.circle(marked_frame, self.selected_pos, 5, (0, 255, 0), -1)
            cv2.imshow("Sélectionner Pixel", marked_frame)
            cv2.waitKey(1000)
            cv2.destroyAllWindows()

    def start_selection(self):
        if self.video_cap is None:
            return
            
        # Close any existing windows
        cv2.destroyAllWindows()
        
        ret, frame = self.video_cap.read()
        if ret:
            self.current_frame = frame
            window_name = "Sélectionner Pixel"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.setMouseCallback(window_name, self.mouse_callback, self)
            cv2.imshow(window_name, frame)
            self.selecting = True
            self.info_label.setText("Cliquez sur le pixel que vous souhaitez suivre")
            cv2.waitKey(0)  # Wait for user input
            cv2.destroyAllWindows()
    
    def toggle_tracking(self):
        if not self.tracking:
            if self.selected_pos is None:
                self.info_label.setText("Veuillez d'abord sélectionner un pixel")
                return
            self.start_time = None  # Reset start time when starting new tracking
            self.time_points.clear()  # Clear previous data
            self.flow_points.clear()
            self.tracking = True
            self.track_button.setText("Arrêter le Suivi")
            self.timer.start()
        else:
            self.tracking = False
            self.track_button.setText("Démarrer le Suivi")
            self.timer.stop()
            cv2.destroyAllWindows()
    
    def track_pixel(self):
        if self.selected_pos is None or self.video_cap is None:
            return

        if not self.video_cap.isOpened():
            self.info_label.setText("Erreur: Source vidéo perdue")
            self.toggle_tracking()
            return

        self.current_frame_pos = int(self.video_cap.get(cv2.CAP_PROP_POS_FRAMES))
        
        # Check if we're at the end of the video
        if self.current_frame_pos >= self.frame_count - 1:
            self.info_label.setText("Fin de la vidéo atteinte")
            self.toggle_tracking()
            return

        ret, frame = self.video_cap.read()
        
        if not ret or frame is None:
            if self.current_frame_pos < self.frame_count - 1:
                # Try to recover by seeking to next frame
                self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_pos + 1)
                ret, frame = self.video_cap.read()
            
            if not ret or frame is None:
                self.info_label.setText("Erreur: Impossible de lire la trame vidéo")
                self.toggle_tracking()
                return
        
        if ret:
            # Draw marker on tracking window
            tracking_frame = frame.copy()
            cv2.circle(tracking_frame, self.selected_pos, 5, (0, 255, 0), -1)
            
            color = frame[int(self.selected_pos[1]), int(self.selected_pos[0])]
            
            # Define color thresholds for black and yellow
            is_black = (np.mean(color) < 100 and  # Overall darkness
                       np.max(color) < 150)        # No channel too bright
            
            is_yellow = (color[0] < 150 and       # Low blue
                        color[1] > 150 and        # High green
                        color[2] > 150)           # High red
            
            if self.last_color is not None:
                if (is_black and not self.last_color) or (not is_black and self.last_color):
                    current_time = self.video_cap.get(cv2.CAP_PROP_POS_MSEC)
                    if self.last_time != 0:
                        if self.revolution_count % 4 == 0:
                            time_diff = (current_time - self.last_time) / 1000.0 * self.slow_motion_factor
                            rpm = 60.0 / (time_diff * 4)
                            
                            # Update parameters from inputs
                            self.slow_motion_factor = self.slow_motion_input.value()
                            self.blade_area = self.area_input.value()
                            self.flow_coefficient = self.coef_input.value()
                            
                            # Calculate and display flow rate
                            flow_rate = self.calculate_flow_rate(rpm)
                            self.speed_label.setText(f"Vitesse: {rpm:.1f} tr/min")
                            self.flow_label.setText(f"Débit: {flow_rate:.3f} m³/s")
                            
                            # Update graph
                            self.update_graph(current_time/1000.0, flow_rate)
                    
                    self.last_time = current_time
                    self.revolution_count += 1
                    
            self.last_color = is_black
            cv2.imshow("Suivi", tracking_frame)

    def calculate_flow_rate(self, rpm):
        # Convert RPM to radians per second
        omega = (rpm * 2 * np.pi) / 60
        
        # Calculate flow rate (Q = ω * A * Cf)
        flow_rate = omega * self.blade_area * self.flow_coefficient
        
        return flow_rate
        
    def update_graph(self, time, flow_rate):
        if self.start_time is None:
            self.start_time = time
        
        elapsed_time = time - self.start_time
        self.time_points.append(elapsed_time)
        self.flow_points.append(flow_rate)
        
        self.ax.clear()
        self.ax.plot(list(self.time_points), list(self.flow_points))
        self.ax.set_xlabel('Temps écoulé (s)')
        self.ax.set_ylabel('Débit (m³/s)')
        self.ax.set_title('Débit en Temps Réel')
        self.canvas.draw()

    def __del__(self):
        # Cleanup
        if self.video_cap is not None:
            self.video_cap.release()
        cv2.destroyAllWindows()

def main():
    app = QApplication(sys.argv)
    window = TurbineTracker()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
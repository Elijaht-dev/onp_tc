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
        self.last_calculation_time = 0  # Temps réel de la dernière calcul de débit
        
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
        
        self.info_label = QLabel("Chargez d'abord une vidéo")
        
        # Add widgets to layout
        layout.addWidget(self.load_button)
        layout.addWidget(self.select_button)
        layout.addWidget(self.info_label)
        
        self.setGeometry(100, 100, 1024, 768)  # Adjusted window size to 1024x768
        
        # Add configuration parameters
        self.slow_motion_factor = 1.0  # e.g., 8x slow motion = 8.0
        self.blade_area = 0.0065  # Surface d'une pale en m²
        
        # Data for plotting
        self.time_points = deque(maxlen=1000)  # Augmenté de 100 à 1000
        self.flow_points = deque(maxlen=1000)  # Augmenté de 100 à 1000
        
        # Create figure for plotting
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        
        # Add configuration inputs with more precision
        self.slow_motion_input = QDoubleSpinBox()
        self.slow_motion_input.setDecimals(2)  # Beaucoup plus de décimales
        self.slow_motion_input.setValue(8.5)
        self.slow_motion_input.setPrefix("1s réelle = ")
        self.slow_motion_input.setSuffix(" s vidéo")
        
        self.area_input = QDoubleSpinBox()
        self.area_input.setDecimals(6)  # Beaucoup plus de décimales
        self.area_input.setValue(self.blade_area)
        self.area_input.setPrefix("Surface Pale (m²): ")  # Correction de l'unité
        
        self.flow_label = QLabel("Débit: 0.000 ml/s")  # Plus de décimales
        
        # Add new widgets to layout
        layout.addWidget(self.slow_motion_input)
        layout.addWidget(self.area_input)
        layout.addWidget(self.flow_label)
        layout.addWidget(self.canvas)
        
        # Constants for turbine avec unités corrigées
        self.BLADE_AREA = 0.65 / 10000.0  # 0.65 cm² -> m²
        self.BLADE_RADIUS = 0.019  # 1.9 cm -> 0.019 m
        
        # Paramètres de conversion
        self.MS_TO_S = 1000.0  # Conversion ms -> s
        self.M3S_TO_MLS = 1000000.0  # Conversion m³/s -> ml/s
        
        # Update area_input default value
        self.area_input.setValue(self.BLADE_AREA)
        
        # Update radius display with more precision
        self.radius_label = QLabel(f"Rayon de la turbine: {self.BLADE_RADIUS*100} cm")
        layout.addWidget(self.radius_label)

        # Ajout du bouton de sauvegarde
        self.save_button = QPushButton("Sauvegarder Données")
        self.save_button.clicked.connect(self.save_data)
        layout.addWidget(self.save_button)
        
        # Liste pour stocker toutes les données
        self.all_flow_rates = []
        self.all_times = []

        # Ajout du bouton de traitement rapide
        self.process_button = QPushButton("Traitement Rapide")
        self.process_button.setEnabled(False)
        self.process_button.clicked.connect(self.process_video)
        layout.addWidget(self.process_button)

        # Ajouter après self.all_times = []
        self.current_revolution_flows = []  # Pour stocker les débits du tour actuel
        self.all_revolution_flows = []      # Pour stocker les débits moyens par tour
        self.first_revolution_completed = False

        # Ajout des variables de suivi des tours
        self.complete_revolutions = 0  # Compte des tours complets
        self.measurement_started = False  # Indique si on a commencé à mesurer

        # Add spike filtering parameters with better defaults
        self.window_size = 7  # Increased window size for better averaging
        self.outlier_threshold = 2.5  # Standard deviations for outlier detection
        self.min_flow_rate = 0.0001    # Valeur minimum plus petite
        self.max_flow_rate = 10000.0   # Valeur maximum plus grande
        self.max_rate_change = 100.0   # Tolérance plus grande pour les changements
        self.last_valid_flow = None
        self.flow_buffer = deque(maxlen=self.window_size)

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
            
            # Redimensionner la fenêtre de sélection
            height, width = frame.shape[:2]
            target_width = 800
            target_height = int(target_width * height / width)
            cv2.resizeWindow(window_name, target_width, target_height)
            
            cv2.setMouseCallback(window_name, self.mouse_callback, self)
            cv2.imshow(window_name, frame)
            self.selecting = True
            self.info_label.setText("Cliquez sur le pixel que vous souhaitez suivre")
            cv2.waitKey(0)  # Wait for user input
            cv2.destroyAllWindows()
            # Activer le bouton de traitement rapide après la sélection
            self.process_button.setEnabled(True)

    def calculate_flow_rate(self, rpm):
        # Calcul vitesse angulaire (rad/s)
        omega = (2 * np.pi * rpm) / 60.0
        
        # Calcul vitesse linéaire (m/s)
        fluid_velocity = omega * self.BLADE_RADIUS
        
        # Calcul débit avec conversion explicite (m³/s -> ml/s)
        flow_rate = fluid_velocity * self.BLADE_AREA * self.M3S_TO_MLS
        
        return flow_rate
        
    def update_graph(self, time, flow_rate):
        """Met à jour le graphique avec les nouvelles données"""
        if self.start_time is None:
            self.start_time = time
        
        elapsed_time = time - self.start_time
        self.time_points.append(elapsed_time)
        self.flow_points.append(flow_rate)
        
        # Mise à jour du graphique
        self.ax.clear()
        self.ax.plot(list(self.time_points), list(self.flow_points), 'b.', markersize=2)  # Points pour chaque mesure
        self.ax.plot(list(self.time_points), list(self.flow_points), 'b-', alpha=0.5)     # Ligne continue
        
        # Paramètres du graphique avec nouvelle unité
        self.ax.set_xlabel('Temps (s)')
        self.ax.set_ylabel('Débit (ml/s)')
        self.ax.set_title('Débit instantané')
        self.ax.grid(True)
        
        # Forcer la mise à jour immédiate
        self.canvas.draw()
        self.canvas.flush_events()

    def save_data(self):
        """Sauvegarde les données de débit et le graphique"""
        if not self.all_flow_rates:
            self.info_label.setText("Pas de données à sauvegarder")
            return
            
        # Sauvegarder les données dans un fichier CSV
        filename, _ = QFileDialog.getSaveFileName(self, "Sauvegarder les données", "", 
                                                "Fichiers CSV (*.csv);;Tous les fichiers (*)")
        if filename:
            if not filename.endswith('.csv'):
                filename += '.csv'
            
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write("Temps (s);Débit (ml/s);Vitesse (tr/min)\n")  # Ajout de la vitesse
                    for t, q in zip(self.all_times, self.all_flow_rates):
                        v = q / (self.blade_area * self.BLADE_RADIUS)  # Calcul de la vitesse
                        line = f"{t};{q};{v}\n"  # Suppression des limitations de décimales
                        f.write(line)
                
                # Sauvegarder le graphique
                graph_filename = filename.replace('.csv', '_graph.png')
                self.figure.savefig(graph_filename)
                
                self.info_label.setText("Données sauvegardées avec succès")
            except Exception as e:
                self.info_label.setText(f"Erreur lors de la sauvegarde: {str(e)}")

    def process_video(self):
        if self.selected_pos is None or self.video_cap is None:
            return

        self.info_label.setText("Traitement en cours...")
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.time_points.clear()
        self.flow_points.clear()
        self.all_flow_rates = []
        self.all_times = []
        self.start_time = None
        
        last_color = None
        last_time = 0
        revolution_count = 0
        complete_revolutions = 0
        measurement_started = False

        while True:
            ret, frame = self.video_cap.read()
            if not ret:
                break

            color = frame[int(self.selected_pos[1]), int(self.selected_pos[0])]

            is_black = (np.mean(color) < 100 and np.max(color) < 150)
            
            if last_color is not None:
                if (is_black and not last_color) or (not is_black and last_color):
                    current_time = self.video_cap.get(cv2.CAP_PROP_POS_MSEC)
                    real_current_time = current_time / 1000.0 / self.slow_motion_input.value()
                    
                    if last_time != 0:
                        if revolution_count % 4 == 0:
                            complete_revolutions += 1
                            
                            if complete_revolutions >= 2:
                                measurement_started = True
                                
                                real_time_diff = (current_time - last_time) / 1000.0
                                actual_time_diff = real_time_diff / self.slow_motion_input.value()
                                
                                rpm = 60.0 / (actual_time_diff * 4)
                                flow_rate = self.calculate_flow_rate(rpm)
                                
                                flow_rate = self.filter_spike(flow_rate)
                                
                                self.all_times.append(real_current_time)
                                self.all_flow_rates.append(flow_rate)
                                self.update_graph(real_current_time, flow_rate)
                    
                    last_time = current_time
                    revolution_count += 1
            last_color = is_black

        if self.all_flow_rates:
            avg_flow = np.mean(self.all_flow_rates)
            std_flow = np.std(self.all_flow_rates)
            
            self.flow_label.setText(f"Débit moyen: {avg_flow:.3f} ± {std_flow:.3f} ml/s")
            self.info_label.setText(f"Traitement terminé - {len(self.all_flow_rates)} mesures")
        else:
            self.info_label.setText("Aucune donnée n'a pu être analysée")

    def filter_spike(self, flow_rate):
        """
        Enhanced spike filtering using multiple criteria:
        1. Basic range validation
        2. Rate of change validation
        3. Statistical outlier detection
        """
        # Basic range check
        if not self.min_flow_rate <= flow_rate <= self.max_flow_rate:
            return self.last_valid_flow if self.last_valid_flow is not None else self.min_flow_rate

        # Rate of change check
        if self.last_valid_flow is not None:
            rate_change = abs(flow_rate - self.last_valid_flow)
            if rate_change > self.max_rate_change:
                return self.last_valid_flow

        # Statistical outlier detection
        self.flow_buffer.append(flow_rate)
        if len(self.flow_buffer) >= 3:  # Need at least 3 points for statistics
            mean = np.mean(self.flow_buffer)
            std = np.std(self.flow_buffer)
            
            # Check if current value is within acceptable range
            if abs(flow_rate - mean) > self.outlier_threshold * std:
                return self.last_valid_flow if self.last_valid_flow is not None else mean
            
            # If we have enough points, use median filtering
            if len(self.flow_buffer) >= 5:
                sorted_values = sorted(self.flow_buffer)
                median = sorted_values[len(sorted_values)//2]
                if abs(flow_rate - median) > self.max_rate_change:
                    return median

        self.last_valid_flow = flow_rate
        return flow_rate

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
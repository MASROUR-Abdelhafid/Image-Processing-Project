import sys
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFileDialog, QLabel, QAction, QMenuBar, QPushButton, QSlider, QToolBar, QFrame
from PyQt5.QtGui import QPixmap, QImage, QIcon, QColor, QFont
from PyQt5.QtCore import Qt, QSize, QPropertyAnimation, QEasingCurve

# Ajouter le dossier principal du projet au chemin de recherche de Python
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

# Importer les modules de traitement d'image
from image_utils.filters_low import *
from image_utils.filters_high import *
from image_utils.transformations import *
from image_utils.morphology import *
from image_utils.detection import *
from image_utils.histogram import *
from image_utils.frequency import *

# Mise à jour des cadres pour mieux gérer l'affichage
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("ImageMaster Pro")
        self.setGeometry(100, 100, 1400, 900)

        # Appliquer un style moderne
        self.setStyleSheet(self.get_stylesheet())

        # Créer un widget central
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        # Layout principal
        self.layout = QVBoxLayout(self.central_widget)
        
        # Créer un conteneur pour afficher les images
        self.image_container = QHBoxLayout()
        self.layout.addLayout(self.image_container)

        # Créer un cadre pour l'image d'entrée
        self.original_frame = QFrame(self)
        self.original_frame.setFrameShape(QFrame.StyledPanel)
        self.original_frame.setLineWidth(2)
        self.original_frame.setStyleSheet("QFrame { background-color: #2a2a3f; border-radius: 10px; padding: 15px; }")
        self.original_image_section = QVBoxLayout(self.original_frame)

        # Titre pour l'image d'entrée
        self.original_title = QLabel("Image Entrée", self)
        self.original_title.setAlignment(Qt.AlignCenter)
        self.original_title.setStyleSheet("font-size: 18px; color: #ffffff; font-weight: bold;")
        self.original_image_section.addWidget(self.original_title)

        # Label pour afficher l'image d'entrée
        self.original_image_label = QLabel(self)
        self.original_image_label.setAlignment(Qt.AlignCenter)
        self.original_image_label.setFixedSize(750, 550)  # Fixer une taille plus grande pour l'image
        self.original_image_section.addWidget(self.original_image_label)

        # Créer un cadre pour l'image de sortie
        self.filtered_frame = QFrame(self)
        self.filtered_frame.setFrameShape(QFrame.StyledPanel)
        self.filtered_frame.setLineWidth(2)
        self.filtered_frame.setStyleSheet("QFrame { background-color: #2a2a3f; border-radius: 10px; padding: 15px; }")
        self.filtered_image_section = QVBoxLayout(self.filtered_frame)

        # Titre pour l'image de sortie
        self.filtered_title = QLabel("Image Sortie", self)
        self.filtered_title.setAlignment(Qt.AlignCenter)
        self.filtered_title.setStyleSheet("font-size: 18px; color: #ffffff; font-weight: bold;")
        self.filtered_image_section.addWidget(self.filtered_title)

        # Label pour afficher l'image de sortie
        self.filtered_image_label = QLabel(self)
        self.filtered_image_label.setAlignment(Qt.AlignCenter)
        self.filtered_image_label.setFixedSize(750, 550)  # Fixer une taille plus grande pour l'image filtrée
        self.filtered_image_section.addWidget(self.filtered_image_label)

        # Ajouter les cadres au conteneur
        self.image_container.addWidget(self.original_frame)
        self.image_container.addWidget(self.filtered_frame)

        # Initialiser les variables
        self.original_image = None
        self.filtered_image = None

        # Créer la barre de menu
        self.init_menu()

        # Ajouter des boutons avec des icônes
        self.init_toolbar()

    # Mise à jour du style
    def get_stylesheet(self):
        return """
            QMainWindow {
                background-color: #1e1e2f;
            }
            QWidget {
                background-color: #2a2a3f;
                border-radius: 15px;
                padding: 15px;
            }
            QLabel {
                font-size: 18px;
                color: #ffffff;
                font-weight: bold;
                text-align: center;
                padding: 10px;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 15px 30px;
                border-radius: 10px;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
            QMenuBar {
                background-color: #1e1e2f;
                color: white;
                font-size: 20px;
                font-weight: bold;
            }
            QMenuBar::item {
                background: transparent;
                padding: 10px 20px;
            }
            QMenuBar::item:selected {
                background: #2a2a3f;
                border-radius: 5px;
            }
            QMenu {
                background-color: #2a2a3f;
                border-radius: 10px;
                padding: 10px;
            }
            QMenu::item {
                padding: 10px 20px;
                color: white;
                font-size: 14px;
            }
            QMenu::item:selected {
                background-color: #4CAF50;
                border-radius: 5px;
            }
            QSlider::groove:horizontal {
                background: #4CAF50;
                height: 8px;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #ffffff;
                width: 20px;
                height: 20px;
                margin: -6px 0;
                border-radius: 10px;
            }
        """


    def init_menu(self):
        menubar = self.menuBar()

        # Ajouter un menu "Fichier"
        file_menu = menubar.addMenu("Fichier")
        file_menu.addAction(self.create_action("Ouvrir", self.open_image, "open.png"))
        file_menu.addAction(self.create_action("Enregistrer", self.save_image, "save.png"))
        file_menu.addAction(self.create_action("Quitter", self.close, "exit.png"))

        # Ajouter un menu pour les filtres Pass-Bas
        filter_menu = menubar.addMenu("Filtres Pass-Bas")
        self.add_filter_actions(filter_menu, [
            ("Moyenneur 3x3", self.apply_filtre_moyenneur_3x3),
            ("Moyenneur 5x5", self.apply_filtre_moyenneur_5x5),
            ("Gaussien 3x3", self.apply_filtre_gaussien_3x3),
            ("Gaussien 5x5", self.apply_filtre_gaussien_5x5),
            ("Pyramidal", self.apply_filtre_pyramidal),
            ("Médiane", self.apply_filtre_mediane),
        ])

        # Ajouter un menu pour les filtres Pass-Haut
        passhaut_menu = menubar.addMenu("Filtres Pass-Haut")
        self.add_filter_actions(passhaut_menu, [
            ("Laplacien", self.apply_filtre_laplacien),
            ("Sobel", self.apply_filtre_sobel),
            ("Gradient", self.apply_filtre_gradient),
            ("Prewitt", self.apply_filtre_prewwitt),
            ("Robert", self.apply_filtre_robert),
        ])

        # Ajouter un menu pour les transformations
        transform_menu = menubar.addMenu("Transformations")
        self.add_filter_actions(transform_menu, [
            ("Inversion", self.apply_inversion),
            ("Binarisation", self.apply_binarisation),
            ("Contraste", self.apply_contraste),
            ("Division", self.apply_division),
            ("Niveaux de gris", self.apply_niveaux_de_gris),
            ("Hough", self.apply_hough_image),
            ("FFT", self.apply_fourier_transform),
        ])
        
        # Ajouter un menu "Filtres Fréquentiels"
        frequency_menu = menubar.addMenu("Filtres Fréquentiels")
        self.add_filter_actions(frequency_menu, [
            ("Passe-bas (FPB)", self.apply_lowpass_filter),
            ("Passe-haut (FPH)", self.apply_highpass_filter),
            ("Passe-bande (FPBB)", self.apply_bandpass_filter),
            ("Passe-haut-bande (FPHB)", self.apply_bandstop_filter),
        ])

        # Ajouter un menu pour les opérations morphologiques
        morpho_menu = menubar.addMenu("Morphologie")
        self.add_filter_actions(morpho_menu, [
            ("Erosion", self.apply_erosion),
            ("Dilatation", self.apply_dilatation),
            ("Ouverture", self.apply_ouverture),
            ("Fermeture", self.apply_fermeture),
            ("Contours Externes", self.apply_contours_externes),  # Contours externes
            ("Contours Internes", self.apply_contours_internes),  # Contours internes            
        ])

        # Ajouter un menu pour la détection des points d'intérêt
        detection_menu = menubar.addMenu("Détection")
        self.add_filter_actions(detection_menu, [
            ("Harris", self.apply_harris),
            ("Susan", self.apply_susan),
        ])

        # Ajouter un bouton pour afficher l'histogramme
        histogram_button = QAction("Afficher Histogramme", self)
        histogram_button.triggered.connect(self.display_histogram)
        menubar.addAction(histogram_button)

    def init_toolbar(self):
        toolbar = self.addToolBar("Outils")
        toolbar.setIconSize(QSize(32, 32))

        # Bouton pour ouvrir une image
        open_action = QAction(QIcon("open.png"), "Ouvrir", self)
        open_action.triggered.connect(self.open_image)
        toolbar.addAction(open_action)

        # Bouton pour enregistrer une image
        save_action = QAction(QIcon("save.png"), "Enregistrer", self)
        save_action.triggered.connect(self.save_image)
        toolbar.addAction(save_action)

        # Bouton pour quitter
        exit_action = QAction(QIcon("exit.png"), "Quitter", self)
        exit_action.triggered.connect(self.close)
        toolbar.addAction(exit_action)

    def create_action(self, text, func, icon_path=None):
        action = QAction(text, self)
        if icon_path:
            action.setIcon(QIcon(icon_path))
        action.triggered.connect(func)
        return action

    def add_filter_actions(self, menu, actions):
        for text, func in actions:
            menu.addAction(self.create_action(text, func))

    def apply_filter(self, filter_function):
        if self.original_image is not None:
            try:
                self.filtered_image = filter_function(self.original_image)
                self.display_image(self.original_image, self.filtered_image)
            except Exception as e:
                print(f"Erreur lors de l'application du filtre: {e}")

    def apply_filtre_moyenneur_3x3(self):
        self.apply_filter(filtre_moyenneur_3x3)

    def apply_filtre_moyenneur_5x5(self):
        self.apply_filter(filtre_moyenneur_5x5)

    def apply_filtre_gaussien_3x3(self):
        self.apply_filter(filtre_gaussien_3x3)

    def apply_filtre_gaussien_5x5(self):
        self.apply_filter(filtre_gaussien_5x5)

    def apply_filtre_pyramidal(self):
        self.apply_filter(filtre_pyramidal)

    def apply_filtre_mediane(self):
        self.apply_filter(filtre_mediane)

    def apply_filtre_laplacien(self):
        self.apply_filter(filtre_laplacien)

    def apply_filtre_sobel(self):
        self.apply_filter(filtre_sobel)

    def apply_filtre_gradient(self):
        self.apply_filter(filtre_gradient)

    def apply_filtre_prewwitt(self):
        self.apply_filter(filtre_prewwitt)

    def apply_filtre_robert(self):
        self.apply_filter(filtre_robert)

    def apply_inversion(self):
        self.apply_filter(inversion_image)

    def apply_binarisation(self):
        self.apply_filter(binarisation_image)

    def apply_contraste(self):
        self.apply_filter(contraste_image)

    def apply_division(self):
        self.apply_filter(division_image)

    def apply_niveaux_de_gris(self):
        self.apply_filter(niveaux_de_gris)

    def apply_hough_image(self):
        self.apply_filter(hough_image)

    def apply_fourier_transform(self):
     if self.original_image is not None:
        # Appliquer la transformation de Fourier et afficher le spectre
        magnitude_spectrum = fourier_transform(self.original_image)

        # Afficher le spectre dans l'image filtrée
        self.filtered_image = magnitude_spectrum
        self.display_image(self.original_image, self.filtered_image)



    def apply_lowpass_filter(self):
        if self.original_image is not None:
            # Appliquer un filtre passe-bas (FPB)
            cutoff = 30  # Fréquence de coupure, ajuste selon ton besoin
            # Appeler la fonction dans frequency.py
            self.filtered_image = apply_lowpass_filter(self.original_image, cutoff)
            self.display_image(self.original_image, self.filtered_image)

    def apply_highpass_filter(self):
        if self.original_image is not None:
            # Appliquer un filtre passe-haut (FPH)
            cutoff = 10  # Ajuste la fréquence de coupure selon ton besoin
            # Appeler la fonction dans frequency.py
            self.filtered_image = apply_highpass_filter(self.original_image, cutoff)
            self.display_image(self.original_image, self.filtered_image)

    def apply_bandpass_filter(self):
        if self.original_image is not None:
            # Appliquer un filtre passe-bande (FPBB)
            band = (20, 50)  # Ajuste la bande de fréquences selon ton besoin
            # Appeler la fonction dans frequency.py
            self.filtered_image = apply_bandpass_filter(self.original_image, band)
            self.display_image(self.original_image, self.filtered_image)

    def apply_bandstop_filter(self):
        if self.original_image is not None:
            # Appliquer un filtre passe-haut-bande (FPHB)
            band = (20, 50)  # Ajuste la bande de fréquences selon ton besoin
            # Appeler la fonction dans frequency.py
            self.filtered_image = apply_bandstop_filter(self.original_image, band)
            self.display_image(self.original_image, self.filtered_image)


    def apply_erosion(self):
        self.apply_filter(erosion_image)

    def apply_dilatation(self):
        self.apply_filter(dilatation_image)

    def apply_ouverture(self):
        self.apply_filter(ouverture_image)

    def apply_fermeture(self):
        self.apply_filter(fermeture_image)
    
    def apply_contours_externes(self):
        if self.original_image is not None:
            self.filtered_image = contours_externes(self.original_image)
            self.display_image(self.original_image, self.filtered_image)

    def apply_contours_internes(self):
        if self.original_image is not None:
            self.filtered_image = contours_internes(self.original_image)
            self.display_image(self.original_image, self.filtered_image)



    def apply_harris(self):
        self.apply_filter(harris_detection)

    def apply_susan(self):
        self.apply_filter(susan_detection)

    def display_histogram(self):
        if self.original_image is not None:
        # Afficher l'histogramme avec la fonction du fichier histogramme.py
            afficher_histogramme(self.original_image)


    def display_image(self, image, filtered_image=None):
        if image is not None:
            try:
                if len(image.shape) == 2:  # Image 2D (grayscale)
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

                if image.dtype != np.uint8:
                    image = np.uint8(np.clip(image, 0, 255))

                height, width, _ = image.shape
                bytes_per_line = 3 * width
                q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
                pixmap = QPixmap.fromImage(q_image)

                # Conserver la taille originale de l'image avant de l'appliquer aux labels
                original_pixmap = pixmap.scaled(self.original_image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.original_image_label.setPixmap(original_pixmap)

                if filtered_image is not None:
                    if len(filtered_image.shape) == 2:  # Image 2D (grayscale)
                        filtered_image = cv2.cvtColor(filtered_image, cv2.COLOR_GRAY2BGR)

                    if filtered_image.dtype != np.uint8:
                        filtered_image = np.uint8(np.clip(filtered_image, 0, 255))

                    height, width, _ = filtered_image.shape
                    bytes_per_line = 3 * width
                    q_filtered_image = QImage(filtered_image.data, filtered_image.shape[1], filtered_image.shape[0], bytes_per_line, QImage.Format_RGB888).rgbSwapped()
                    filtered_pixmap = QPixmap.fromImage(q_filtered_image)

                    # Redimensionner l'image filtrée en utilisant la taille de l'étiquette
                    filtered_pixmap_resized = filtered_pixmap.scaled(self.filtered_image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    self.filtered_image_label.setPixmap(filtered_pixmap_resized)

            except Exception as e:
                print(f"Erreur lors de l'affichage de l'image: {e}")

    def open_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Ouvrir une Image", "", "Images (*.png *.jpg *.bmp *.jpeg *.tiff *.gif)")
        if file_path:
            self.original_image = cv2.imread(file_path)
            self.display_image(self.original_image)

    def save_image(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Enregistrer l'Image", "", "Images (*.png *.jpg *.bmp *.jpeg *.tiff *.gif)")
        if file_path:
            cv2.imwrite(file_path, self.filtered_image)

# Lancer l'application PyQt5
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

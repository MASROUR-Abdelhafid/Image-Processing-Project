import sys
from PyQt5.QtWidgets import QApplication
from gui.main_window import MainWindow  # Assurez-vous d'importer la classe MainWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)  # Créer l'application Qt
    window = MainWindow()  # Créer une instance de la fenêtre principale
    window.show()  # Afficher la fenêtre principale
    sys.exit(app.exec_())  # Lancer l'application

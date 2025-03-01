import cv2 
import numpy as np
import matplotlib.pyplot as plt

# Calcul de l'histogramme de l'image
def histogramme_image(image):
    # Si l'image est en couleur (3 canaux)
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Calcul des histogrammes pour chaque canal (R, G, B)
        hist_r = cv2.calcHist([image], [0], None, [256], [0, 256])  # Rouge
        hist_g = cv2.calcHist([image], [1], None, [256], [0, 256])  # Vert
        hist_b = cv2.calcHist([image], [2], None, [256], [0, 256])  # Bleu
        return hist_r, hist_g, hist_b
    # Si l'image est en niveaux de gris (1 canal)
    elif len(image.shape) == 2:
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])  # Histogramme de l'image en niveaux de gris
        return hist
    else:
        raise ValueError("Image format non supporté")

# Affichage de l'histogramme de l'image
def afficher_histogramme(image):
    if len(image.shape) == 3 and image.shape[2] == 3:  # Image couleur (3 canaux)
        # Extraire les histogrammes pour chaque canal
        hist_r, hist_g, hist_b = histogramme_image(image)

        plt.figure(figsize=(15, 5))

        # Afficher l'histogramme du canal Rouge
        plt.subplot(1, 3, 1)
        plt.plot(hist_r, color='r')
        plt.title("Histogramme - Canal Rouge")
        plt.xlabel("Valeur des pixels")
        plt.ylabel("Fréquence")

        # Afficher l'histogramme du canal Vert
        plt.subplot(1, 3, 2)
        plt.plot(hist_g, color='g')
        plt.title("Histogramme - Canal Vert")
        plt.xlabel("Valeur des pixels")
        plt.ylabel("Fréquence")

        # Afficher l'histogramme du canal Bleu
        plt.subplot(1, 3, 3)
        plt.plot(hist_b, color='b')
        plt.title("Histogramme - Canal Bleu")
        plt.xlabel("Valeur des pixels")
        plt.ylabel("Fréquence")

        plt.tight_layout()
        plt.show()

    elif len(image.shape) == 2:  # Image en niveaux de gris (1 canal)
        hist = histogramme_image(image)
        plt.plot(hist, color='gray')
        plt.title("Histogramme de l'image en niveaux de gris")
        plt.xlabel("Niveaux de gris")
        plt.ylabel("Fréquence")
        plt.show()
    else:
        print("Erreur : Format d'image non pris en charge.")


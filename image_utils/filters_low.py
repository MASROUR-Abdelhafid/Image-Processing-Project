import cv2
import numpy as np

# Filtre Moyenneur 3x3
def filtre_moyenneur_3x3(image):
    return cv2.blur(image, (3, 3))

# Filtre Moyenneur 5x5
def filtre_moyenneur_5x5(image):
    return cv2.blur(image, (5, 5))

# Filtre Gaussien 3x3
def filtre_gaussien_3x3(image):
    return cv2.GaussianBlur(image, (3, 3), 0)

# Filtre Gaussien 5x5
def filtre_gaussien_5x5(image):
    return cv2.GaussianBlur(image, (5, 5), 0)

# Filtre Conique
def filtre_conique(image):
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], np.float32)
    return cv2.filter2D(image, -1, kernel)

# Filtre Pyramidal (Réduction de l'image)
def filtre_pyramidal(image):
    return cv2.pyrDown(image)

# Filtre Médian
def filtre_mediane(image):
    return cv2.medianBlur(image, 5)

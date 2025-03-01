import cv2
import numpy as np

# Erosion de l'image
def erosion_image(image):
    kernel = np.ones((5,5), np.uint8)
    return cv2.erode(image, kernel)

# Dilatation de l'image
def dilatation_image(image):
    kernel = np.ones((5,5), np.uint8)
    return cv2.dilate(image, kernel)

# Ouverture de l'image
def ouverture_image(image):
    kernel = np.ones((5,5), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

# Fermeture de l'image
def fermeture_image(image):
    kernel = np.ones((5,5), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

# Contours externes de l'image
def contours_externes(image):
    # Appliquer une dilatation suivie d'une érosion pour détecter les contours externes
    dilated = cv2.dilate(image, np.ones((5, 5), np.uint8))
    contours = dilated - image  # Soustraction de l'image d'origine à l'image dilatée
    return contours

# Contours internes de l'image
def contours_internes(image):
    # Appliquer une érosion suivie d'une dilatation pour détecter les contours internes
    eroded = cv2.erode(image, np.ones((5, 5), np.uint8))
    contours = image - eroded  # Soustraction de l'image érodée à l'image d'origine
    return contours


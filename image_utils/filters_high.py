import cv2
import numpy as np

def filtre_laplacien(image):
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    laplacian = cv2.convertScaleAbs(laplacian)  # Conversion en abs et normalisation dans [0, 255]
    return laplacian


def filtre_sobel(image):
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.magnitude(grad_x, grad_y)
    sobel = cv2.convertScaleAbs(sobel)  # Conversion en abs et normalisation dans [0, 255]
    return sobel


import numpy as np

def filtre_gradient(image):
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    gradient = cv2.magnitude(grad_x, grad_y)  # Calcul de la magnitude du gradient
    return cv2.convertScaleAbs(gradient)  # Conversion en type 8-bit pour affichage


# Filtre Prewitt
def filtre_prewwitt(image):
    kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], np.float32)
    kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], np.float32)

    grad_x = cv2.filter2D(image, -1, kernel_x)
    grad_y = cv2.filter2D(image, -1, kernel_y)

    grad_x = np.float32(grad_x)
    grad_y = np.float32(grad_y)

    magnitude = cv2.magnitude(grad_x, grad_y)

    return np.uint8(np.clip(magnitude, 0, 255))

# Filtre Robert
def filtre_robert(image):
    kernel_x = np.array([[1, 0], [0, -1]], np.float32)
    kernel_y = np.array([[0, 1], [-1, 0]], np.float32)
    grad_x = cv2.filter2D(image, -1, kernel_x)
    grad_y = cv2.filter2D(image, -1, kernel_y)

    grad_x = np.float32(grad_x)
    grad_y = np.float32(grad_y)

    result = cv2.magnitude(grad_x, grad_y)

    result = np.uint8(np.clip(result, 0, 255))
    return result

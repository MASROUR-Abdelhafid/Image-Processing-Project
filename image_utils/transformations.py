import cv2
import numpy as np

def inversion_image(image):
    # Assurer que l'image est binaire (0 et 255)
    image = np.where(image > 127, 255, 0).astype(np.uint8)  # Convertir en binaire
    return 255 - image  # Inverser l'image binaire



def binarisation_image(image, threshold=127):
    # Convertir l'image en niveaux de gris si elle est en couleur
    if len(image.shape) == 3:  # Si l'image a 3 canaux (couleur)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Appliquer la binarisation
    _, binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return binary_image


def contraste_image(image, alpha=1.5, beta=0):
    # # Convertir l'image en niveaux de gris si elle est en couleur
    # if len(image.shape) == 3:  # Si l'image a 3 canaux (couleur)
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


def division_image(image, divisor=2):
    # Vérifier si l'image est en couleur ou en niveaux de gris
    if len(image.shape) == 3:  # Si l'image a 3 canaux (couleur)
        return cv2.divide(image, divisor)
    else:  # Image en niveaux de gris
        return cv2.divide(image, divisor)


def niveaux_de_gris(image):
    if len(image.shape) == 3:  # Image en couleur (3 canaux)
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:  # Image déjà en niveaux de gris
        return image


def hough_image(image):
    if len(image.shape) == 3:  # Si l'image est en couleur
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Conversion en niveaux de gris
    edges = cv2.Canny(image, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            x1 = int(rho * np.cos(theta) - 1000 * np.sin(theta))
            y1 = int(rho * np.sin(theta) + 1000 * np.cos(theta))
            x2 = int(rho * np.cos(theta) + 1000 * np.sin(theta))
            y2 = int(rho * np.sin(theta) - 1000 * np.cos(theta))
            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Dessiner les lignes
    return image



# Transformation de Fourier et calcul du spectre
def fourier_transform(image):
    # Convertir l'image en niveaux de gris (si l'image est en couleur)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Appliquer la transformation de Fourier
    f = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
    fshift = np.fft.fftshift(f)  # Décalage du spectre pour le centrer

    # Calculer le spectre de magnitude
    magnitude_spectrum = cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1])

    # Normaliser le spectre pour l'affichage
    magnitude_spectrum = np.log(magnitude_spectrum + 1)
    magnitude_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX)

    # Convertir en image affichable
    magnitude_spectrum = np.uint8(magnitude_spectrum)

    return magnitude_spectrum

# Transformation inverse de Fourier
def inverse_fourier_transform(magnitude_spectrum):
    # Reconvertir l'image depuis le spectre de magnitude en utilisant la transformation inverse de Fourier
    rows, cols = magnitude_spectrum.shape
    # Créer le spectre de Fourier initial avec des fréquences complexes
    fshift = np.fft.ifftshift(magnitude_spectrum)  # Inverser le décalage pour restaurer les fréquences
    
    # Reconstruire les données complexes à partir du spectre de magnitude
    complex_image = np.zeros((rows, cols, 2), np.float32)
    complex_image[:, :, 0] = np.cos(fshift)
    complex_image[:, :, 1] = np.sin(fshift)

    # Appliquer la transformation inverse de Fourier pour obtenir l'image spatiale
    img_back = cv2.idft(complex_image)
    
    # Conversion du résultat vers une image affichable (en valeurs réelles)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    
    # Normalisation pour avoir une image visible (facultatif, selon le besoin)
    img_back = np.uint8(np.clip(img_back, 0, 255))

    return img_back



# # --- Correction pour la binarisation (assurez-vous que l'image est binaire avec uniquement 0 ou 255) ---
# def test_binarisation_image(self):
#     # Tester la binarisation
#     result = binarisation_image(self.image, 127)
#     self.assertEqual(result.shape, self.image.shape)
    
#     # Vérifier si l'image est bien binaire (composée uniquement de 0 et 255)
#     unique_values = np.unique(result)
#     self.assertTrue(np.array_equal(unique_values, [0, 255]))  # Vérifier uniquement 0 et 255

# # --- Correction pour l'inversion (vérifier que l'inversion est correcte) ---
# def test_inversion_image(self):
#     # Tester l'inversion de l'image
#     result = inversion_image(self.image)
#     self.assertEqual(result.shape, self.image.shape)
    
#     # Vérifier que l'inversion de l'image transforme 255 en 0 et 0 en 255
#     self.assertTrue(np.all(result == 0) or np.all(result == 255))  # Image inversée

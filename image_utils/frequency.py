import cv2
import numpy as np

def apply_fourier_transform(image):
    """
    Applique la transformation de Fourier à l'image et retourne l'image dans le domaine fréquentiel.
    """
    # Convertir l'image en niveaux de gris
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Appliquer la transformée de Fourier
    f = np.fft.fft2(gray_image)
    fshift = np.fft.fftshift(f)

    return fshift, gray_image

def inverse_fourier_transform(fshift):
    """
    Applique l'inverse de la transformation de Fourier pour revenir à l'image d'origine.
    """
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    return img_back

def create_filter(shape, filter_type='lowpass', cutoff=30, band=None):
    """
    Crée un filtre fréquentiel (passe-bas, passe-haut, passe-bande, ou passe-haut-bande).
    """
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2

    # Créer une matrice de filtres
    filter_matrix = np.zeros((rows, cols), dtype=np.float32)

    if filter_type == 'lowpass':
        # Passe-bas : garde les basses fréquences et supprime les hautes
        filter_matrix[crow - cutoff:crow + cutoff, ccol - cutoff:ccol + cutoff] = 1

    elif filter_type == 'highpass':
        # Passe-haut : garde les hautes fréquences et supprime les basses
        filter_matrix[:crow - cutoff, :] = 0
        filter_matrix[crow + cutoff:, :] = 0
        filter_matrix[:, :ccol - cutoff] = 0
        filter_matrix[:, ccol + cutoff:] = 0
        filter_matrix[crow - cutoff:crow + cutoff, ccol - cutoff:ccol + cutoff] = 0

    elif filter_type == 'bandpass' and band:
        # Passe-bande : garde les fréquences entre deux valeurs
        lower_cutoff, upper_cutoff = band
        filter_matrix[crow - lower_cutoff:crow + upper_cutoff, ccol - lower_cutoff:ccol + upper_cutoff] = 1

    elif filter_type == 'bandstop' and band:
        # Passe-haut-bande : supprime une bande de fréquences spécifiques
        lower_cutoff, upper_cutoff = band
        filter_matrix[crow - lower_cutoff:crow + upper_cutoff, ccol - lower_cutoff:ccol + upper_cutoff] = 0

    return filter_matrix

def apply_frequency_filter(image, filter_type='lowpass', cutoff=30, band=None):
    """
    Applique un filtre fréquentiel à l'image dans le domaine fréquentiel.
    """
    # Appliquer la transformée de Fourier
    fshift, gray_image = apply_fourier_transform(image)

    # Créer le filtre
    filter_matrix = create_filter(fshift.shape, filter_type, cutoff, band)

    # Appliquer le filtre dans le domaine fréquentiel
    fshift_filtered = fshift * filter_matrix

    # Revenir dans le domaine spatial
    img_back = inverse_fourier_transform(fshift_filtered)

    # Normaliser l'image
    img_back = np.uint8(np.clip(img_back, 0, 255))

    return img_back

def apply_lowpass_filter(image, cutoff=30):
    """
    Applique un filtre passe-bas (FPB) sur l'image.
    """
    return apply_frequency_filter(image, filter_type='lowpass', cutoff=cutoff)

def apply_highpass_filter(image, cutoff=30):
    """
    Applique un filtre passe-haut (FPH) sur l'image.
    """
    return apply_frequency_filter(image, filter_type='highpass', cutoff=cutoff)

def apply_bandpass_filter(image, band=(30, 60)):
    """
    Applique un filtre passe-bande (FPBB) sur l'image.
    """
    return apply_frequency_filter(image, filter_type='bandpass', band=band)

def apply_bandstop_filter(image, band=(30, 60)):
    """
    Applique un filtre passe-haut-bande (FPHB) sur l'image.
    """
    return apply_frequency_filter(image, filter_type='bandstop', band=band)

import sys
import os
import unittest
import cv2
import numpy as np
from image_utils.filters_low import filtre_moyenneur_3x3, filtre_gaussien_3x3
from image_utils.filters_high import filtre_sobel, filtre_laplacien
from image_utils.transformations import binarisation_image, inversion_image

# Ajouter le dossier principal du projet au chemin de recherche de Python
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

class TestImageProcessing(unittest.TestCase):

    def setUp(self):
        # Créer une image d'exemple (noir et blanc)
        self.image = np.array([[255, 255, 255], [255, 0, 0], [255, 0, 0]], dtype=np.uint8)
    
    def test_filtre_moyenneur_3x3(self):
        # Tester le filtre moyenneur 3x3
        result = filtre_moyenneur_3x3(self.image)
        self.assertEqual(result.shape, self.image.shape)
    
    def test_filtre_gaussien_3x3(self):
        # Tester le filtre gaussien 3x3
        result = filtre_gaussien_3x3(self.image)
        self.assertEqual(result.shape, self.image.shape)
    
    def test_filtre_sobel(self):
        # Tester le filtre Sobel
        result = filtre_sobel(self.image)
        self.assertEqual(result.shape, self.image.shape)
    
    def test_filtre_laplacien(self):
        # Tester le filtre Laplacien
        result = filtre_laplacien(self.image)
        self.assertEqual(result.shape, self.image.shape)
    
    def test_binarisation_image(self):
        # Tester la binarisation
        result = binarisation_image(self.image, 127)
        self.assertEqual(result.shape, self.image.shape)
        
        # Vérifier que l'image est binaire (0 ou 255)
        unique_values = np.unique(result)
        self.assertTrue(np.array_equal(unique_values, [0, 255]))  # Vérifier uniquement 0 et 255
    
    def test_inversion_image(self):
        # Tester l'inversion de l'image
        result = inversion_image(self.image)
        self.assertEqual(result.shape, self.image.shape)
        
        # Vérifier l'inversion correcte (255 -> 0 et 0 -> 255)
        inverted = cv2.bitwise_not(self.image)
        self.assertTrue(np.array_equal(result, inverted))  # Compare l'image inversée à celle obtenue avec bitwise_not

if __name__ == '__main__':
    unittest.main()

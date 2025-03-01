import cv2
import numpy as np

# Détection de coins de Harris
def harris_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    dst = cv2.dilate(dst, None)
    image[dst > 0.01 * dst.max()] = [0, 0, 255]  # Marquer les coins en rouge
    return image



def susan_detection(image, threshold=0.5, radius=5):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rows, cols = gray.shape
    image_with_corners = image.copy()  # Créer une copie de l'image originale
    
    gray = np.float32(gray)
    
    # Appliquer un noyau de voisinage de taille (2*radius+1)x(2*radius+1)
    for i in range(radius, rows - radius):
        for j in range(radius, cols - radius):
            pixel_value = gray[i, j]
            sum_sim = 0
            count = 0
            
            # Extraire la région autour du pixel (voisinage)
            region = gray[i-radius:i+radius+1, j-radius:j+radius+1]
            
            # Calculer les différences avec les voisins
            diff = np.abs(region - pixel_value)
            
            # Calculer la similitude en fonction de la différence
            similarity = np.exp(-(diff ** 2) / (2 * threshold ** 2))
            
            # Exclure la valeur du pixel central
            similarity[radius, radius] = 0
            
            sum_sim = np.sum(similarity)
            count = np.count_nonzero(similarity)
            
            # Afficher les informations pour débogage
            print(f"Pixel: {i},{j}, Sum Sim: {sum_sim}, Count: {count}, Threshold: {threshold}")
            
            # Si la somme des similarités est inférieure au seuil, marquer le pixel comme un coin
            if sum_sim < (count * threshold):
                # Marquer les coins détectés en rouge sur l'image originale
                image_with_corners[i, j] = [0, 0, 255]  # Rouge en BGR
    
    return image_with_corners







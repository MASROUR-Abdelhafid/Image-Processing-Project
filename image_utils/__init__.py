# Initialisation du package image_utils

# Optionnellement, tu peux importer des fonctions specifiques pour faciliter l'acces direct
from .filters_low import *
from .filters_high import *
from .transformations import *
from .morphology import *
from .detection import *
from .histogram import *

# Si tu veux que le module expose certaines fonctions directement, par exemple :
# from .filters_low import filtre_moyenneur_3x3, filtre_gaussien_3x3

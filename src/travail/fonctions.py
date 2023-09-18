# Ce script contient les fonctions concernant le projet

#####################
#    imports        #
#####################
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt

from travail.constantes import classes_principales


#####################
#    fonctions        #
#####################

def add_color_masks_from_json(json_file, image_file, color_file):
    """
    Cette fonction prend en entrée un fichier JSON contenant les polygones des classes d'une image,
    le chemin vers l'image originale, ainsi que le chemin vers une autre image de couleur.
    Elle affiche quatre images : l'image originale, l'image avec les masques de couleur correspondant aux classes,
    l'image couleur spécifiée, et une image montrant uniquement les masques de couleur sans l'image originale.

    :param json_file: Chemin vers le fichier JSON contenant les polygones des classes.
    :param image_file: Chemin vers l'image originale.
    :param color_file: Chemin vers l'image couleur.
    """
    # Charger le fichier JSON contenant les polygones
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Charger les images
    image = cv2.imread(image_file)
    color_image = cv2.imread(color_file)

    # Créer une image vide avec les mêmes dimensions que l'image originale
    masks = np.zeros_like(image)

    # Définir les couleurs pour chaque groupe de classes
    couleurs_groupes = {
        'flat': (0, 0, 255),  # Rouge
        'human': (0, 255, 0),  # Vert
        'vehicle': (255, 0, 0),  # Bleu
        'construction': (255, 255, 0),  # Cyan
        'object': (255, 0, 255),  # Magenta
        'nature': (0, 255, 255),  # Jaune
        'sky': (255, 255, 255),  # Blanc
        'void': (0, 0, 0)  # Noir
    }

    # Parcourir les annotations du fichier JSON
    for obj in data['objects']:
        class_name = obj['label']
        points = np.array(obj['polygon'], dtype=np.int32)

        # Recherche du groupe de classe correspondant
        groupe_classe = None
        for groupe, classes_incluses in classes_principales.items():
            if class_name in classes_incluses:
                groupe_classe = groupe
                break

        # Utilisation de la couleur correspondante au groupe de classe pour le masque
        if groupe_classe is not None:
            color = couleurs_groupes[groupe_classe]
            masks = cv2.fillPoly(masks, [points], color)

    # Combiner l'image originale avec les masques de couleur
    output_image = cv2.addWeighted(image, 0.7, masks, 0.3, 0)

    # Créer les images supplémentaires
    color_image_resized = cv2.resize(color_image, (image.shape[1], image.shape[0]))
    masks_only_image = np.copy(masks)
    masks_only_image[masks_only_image != 0] = 255

    # Afficher les images avec un espace réduit et un cadre
    fig, ax = plt.subplots(2, 2, figsize=(12, 8), tight_layout=True)
    fig.suptitle('Visualisation des masques d\'image', fontsize=16)
    fig.patch.set_edgecolor('black')
    fig.patch.set_linewidth(2.0)
    ax[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax[0, 0].set_title('Image originale')
    ax[0, 0].axis('off')
    ax[0, 1].imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    ax[0, 1].set_title('Image avec masques de couleur')
    ax[0, 1].axis('off')
    ax[1, 0].imshow(cv2.cvtColor(color_image_resized, cv2.COLOR_BGR2RGB))
    ax[1, 0].set_title('Image couleur')
    ax[1, 0].axis('off')
    ax[1, 1].imshow(masks_only_image, cmap='gray')
    ax[1, 1].set_title('Masques uniquement')
    ax[1, 1].axis('off')
    plt.show()




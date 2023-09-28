import os
from keras.utils import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

from src.travail.multiclassunet import Unet, unet_view

##############################################
#      Chargement variables et modèles       #
##############################################

source = os.getcwd()
model_path = os.path.join(source, "src/model/weights_model_final.h5").replace("\\", "/")

model = unet_view(Unet(256, 512))
model.load_weights(model_path)


def afficher_images_cote_a_cote(image1, image2):
    """
    Affiche deux images côte à côte avec des titres au-dessus de chaque image.

    Args:
        image1 (numpy.ndarray): Première image sous forme de tableau NumPy.
        image2 (numpy.ndarray): Deuxième image sous forme de tableau NumPy.
    """
    # Créer une figure avec 2 sous-plots côte à côte
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Afficher la première image avec son titre
    axes[0].imshow(image1, cmap='gray')
    axes[0].set_title("Image originale")
    axes[0].axis('off')

    # Afficher la deuxième image avec son titre
    axes[1].imshow(image2, cmap='gray')
    axes[1].set_title("Masque prédit")
    axes[1].axis('off')


    # Sauvegarder la figure dans un objet mémoire
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0)
    buffer.seek(0)
    plt.close()

    # Convertir l'image en base64
    image_base64 = base64.b64encode(buffer.read()).decode()

    return image_base64


def predict_img(filename):
    img = load_img(filename, target_size=(256, 512))
    img_arr = img_to_array(img)
    img_array = np.expand_dims(img_arr, axis=0)
    predicted_mask = model.predict(img_array)[0]

    image_base64 = afficher_images_cote_a_cote(img_arr.astype("uint8"), predicted_mask)

    return image_base64


if __name__ == '__main__':
    pass
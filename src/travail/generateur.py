
batch_size = 16
samples = 50000
steps = samples//batch_size
img_height, img_width = 256, 256
classes = 8
filters_n = 64


import os
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from keras.utils import load_img, img_to_array
import math


def data_generator(image_folder, mask_folder, batch_size, img_height, img_width, nb_classes=29):
    """
    Fonction génératrice pour alimenter un modèle U-Net avec des données d'entraînement.

    :param image_folder: Chemin du dossier contenant les images.
    :param mask_folder: Chemin du dossier contenant les images segmentées (masques).
    :param batch_size: Taille du batch (nombre d'images) à générer à chaque itération.
    :param img_height: Hauteur des images d'entrée du modèle.
    :param img_width: Largeur des images d'entrée du modèle.
    :param nb_classes: Nombre de classes à segmenter

    :yield: Tuple contenant un lot d'images et le lot correspondant de masques (labels).
    """
    image_filenames = os.listdir(image_folder)
    num_samples = len(image_filenames)

    while True:
        indices = np.arange(num_samples)
        np.random.shuffle(indices)

        for start_idx in range(0, num_samples - batch_size + 1, batch_size):
            batch_indices = indices[start_idx: start_idx + batch_size]
            batch_images = []
            batch_masks = []

            for idx in batch_indices:
                image_name = image_filenames[idx]
                image_path = os.path.join(image_folder, image_name)
                mask_name = image_name[:-16] + '_gtFine_color_converted.png'
                mask_path = os.path.join(mask_folder, mask_name)

                # Charger et prétraiter l'image
                img = load_img(image_path, target_size=(img_height, img_width))
                img_array = img_to_array(img)
                batch_images.append(img_array)

                # Charger et prétraiter le masque (image segmentée)
                mask = load_img(mask_path, target_size=(img_height, img_width), color_mode='grayscale')
                mask_array = img_to_array(mask, dtype='uint8').squeeze()

                # Conversion au format voulu : d'abord applati puis one hot encodé
                mask_one_hot_enc = pd.DataFrame({'cat_col': mask_array.reshape((img_height * img_width))})

                ordered_categories = np.arange(0, nb_classes)

                # Créer une catégorie ordonnée
                ordered_cat = pd.Categorical(mask_one_hot_enc['cat_col'], categories=ordered_categories, ordered=True)

                # Remplacer la colonne originale par la catégorie ordonnée
                mask_one_hot_enc['cat_col'] = ordered_cat

                # Effectuer le one-hot encoding
                mask_one_hot_enc = pd.get_dummies(mask_one_hot_enc, columns=['cat_col'])

                mask_one_hot_enc = np.array(mask_one_hot_enc)

                batch_masks.append(mask_one_hot_enc)

            batch_images = np.array(batch_images)
            batch_masks = np.array(batch_masks)

            yield batch_images, batch_masks


def _largest_rotated_rect(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle within the rotated rectangle.
    Original JS code by 'Andri' and Magnus Hoff from Stack Overflow
    Converted to Python by Aaron Snoswell
    Source: http://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders
    """

    quadrant = int(math.floor(angle / (math.pi / 2))) & 3
    sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
    alpha = (sign_alpha % math.pi + math.pi) % math.pi

    bb_w = w * math.cos(alpha) + h * math.sin(alpha)
    bb_h = w * math.sin(alpha) + h * math.cos(alpha)

    gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

    delta = math.pi - alpha - gamma

    length = h if (w < h) else w

    d = length * math.cos(alpha)
    a = d * math.sin(alpha) / math.sin(delta)

    y = a * math.cos(gamma)
    x = y * math.tan(gamma)

    return (
        bb_w - 2 * x,
        bb_h - 2 * y
    )


def rotation(path_image, angle):
    """
    Cette fonction effectue une rotation à une image, la recadre sur la partie centrale pour garder le format de l'image
    d'origine sans avoir de bords puis la redimensionne à la taille voulue

    :param path_image: chemin vers l'image
    :param angle: angle de rotation
    :return: image après rotation
    """
    im = Image.open(path_image)
    im = im.rotate(angle, expand=False)
    w, h = im.size

    lrr_width, lrr_height = _largest_rotated_rect(w, h, math.radians(angle))
    resized_image = tf.image.central_crop(im, float(lrr_height) / h)

    return resized_image.numpy().astype('uint8')


def augmentation(path_image, path_masque, output_height, output_width, seed_n):
    """
    Applique diverses transformations à une image et à son masque.

    :param path_image: Chemin menant à l'image d'entrée.
    :param path_masque: Chemin menant au masque correspondant à l'image.
    :param output_height: Hauteur de sortie souhaitée.
    :param output_width: Largeur de sortie souhaitée.
    :param seed_n: Graine pour la génération pseudo-aléatoire.

    :return: Tuple contenant l'image et le masque transformés.
    """

    # Génère aléatoirement un facteur d'échelle, du bruit et un angle de rotation
    scale_factor = tf.random.stateless_uniform([2], seed=seed_n, minval=0.9, maxval=1, dtype=tf.float16).numpy()
    noise = tf.random.stateless_uniform([1], seed=seed_n, minval=0, maxval=5, dtype=tf.int32).numpy()
    # theta = tf.random.stateless_uniform([1], seed=seed_n, minval=-5, maxval=6, dtype=tf.int32).numpy()
    theta = 0

    # Modification commune - Image
    img0_aug = rotation(path_image, angle=theta)
    orig_h, orig_w, nb_can = img0_aug.shape
    new_h, new_w = int(orig_h * scale_factor[0]), int(orig_w * scale_factor[1])

    img0_aug = tf.image.stateless_random_crop(img0_aug, (orig_h, orig_w, nb_can), seed=seed_n)
    img0_aug = tf.image.resize(img0_aug, [orig_h, orig_w], method=tf.image.ResizeMethod.BILINEAR)
    img0_aug = tf.image.stateless_random_flip_left_right(img0_aug, seed=seed_n)

    # Modification propre au images
    if np.random.rand(1) < 0.2:
        img0_aug = tf.image.stateless_random_brightness(img0_aug, seed=seed_n, max_delta=5)

    elif np.random.rand(1) < 0.2:
        img0_aug = tf.image.stateless_random_contrast(img0_aug, lower=0.8, upper=1.2, seed=seed_n)

    elif np.random.rand(1) < 0.2:
        img0_aug = tf.image.stateless_random_hue(img0_aug, max_delta=0.1, seed=seed_n)

    elif np.random.rand(1) < 0.2:
        img0_aug = tf.image.stateless_random_saturation(img0_aug, lower=0.8, upper=1.2, seed=seed_n)

    elif np.random.rand(1) < 0.2:
        img0_aug = img0_aug + tf.random.normal(shape=tf.shape(img0_aug), mean=0.0, stddev=noise, dtype=tf.float32)

    img0_aug = tf.clip_by_value(img0_aug, 0.0, 255.0)
    img0_aug = tf.image.resize(img0_aug, [output_height, output_width], method=tf.image.ResizeMethod.BILINEAR)

    # Modification commune - Masque
    mask0_aug = np.expand_dims(rotation(path_masque, angle=theta)[:, :, 0], axis=2)
    orig_h, orig_w, nb_can = mask0_aug.shape
    new_h, new_w = int(orig_h * scale_factor[0]), int(orig_w * scale_factor[1])

    mask0_aug = tf.image.stateless_random_crop(mask0_aug, (orig_h, orig_w, nb_can), seed=seed_n)
    mask0_aug = tf.image.stateless_random_flip_left_right(mask0_aug, seed=seed_n)
    mask0_aug = tf.image.resize(mask0_aug, [output_height, output_width], method=tf.image.ResizeMethod.BILINEAR)

    return img0_aug.numpy(), mask0_aug.numpy()


def data_generator_aug(image_folder, mask_folder, batch_size, img_height, img_width, nb_classes=29):
    """
    Fonction génératrice pour alimenter un modèle U-Net avec des données d'entraînement.

    :param image_folder: Chemin du dossier contenant les images.
    :param mask_folder: Chemin du dossier contenant les images segmentées (masques).
    :param batch_size: Taille du batch (nombre d'images) à générer à chaque itération.
    :param img_height: Hauteur des images souhaitée en sortie du modèle.
    :param img_width: Largeur des images souhaitée en sortie du modèle.
    :param nb_classes: Nombre de classes à segmenter

    :yield: Tuple contenant un lot d'images et le lot correspondant de masques (labels).
    """
    image_filenames = os.listdir(image_folder)
    num_samples = len(image_filenames)
    j = 0
    while True:
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        j += 1

        for start_idx in range(0, num_samples - batch_size + 1, batch_size):
            batch_indices = indices[start_idx: start_idx + batch_size]
            batch_images = []
            batch_masks = []

            for idx in batch_indices:
                image_name = image_filenames[idx]
                image_path = os.path.join(image_folder, image_name)
                mask_name = image_name[:-16] + '_gtFine_color_converted.png'
                mask_path = os.path.join(mask_folder, mask_name)

                img_array, mask_array = augmentation(image_path, mask_path, img_height, img_width, (idx, j))

                # Conversion au format voulu : d'abord applati puis one hot encodé
                mask_one_hot_enc = pd.DataFrame({'cat_col': mask_array.reshape((img_height * img_width))})

                ordered_categories = np.arange(0, nb_classes)

                # Créer une catégorie ordonnée
                ordered_cat = pd.Categorical(mask_one_hot_enc['cat_col'], categories=ordered_categories, ordered=True)

                # Remplacer la colonne originale par la catégorie ordonnée
                mask_one_hot_enc['cat_col'] = ordered_cat

                # Effectuer le one-hot encoding
                mask_one_hot_enc = pd.get_dummies(mask_one_hot_enc, columns=['cat_col'])

                mask_one_hot_enc = np.array(mask_one_hot_enc)

                batch_images.append(img_array)
                batch_masks.append(mask_one_hot_enc)

            batch_images = np.array(batch_images)
            batch_masks = np.array(batch_masks)

            yield batch_images, batch_masks


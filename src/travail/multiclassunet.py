# import os
# import random
import numpy as np
# import matplotlib.pyplot as plt
from keras import backend as K
from keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Conv2DTranspose, \
    BatchNormalization, Activation, Dropout, Reshape, Lambda, Permute
# from tensorflow.keras.optimizers import Adadelta, Nadam
from keras.models import Model, load_model
# from tensorflow.python.keras.utils import multi_gpu_model, plot_model
# from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
# from tensorflow.python.keras.preprocessing import image
import tensorflow as tf


original_color = {
    '0': [0, 0, 0], '1': [0, 0, 70], '2': [0, 0, 90], '3': [0, 0, 110], '4': [0, 0, 142], '5': [0, 0, 230],
    '6': [0, 60, 100], '7': [0, 80, 100], '8': [70, 70, 70], '9': [70, 130, 180], '10': [81, 0, 81],
    '11': [102, 102, 156], '12': [107, 142, 35], '13': [111, 74, 0], '14': [119, 11, 32], '15': [128, 64, 128],
    '16': [150, 100, 100], '17': [150, 120, 90], '18': [152, 251, 152], '19': [153, 153, 153], '20': [180, 165, 180],
    '21': [190, 153, 153], '22': [220, 20, 60], '23': [220, 220, 0], '24': [230, 150, 140], '25': [244, 35, 232],
    '26': [250, 170, 30], '27': [250, 170, 160], '28': [255, 0, 0]
}

original_color_matrix = np.array([
    [0, 0, 0], [0, 0, 70], [0, 0, 90], [0, 0, 110], [0, 0, 142], [0, 0, 230],
    [0, 60, 100], [0, 80, 100], [70, 70, 70], [70, 130, 180], [81, 0, 81],
    [102, 102, 156], [107, 142, 35], [111, 74, 0], [119, 11, 32], [128, 64, 128],
    [150, 100, 100], [150, 120, 90], [152, 251, 152], [153, 153, 153], [180, 165, 180],
    [190, 153, 153], [220, 20, 60], [220, 220, 0], [230, 150, 140], [244, 35, 232],
    [250, 170, 30], [250, 170, 160], [255, 0, 0]
])


def conv_block(tensor, nfilters, size=3, padding='same', initializer="he_normal"):
    """
    Bloc de convolution : Une unité de base qui consiste en deux couches de convolution, chacune suivie d'une
    couche de normalisation en lots (BatchNormalization) et d'une activation ReLU. Ce bloc aide à capturer les
    caractéristiques importantes dans les données d'entrée et introduit de la non-linéarité.

    :param tensor: Tenseur en entrée pour le bloc de convolution.
    :param nfilters: Nombre de filtres (ou canaux) dans les couches de convolution.
    :param size: Taille du noyau de convolution à utiliser. Par défaut, 3x3.
    :param padding: Type de remplissage pour la convolution. Par défaut, 'same' pour maintenir la même taille de sortie.
    :param initializer: Méthode d'initialisation des poids du réseau. Par défaut, 'he_normal' pour une initialisation
                        adaptée aux fonctions d'activation ReLU.

    :return: Tenseur après le passage par le bloc de convolution.
    """
    x = Conv2D(filters=nfilters, kernel_size=(size, size), padding=padding, kernel_initializer=initializer)(tensor)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(filters=nfilters, kernel_size=(size, size), padding=padding, kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x


def deconv_block(tensor, residual, nfilters, size=3, padding='same', strides=(2, 2)):
    """
    Bloc de déconvolution : Cette fonction réalise une opération de déconvolution (ou up-sampling) pour augmenter la
    résolution spatiale du tenseur en entrée `tensor`. Elle est utilisée dans la partie du décodeur du modèle U-Net pour
    reconstruire une sortie de segmentation détaillée à partir des caractéristiques apprises dans la partie de l'encodeur.

    :param tensor: Tenseur en entrée pour le bloc de déconvolution. Représente les caractéristiques à augmenter en résolution.
    :param residual: Tenseur représentant les caractéristiques résiduelles (skip connection) provenant de la même couche
                     de l'encodeur que `tensor`. Ces caractéristiques résiduelles sont concaténées avec le résultat de la
                     déconvolution pour conserver les détails importants dans la sortie déconvoluée.
    :param nfilters: Nombre de filtres (ou canaux) à utiliser dans la déconvolution pour augmenter le nombre de
                     caractéristiques dans la sortie de cette étape.
    :param size: Taille du noyau de la couche de déconvolution. Par défaut, 3x3, mais cela peut être personnalisé pour
                 ajuster la taille de l'opération de déconvolution.
    :param padding: Type de remplissage utilisé dans la déconvolution. Par défaut, 'same' est utilisé pour conserver la
                    même taille de sortie que l'entrée.
    :param strides: Pas (strides) utilisé pour la déconvolution. Par défaut, (2, 2) est utilisé, ce qui signifie que la
                    résolution spatiale de la sortie sera deux fois plus grande que celle de l'entrée.

    :return: Tenseur après avoir été passé par l'opération de déconvolution et les étapes supplémentaires du bloc.
    """
    y = Conv2DTranspose(nfilters, kernel_size=(size, size), strides=strides, padding=padding)(tensor)
    y = concatenate([y, residual], axis=3)
    y = conv_block(y, nfilters)
    return y


def Unet(img_height, img_width, nclasses=29, filters=64):
    """
    Crée le modèle U-Net pour la segmentation d'image.

    :param img_height: Hauteur des images en entrée du modèle.
    :param img_width: Largeur des images en entrée du modèle.
    :param nclasses: Nombre de classes à segmenter. Par défaut, 29.
    :param filters: Nombre de filtres (ou canaux) de la première couche de convolution. Ce nombre sera doublé à
                    chaque couche subséquente du modèle U-Net.

    :return: Modèle U-Net complet pour la segmentation d'image.
    """
    # down
    input_layer = Input(shape=(img_height, img_width, 3), name='image_input')
    conv1 = conv_block(input_layer, nfilters=filters)
    conv1_out = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = conv_block(conv1_out, nfilters=filters * 2)
    conv2_out = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = conv_block(conv2_out, nfilters=filters * 4)
    conv3_out = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = conv_block(conv3_out, nfilters=filters * 8)
    conv4_out = MaxPooling2D(pool_size=(2, 2))(conv4)
    conv4_out = Dropout(0.5)(conv4_out)
    conv5 = conv_block(conv4_out, nfilters=filters * 16)
    conv5 = Dropout(0.5)(conv5)
    # up
    deconv6 = deconv_block(conv5, residual=conv4, nfilters=filters * 8)
    deconv6 = Dropout(0.5)(deconv6)
    deconv7 = deconv_block(deconv6, residual=conv3, nfilters=filters * 4)
    deconv7 = Dropout(0.5)(deconv7)
    deconv8 = deconv_block(deconv7, residual=conv2, nfilters=filters * 2)
    deconv9 = deconv_block(deconv8, residual=conv1, nfilters=filters)
    # output
    output_layer = Conv2D(filters=nclasses, kernel_size=(1, 1))(deconv9)
    output_layer = BatchNormalization()(output_layer)
    output_layer = Reshape((img_height * img_width, nclasses), input_shape=(img_height, img_width, nclasses))(
        output_layer)
    output_layer = Activation('softmax')(output_layer)

    model = Model(inputs=input_layer, outputs=output_layer, name='Unet')

    return model


def map_classes(x, img_height, img_width):
    x = tf.cast(x, tf.int32)  # Convertir le tenseur en int32 si nécessaire
    color_matrix = tf.constant(original_color_matrix, dtype=tf.int32)
    mapped_values = tf.gather(color_matrix, x)
    return tf.reshape(mapped_values, shape=(tf.shape(x)[0], img_height, img_width, 3))


def unet_view(model_u):
    """

    :param model_u: modèle unet entraîné
    :return: modèle auquel on ajoute des couches de sorties pour obtenir l'image de sortie
    """

    # Exclut la dimension du batch pour récupérer le format d'entrée des images
    img_height, img_width = model_u.input_shape[1:3]

    # Ajouter une couche de classification
    classification_layer = Lambda(lambda x: tf.argmax(x, axis=-1, output_type=tf.int32))(model_u.output)
    # Créer une couche Lambda pour la correspondance
    mapping_layer = tf.keras.layers.Lambda(lambda x: map_classes(x, img_height, img_width))(classification_layer)

    # Créer un modèle complet avec la couche de classification
    model = Model(inputs=model_u.input, outputs=mapping_layer, name='Unet_with_classification')

    return model



# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import vgg19

from argparse import ArgumentParser

"""
@ Util function to parse command line
"""
def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--content',
            dest='content', help='content image',
            metavar='CONTENT', required=True)
    parser.add_argument('--styles',
            dest='style',
            nargs='+', help='style image',
            metavar='STYLE', required=True)
    parser.add_argument('--output',
            dest='output', help='output path',
            metavar='OUTPUT', required=True)
    parser.add_argument('--iterations', type=int,
            dest='iterations', help='number of iterations',
            metavar='ITERATIONS', default=200)
    parser.add_argument('--width', type=int,
            dest='width', help='output width',
            metavar='WIDTH', default=800)
    return parser
	
"""
@ Util function to load, resize and prepare picture into approprite tensor for VGG-19 network
"""
        
def preprocess_image(image_path, width, height):
    #Load image from path with specified dimensions
    img = keras.preprocessing.image.load_img(image_path, target_size=(height, width))
    img = keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return tf.convert_to_tensor(img)


"""
@ restore an image outputted from VGG-19 network
"""

def deprocess_image(x):
    if len(x.shape) == 4:
        x = np.squeeze(x, 0)
    assert len(x.shape) == 3, ("Input to deprocess image must be an image of dimension"
                             "[1, height, width, channel] or [height, width, channel]")
    if len(x.shape) != 3:
        raise ValueError("Invalid input to deprocessing image")

    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype("uint8")
    return x


"""
@ compute the Gram matrix of an image x
"""

def gram_matrix(x):
    # Setting image in the format (Channels, rows, cols)
    x = tf.transpose(x, (2, 0, 1))
    # Unrolling the matrix in the format (Channels, rows*cols)
    features = tf.reshape(x, (tf.shape(x)[0], -1))
    # Gram matrix X.X_transpose
    gram = tf.matmul(features, tf.transpose(features))
    return gram
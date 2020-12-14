# -*- coding: utf-8 -*-

from utils import *


# List of layers to use for the style cost
STYLE_LAYERS = [
    ('block1_conv1', 0.2),
    ('block2_conv1', 0.2),
    ('block3_conv1', 0.2),
    ('block4_conv1', 0.2),
    ('block5_conv1', 0.2)]


# The layer to use for the content cost
CONTENT_LAYER = "block5_conv2"


"""
@ Compute the style cost for one layer of the network
"""
def compute_layer_style_cost(style, generated):
    # Compute the Gram matrix of both matrixes
    S = gram_matrix(style)
    G = gram_matrix(generated)
    # The cost is 
    return tf.reduce_sum(tf.pow((S - G),2)) 


"""
@ Compute the content cost
"""

def compute_content_cost(content, generated):
    return tf.reduce_sum(tf.square(generated - content))

"""
@ Compute the total cost for the fusion
"""
def compute_cost(generated, content, style):
    # Create a tensor for the three images
    input_tensor = tf.concat( [content, style, generated], axis=0 )
    # Build a VGG19 model loaded with pre-trained ImageNet weights
    model = vgg19.VGG19(weights="imagenet", include_top=False)

    # Get the symbolic outputs of each "key" layer (we gave them unique names).
    outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

    # Set up a model that returns the activation values for every layer in
    # VGG19 (as a dict).
    feature_extractor = keras.Model(inputs=model.inputs, outputs=outputs_dict)
    features = feature_extractor(input_tensor)

    # Initialize the cost
    cost = tf.zeros(shape=())

    _,w,h,c = content.shape
    #Create a normalization term
    coeff =  1e-3/(2 * c * h * w)
    # Add content cost
    layer_features = features[CONTENT_LAYER]
    content_im_features = layer_features[0, :, :, :]
    generated_im_features = layer_features[2, :, :, :]
    cost = cost + coeff * compute_content_cost(
        content_im_features, generated_im_features)
    
    # Add style cost, iterate over the chosen layers in "style_layer_names"
    for layer_name, factor in STYLE_LAYERS:
        layer_features = features[layer_name]
        style_features = layer_features[1, :, :, :]
        generated_features = layer_features[2, :, :, :]
        # Compute style cost for the layer
        style_cost = compute_layer_style_cost(style_features, generated_features)
        # We multiply the layer cost by the factor for each layer and the normilization coeff
        cost = cost + coeff**2 * factor * style_cost 
    
    return cost

"""
@ Compute the gradient of the loss function
"""
def compute_cost_and_grads(generated_im, content_im, style_im):
    with tf.GradientTape() as tape:
        cost = compute_cost(generated_im, content_im, style_im)
    grads = tape.gradient(cost, generated_im)
    return cost, grads
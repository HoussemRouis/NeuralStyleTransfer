# -*- coding: utf-8 -*-

from functions import *
import sys


content_path = sys.argv[1]
style_path = sys.argv[2]
result_prefix = sys.argv[3]

img_nrows = 800
iterations = 4000

# Dimensions of the generated picture.
width, height = keras.preprocessing.image.load_img(content_path).size

img_ncols = int(width * img_nrows / height)


# Setting up the optimizer
optimizer = keras.optimizers.SGD(
    keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=100.0, 
                                                decay_steps=100, decay_rate=0.96)
)

#you can try Adam optimizer but SGD seems to be working better for this task
#optimizer = keras.optimizers.Adam(learning_rate=100.0)

# Create the three images
content_im = preprocess_image(content_path, img_ncols,img_nrows)
style_im = preprocess_image(style_path, img_ncols,img_nrows)
generated_im = tf.Variable(preprocess_image(content_path, img_ncols,img_nrows))


for i in range(iterations):
    # Computing cost and grads
    cost, grads = compute_cost_and_grads(
        generated_im, content_im, style_im
    )
    # Updating the generated image by applying grads
    optimizer.apply_gradients([(grads, generated_im)])
    print("Iteration %d: loss=%.2f" % (i, cost))

img = deprocess_image(generated_im.numpy())        
keras.preprocessing.image.save_img(result_prefix + "Generated.png", img)
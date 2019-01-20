import cv2

import numpy as np
from PIL import Image
from keras import backend as K
# Load pre-trained image recognition model
from keras.models import load_model
from keras.preprocessing import image


def image_processing(path):
    # Load the image to hack
    img = image.load_img(path, target_size=(150, 150))
    input_image = image.img_to_array(img)

    # Scale the image so all pixel intensities are between [-1, 1] as the model expects
    input_image /= 255.
    input_image -= 0.5
    input_image *= 2.

    # Add a 4th dimension for batch size (as Keras expects)
    input_image = np.expand_dims(input_image, axis=0)
    return input_image


def save_image(image, path):
    # De-scale the image's pixels from [-1, 1] back to the [0, 255] range
    img = image[0]
    img /= 2.
    img += 0.5
    img *= 255.
    # Save the hacked image!
    im = Image.fromarray(img.astype(np.uint8))
    im.save(path)


model = load_model('model_keras_2.h5')
model.load_weights('model_weights_2.h5')
# Grab a reference to the first and last layer of the neural net
# model_input_layer = model.layers[0].input
# model_output_layer = model.layers[-1].output
model_input_layer = model.get_input_at(0)
model_output_layer = model.get_output_at(-1)
# Class #1 is "others"
object_type_to_fake = 1

# Load the image to hack
image1 = image_processing('021280.jpg')
# image22 = image_processing('006526_0.jpg')
print(model.predict(image1))
# print(model.predict(image22))
save_image(image1, 'image1.png')

# Pre-calculate the maximum change we will allow to the image
# We'll make sure our hacked image never goes past this so it doesn't look funny.
# A larger number produces an image faster but risks more distortion.
max_change_above = image1 + 0.01
max_change_below = image1 - 0.01
# Create a copy of the input image to hack on
hacked_image = np.copy(image1)
# How much to update the hacked image in each iteration
learning_rate = 0.1
# Define the cost function.
# Our 'cost' will be the likelihood out image is the target class according to the pre-trained model
cost_function = model_output_layer[0, object_type_to_fake]
# We'll ask Keras to calculate the gradient based on the input image and the currently predicted class
# In this case, referring to "model_input_layer" will give us back image we are hacking.
gradient_function = K.gradients(cost_function, model_input_layer)[0]
# Create a Keras function that we can call to calculate the current cost and gradient
grab_cost_and_gradients_from_model = K.function([model_input_layer, K.learning_phase()],
                                                [cost_function, gradient_function])

confidence = 0.0
# In a loop, keep adjusting the hacked image slightly so that it tricks the model more and more
# until it gets to at least 80% confidence
while confidence < 0.80:
    # Check how close the image is to our target class and grab the gradients we
    # can use to push it one more step in that direction.
    # Note: It's really important to pass in '0' for the Keras learning mode here!
    # Keras layers behave differently in prediction vs. train modes!
    confidence, gradients = grab_cost_and_gradients_from_model([hacked_image, 0])
    # Move the hacked image one step further towards fooling the model
    hacked_image += gradients * learning_rate
    # Ensure that the image doesn't ever change too much to either look funny or to become an invalid image
    hacked_image = np.clip(hacked_image, max_change_below, max_change_above)
    hacked_image = np.clip(hacked_image, -1.0, 1.0)
    print("Model's predicted likelihood that the image is not a cartoon: {:.8}%".format(confidence * 100))

save_image(hacked_image, 'hacked_image.png')
# Predict hacked image.
print(model.predict(hacked_image))

import numpy as np
from keras.preprocessing import image
from keras.applications import inception_v3


def image_processing(path):
    # Load the image file and convert it to a numpy array
    img = image.load_img(path, target_size=(299, 299))
    input_image = image.img_to_array(img)
    # Scale the image so all pixel intensities are between [-1, 1] as the model expects
    input_image /= 255.
    input_image -= 0.5
    input_image *= 2.
    # Add a 4th dimension for batch size (as Keras expects)
    return np.expand_dims(input_image, axis=0)


# Load pre-trained image recognition model
model = inception_v3.InceptionV3()

image_data = image_processing('00026.jpg')

# Run the image through the neural network
predictions = model.predict(image_data)

# Convert the predictions into text and print them
predicted_classes = inception_v3.decode_predictions(predictions, top=1)
imagenet_id, name, confidence = predicted_classes[0][0]
print("This is a {} with {:.4}% confidence!".format(name, confidence * 100))

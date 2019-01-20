# Lets take a look at our directory
import os
import random

from keras.preprocessing import image

train_dir = 'data/train/'
test_dir = 'data/validation/'

label_cartoon = 'cartoon'
label_others = 'others'

image_format = 'jpg'


def prepare_data_set(dir):
    dir_cartoon = dir + label_cartoon
    data_set = ['{}/{}'.format(dir_cartoon, i) for i in os.listdir(dir_cartoon) if image_format in i]
    dir_others = dir + label_others
    data_set += ['{}/{}'.format(dir_others, i) for i in os.listdir(dir_others) if image_format in i]
    return data_set


train_set = prepare_data_set(train_dir)
test_set = prepare_data_set(test_dir)
random.shuffle(train_set)  # shuffle it randomly

# Lets declare our image dimensions
# we are using coloured images.
nrows = 150
ncolumns = 150
channels = 3  # change to 1 if you want to use grayscale image

import numpy as np


# A function to read and process the images to an acceptable format for our model
def read_and_process_image(list_of_images):
    """
    Returns two arrays:
        X is an array of resized images
        y is an array of labels
    """
    X = []  # images
    y = []  # labels
    for image_path in list_of_images:
        # Load the image file and convert it to a numpy array
        img = image.load_img(image_path, target_size=(150, 150))
        input_image = image.img_to_array(img)
        # Scale the image so all pixel intensities are between [-1, 1] as the model expects
        input_image /= 255.
        input_image -= 0.5
        input_image *= 2.
        X.append(input_image)
        # Set the labels as Softmax style.
        if label_cartoon in image_path:
            y.append([1., 0.])
        else:
            y.append([0., 1.])
    return X, y


# get the train and label data
X, y = read_and_process_image(train_set)

# Convert list to numpy array
X = np.array(X)
y = np.array(y)

# Lets split the data into train and test set
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=2)

# We will use a batch size of 32. Note: batch size should be a factor of 2.***4,8,16,32,64...***
batch_size = 32
# get the length of the train and validation data
ntrain = len(X_train)
nval = len(X_val)

from keras.applications import InceptionResNetV2
from keras import layers
from keras import models

# conv_base = InceptionV3(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

conv_base = InceptionResNetV2(weights='./inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5',
                              include_top=False, input_shape=(150, 150, 3))
conv_base.summary()
model = models.Sequential()
model.add(conv_base)
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(2, activation='softmax'))
# Lets see our model
model.summary()
print('Number of trainable weights before freezing the conv base:', len(model.trainable_weights))
conv_base.trainable = False
print('Number of trainable weights after freezing the conv base:', len(model.trainable_weights))
# Adam optimizer
# loss function will be categorical cross entropy
# evaluation metric will be accuracy
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Lets create the augmentation configuration
# This helps prevent overfitting, since we are using a small dataset
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

val_datagen = ImageDataGenerator()  # We do not augment validation data. we only perform rescale

# Create the image generators
train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)
val_generator = val_datagen.flow(X_val, y_val, batch_size=batch_size)

# The training part
# We train for 64 epochs with about 100 steps per epoch
history = model.fit_generator(train_generator,
                              steps_per_epoch=ntrain // batch_size,
                              epochs=10,
                              validation_data=val_generator,
                              validation_steps=nval // batch_size)

# Save the model
model.save_weights('model_weights_2.h5')
model.save('model_keras_2.h5')

# get the details form the history object
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

print(acc, val_acc, loss, val_loss, epochs)

acc = [0.723125, 0.7866666666666666, 0.789375, 0.8022916666666666, 0.805625,
       0.8083333333333333, 0.8172916666666666, 0.81875, 0.8202083333333333, 0.8133333333333334]
val_acc = [0.7094594594594594, 0.651541095890411, 0.7705479452054794, 0.6746575342465754, 0.791095890410959,
           0.8073630136986302, 0.8159246575342466, 0.8065068493150684, 0.7251712328767124, 0.7440068493150684]
loss = [0.5674071244398753, 0.4611387696862221, 0.4470943734049797, 0.42513804107904435, 0.41704076061646145,
        0.4183814466993014, 0.40175050020217895, 0.3996272482474645, 0.3893566988905271, 0.3979987812042236]
val_loss = [0.7490346020943409, 2.2767693490198213, 1.2181037868538949, 1.31012781352213, 0.978067207009825,
            0.5370576153062794, 0.8748652687015599, 0.7165581432923879, 1.1822864388766354, 1.091419569433552]

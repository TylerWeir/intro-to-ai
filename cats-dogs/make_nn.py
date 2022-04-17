"""
make_nn.py
Generates a neural network classifier used to classify an
image as a class or a dog.  It takes two command-line arguments:
the directory contianing the training images and the name of the
neural network file to save.
"""
import sys
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.activations import sigmoid
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import AveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Add
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU
from keras.layers import ReLU
from keras.layers import Resizing
from keras.layers import RandomFlip
from keras.layers import RandomRotation
from keras.layers import RandomZoom
from tensorflow.keras.optimizers import Adam
import keras.layers

###########################################
#             DATA PIPELINE               #
###########################################

IMG_WIDTH, IMG_HEIGHT = 100, 100

def process_image(img):
    # Color images
    img = tf.image.decode_jpeg(img, channels=3)

    # convert uint8 tensor to floats in the [0,1] range
    img = tf.image.convert_image_dtype(img, tf.float32)

    # resize
    return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT]) 

def combine_images_labels(filename: tf.Tensor):
    img = tf.io.read_file(filename)
    img = process_image(img)
    label = get_label(filename)
    return img, label

def get_label(filename):
    """Returns the label of an image file.  The label is generated
    assuming cat files start with a c and dog files start with a 
    d."""

    file = filename.numpy().decode("utf-8")
    f = file.split('/')[-1]

    if f[0] == 'c' or f[0] == 'C':
        return 1
    elif f[0] == 'd' or f[0] == 'D':
        return 0
    else:
        print("ERROR: filename must start with \'c\' or \'d\'")
        print(f)
        sys.exit()

def build_dataset(data_path):
    """Builds a tensorflow dataset from a directory of cats and
    dogs images."""

    # make a list of file names
    filenames = os.listdir(data_path)
    filenames = [sys.argv[1] + f for f in filenames]

    ds = tf.data.Dataset.from_tensor_slices(filenames)
    ds_size = ds.cardinality().numpy()

    print("dataset size: ", ds.cardinality().numpy())
    print("splitting into test and train sets...")

    train_ratio = 0.8
    ds_train = ds.take(ds_size*train_ratio)
    ds_test = ds.skip(ds_size*train_ratio)

    print("train size: ", ds_train.cardinality().numpy())
    print("test size: ", ds_test.cardinality().numpy())

    # Process the images in each data set
    # prefectch
    ds_train = ds_train.map( lambda x:
    tf.py_function(func=combine_images_labels,
            inp=[x], Tout=(tf.float32,tf.int64)),
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False)
    ds_train.prefetch(ds_size-ds_size*train_ratio)

    # process the images in the test data set
    ds_test = ds_test.map( lambda x:
    tf.py_function(func=combine_images_labels,
            inp=[x], Tout=(tf.float32,tf.int64)),
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False)
    ds_test.prefetch(ds_size-ds_size*train_ratio)

    # Batch the datasets
    ds_train_batched = ds_train.batch(16).prefetch(tf.data.experimental.AUTOTUNE).cache()
    ds_test_batched = ds_test.batch(16).prefetch(tf.data.experimental.AUTOTUNE).cache()

    # Augment the training data set
    ds_train_batched = ds_train_batched.map(lambda x, y: (data_augmentation(x, training=True), y),
                        num_parallel_calls=tf.data.AUTOTUNE)

    return ds_train_batched, ds_test_batched

###########################################
#            DATA AUGMENTATION            #
###########################################

data_augmentation = keras.Sequential([
    RandomFlip("horizontal", input_shape=(100, 100, 3)),
    RandomRotation(0.1),
    RandomZoom(0.1),
  ])


###########################################
#             MODEL DEFINITION            #
###########################################


def build_model():
    """Builds and returns a CNN classifier model."""
    """
    # Use the quick and dirty 'Functional API'
    inputs = tf.keras.Input(shape=(100, 100, 3))

    # Augment
    x = data_augmentation(inputs)

    # Conv Block 1
    x = Conv2D(filters=6, kernel_size=(5, 5), activation='relu')(x)
    x = AveragePooling2D()(x)

    # Conv Block 2
    x = Conv2D(filters=16, kernel_size=(5, 5), activation='relu')(x)
    x = AveragePooling2D()(x)

    # Neural Net
    x = Flatten()(x)
    x = Dense(units=120)(x)
    x = Dense(units=84)(x)
    x = Dense(units=1)(x)

    classifier = keras.Model(inputs, x)
    """

    inputs = tf.keras.Input(shape=(100, 100, 3))

    # Augment
    x = data_augmentation(inputs)

    # Block 1 (100x100)
    x = Conv2D(64, kernel_size=5, strides=2, padding="same")(x)
    x = ReLU()(x)
    x = MaxPooling2D()(x)
    b1_output = x

    # Block 2 (50x50)
    x = Conv2D(64, kernel_size=3, strides=1, padding="same")(x)
    x = ReLU()(x)
    x = Conv2D(64, kernel_size=3, strides=1, padding="same")(x)
    x = Add()([x, b1_output])
    x = ReLU()(x)
    b2_output = x

    # Block 3 (50x50)
    x = Conv2D(64, kernel_size=3, strides=1, padding="same")(x)
    x = ReLU()(x)
    x = Conv2D(64, kernel_size=3, strides=1, padding="same")(x)
    x = Add()([x, b2_output])
    x = ReLU()(x)
    b3_output = x

    # Block 4 (50x50) x = Conv2D(64, kernel_size=3, strides=1, padding="same")(x)
    x = ReLU()(x)
    x = Conv2D(64, kernel_size=3, strides=1, padding="same")(x)
    x = Add()([x, b3_output])
    x = ReLU()(x)
    b4_output = x

    # Block 5 (50x50)
    x = Conv2D(64, kernel_size=3, strides=1, padding="same")(x)
    x = ReLU()(x)
    x = Conv2D(64, kernel_size=3, strides=1, padding="same")(x)
    x = Add()([x, b4_output])
    x = ReLU()(x)

    # Output
    x = GlobalAveragePooling2D()(x)
    x = Flatten()(x)
    x = Dense(1)(x)

    classifier = keras.Model(inputs, x)

    """
    classifier = tf.keras.Sequential([
      data_augmentation,
      Conv2D(32, 3, padding='same', activation='relu'),
      MaxPooling2D(),
      Conv2D(64, 3, padding='same', activation='relu'),
      MaxPooling2D(),
      Conv2D(128, 3, padding='same', activation='relu'),
      MaxPooling2D(),
      Dropout(0.2),
      Flatten(),
      Dense(units=128, activation='relu'),
      Dense(units=256, activation='relu'),
      Dense(units=128, activation='relu'),
      Dense(units=1)
      ])
      """
    classifier.summary()
    return classifier

def train_model(model, epochs, data, val):
    model.compile(optimizer=Adam(),
            loss=keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=[keras.metrics.BinaryAccuracy()])
    model.fit(data, epochs=epochs, validation_data=val)
    return model

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python make_nn.py [img directory] [file name]')
        sys.exit()

    if not os.path.exists(sys.argv[1]):
        print(f'make_nn: cannot access \'{sys.argv[1]}\': No such file or directory')
        sys.exit()

    train_ds, test_ds = build_dataset(sys.argv[1])
    my_model = build_model()

    train_model(my_model, 20, train_ds, test_ds)

    my_model.save(sys.argv[2])

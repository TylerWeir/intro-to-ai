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
from keras.layers import Flatten
from tensorflow.keras.optimizers import Adam

###########################################
#             DATA PIPELINE               #
###########################################

IMG_WIDTH, IMG_HEIGHT = 100, 100

def process_image(img):
    # Color images
    img = tf.image.decode_jpeg(img, channels=3)

    # convert unit 8 tensor to floats in the [0,1] range
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

    return ds_train_batched, ds_test_batched

###########################################
#             MODEL DEFINITION            #
###########################################

def build_model():
    """Builds and returns a CNN classifier model."""
    # Use the quick and dirty 'Functional API'
    inputs = tf.keras.Input(shape=(100, 100, 3))
    x = Conv2D(filters=6, kernel_size=(5, 5), activation='relu')(inputs)
    x = AveragePooling2D()(x)
    x = Conv2D(filters=16, kernel_size=(5, 5), activation='relu')(x)
    x = AveragePooling2D()(x)
    x = Flatten()(x)
    x = Dense(units=120)(x)
    x = Dense(units=84)(x)
    x = Dense(units=1)(x)

    leNet = keras.Model(inputs, x)
    leNet.summary()
    return leNet

def train_model(model, epochs, data):
    model.compile(optimizer=Adam(),
            loss=keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=[keras.metrics.BinaryAccuracy()])
    model.fit(data, epochs=epochs)
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

    train_model(my_model, 5, train_ds)

import glob
import sys
import numpy as np
import tensorflow as tf
from pathlib import Path
from preproc import *
from sklearn.model_selection import train_test_split
from configs import *


def transform_dataset(dataset):
    features = {
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'char_img': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.string),
    }

    def _parse_image_function(example_proto):
        # Parse the input tf.train.Example proto using the dictionary above.
        return tf.io.parse_single_example(example_proto, features)
    transformed_data = dataset.map(_parse_image_function)
    data_x = []
    data_y = []
    for example in transformed_data:
        img = example["char_img"].numpy()
        label_str = example['label'].numpy()
        label_one_hot = np.zeros(128)
        label_one_hot[ord(label_str)] = 1
        img = np.frombuffer(img, dtype=np.uint8).reshape(
            example['height'], example['width'])
        arr = char_img2arr(img)
        data_x.append(arr)
        data_y.append(label_one_hot)
    data_x = np.array(data_x)
    data_y = np.array(data_y)
    return data_x, data_y


def train_model(input_path, output_path):
    input_path = str(Path(dataset_folder_path, input_path))
    output_path = str(Path(model_folder_path, output_path))
    datasets = glob.glob(str(Path(input_path)))
    labeled_set = tf.data.TFRecordDataset(datasets)
    full_x, full_y = transform_dataset(labeled_set)
    model = prepare_model()
    # train_x, val_x, train_y, val_y = train_test_split(
    #     full_x, full_y, test_size=0.10, random_state=1)
    # model.fit(train_x, train_y,
    #           batch_size=100,
    #           epochs=100,
    #           verbose=1,
    #           validation_data=(val_x, val_y),)
    model.fit(full_x, full_y,
              batch_size=100,
              epochs=100,
              verbose=1,
              )
    model.save(output_path)


def prepare_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                                     activation='relu',
                                     input_shape=(28, 28, 1)))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (5, 5), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1000, activation='relu'))
    model.add(tf.keras.layers.Dense(128, activation='softmax'))
    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=['accuracy'])
    return model


if __name__ == "__main__":
    if not len(sys.argv) == 3:
        print("Error - wrong command line arguments")
        print("Usage: python train.py *.tfrecords output.h5")
        print("OR: python train.py dataset.tfrecords output.h5")
    else:
        path_input = sys.argv[1]
        path_output = sys.argv[2]
        train_model(path_input, path_output)

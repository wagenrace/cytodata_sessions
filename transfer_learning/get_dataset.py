import os
import pathlib
from dataclasses import dataclass

import numpy as np
import tensorflow as tf

'''
This class is mainly just a copy of tensorflow tutorial
https://www.tensorflow.org/tutorials/images/data_augmentation

'''
physical_devices = tf.config.list_physical_devices("GPU")
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

AUTOTUNE = tf.data.experimental.AUTOTUNE

@dataclass
class DataSetGenerator:
    data_location:str=r"D:\data\food"
    batch_size:int=64
    patch_dimension:int=128
    image_type:str="jpg"

    def get_label(self, file_path):
        # Transform label name in one-hot encoded label
        parts = tf.strings.split(file_path, os.path.sep)
        one_hot = parts[-2] == self.class_names
        return one_hot

    def decode_img(self, img, extra_padding=0):
        # Decode and resize image
        if self.image_type == "jpg":
            img = tf.image.decode_jpeg(img, channels=3)
        else:
            img = tf.image.decode_png(img, channels=3)
        img_size = self.patch_dimension + extra_padding
        img = tf.image.resize(img, (img_size, img_size))
        return img

    def process_path(self, file_path, extra_padding=0):
        # Get label and image from image path
        label = self.get_label(file_path)
        img = tf.io.read_file(file_path)
        img = self.decode_img(img, extra_padding=extra_padding)
        return img, label

    def configure_for_performance(self, ds):
        ds = (
            ds.cache()
            .shuffle(buffer_size=1000)
            .batch(self.batch_size)
            .prefetch(buffer_size=AUTOTUNE)
        )
        return ds

    def augment(self, file_path):
        image, label = self.process_path(file_path, 6)
        # Make a new seed
        # Random crop back to the original size
        image = tf.image.random_crop(
            image, size=[self.patch_dimension, self.patch_dimension, 3])
        
        # Random flipping
        tf.image.random_flip_left_right(
            image
        )

        # Random brightness
        image = tf.image.random_brightness(
            image, max_delta=0.5)
        image = tf.clip_by_value(image, 0, 255)
        return image, label

    def get_datasets(self):
        '''
        Expect data in the data location to have the form
        data_location/label/file_name.jpg
        label will be used as the label of the dataset
        '''

        # Load all data locations
        data_dir = pathlib.Path(self.data_location)
        if self.image_type == "jpg":
            image_count = len(list(data_dir.glob("*/*.jpg")))
            list_ds = tf.data.Dataset.list_files(str(data_dir / "*/*.jpg"), shuffle=False)

        else:
            image_count = len(list(data_dir.glob("*/*.png")))
            list_ds = tf.data.Dataset.list_files(str(data_dir / "*/*.png"), shuffle=False)

        # Get all unique class names
        self.class_names = np.array(
            sorted([item.name for item in data_dir.glob("*")])
        )

        # shuffle data and split into train and validation
        list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)
        val_size = int(image_count * 0.3)
        train_ds = list_ds.skip(val_size)
        val_ds = list_ds.take(val_size)

        # Create validation and train dataset
        train_ds = train_ds.map(self.augment, num_parallel_calls=AUTOTUNE)
        val_ds = val_ds.map(self.process_path, num_parallel_calls=AUTOTUNE)

        train_ds = self.configure_for_performance(train_ds)
        val_ds = self.configure_for_performance(val_ds)

        return train_ds, val_ds

import numpy as np
import os
import tensorflow as tf
import logging
from sklearn.utils import class_weight


class Dataset(object):
    def __init__(self, path: str, config: dict, shuffle: bool=True, expected_img_width: int=256, expected_img_height: int=256, expected_channels: int=3, class_names: list=['good', 'bad']) -> None:
        super().__init__()
        self.path = path
        self.class_names = class_names
        self.expected_img_width = expected_img_width
        self.expected_img_height = expected_img_height
        if expected_channels == 1:
            self.color_mode = 'grayscale'
        elif expected_channels == 3:
            self.color_mode = 'rgb'
        elif expected_channels == 4:
            self.color_mode = 'rgba'
        else:
            raise Exception('Invalid channel count provided. Must be 1, 3 or 4')
        self.shuffle = shuffle
        self.config = config

    def input_preprocess(self, image, label):
        label = tf.one_hot(tf.cast(label, tf.int32), self.config['parameters']['target_classes'], axis=1)
        label = tf.reshape(label, [-1, 2]) # convert 3d to 2d
        return image, label

    def load(self):
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.path,
            shuffle=self.shuffle,
            color_mode=self.color_mode,
            class_names=self.class_names,
            label_mode='binary',
            seed=self.config['seed'],
            image_size=(self.expected_img_height, self.expected_img_width),
            batch_size=self.config['parameters']['batch_size']
        )
        # ds = ds.map(self.input_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds = ds.cache()
        if self.shuffle:
            ds = ds.shuffle(1000)
        return ds.prefetch(buffer_size=AUTOTUNE)

    def get_weights(self, weighting='balanced'):
        found_folders = list()
        # check subfolders
        for subdir in self.class_names:
            if os.path.isdir(os.path.join(self.path, subdir)):
                found_folders.append(os.path.join(self.path, subdir))
        # get counts
        counts = dict()
        labels = list()
        for i, folder in enumerate(found_folders):
            item_count = len(os.listdir(folder))
            counts[i] = item_count
            labels.extend([i]*item_count)
        weights = class_weight.compute_class_weight(
            class_weight=weighting, 
            classes=np.unique(labels),
            y=labels
        )
        calculated_weights = dict({0: weights[0], 1: weights[1]})
        return calculated_weights, counts

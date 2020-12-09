"""
Builds TFRecords from image folder
"""

import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from config import (
    DATAPATH,
    SEED,
    NUM_CLASSES,
    TFRECORDS_TRAIN_PATH,
    TFRECORDS_VAL_PATH,
)


def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = (
            value.numpy()
        )  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


if __name__ == "__main__":

    seed_everything(SEED)

    NUM_IMAGES_VAL = 200  # per class

    filename_train = "train.tfrecord"
    filename_val = "val.tfrecord"

    _feature_columns_train = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "target": tf.io.FixedLenFeature([], tf.int64),
        "image_name": tf.io.FixedLenFeature([], tf.string),
    }

    df = pd.read_csv(
        os.path.join(DATAPATH, "train.csv"), header=0, index_col=None
    )

    upsample_multipliers = [5, 2, 2, 1, 2]  # considering 200 val images

    # split inital dataset into train and val
    dfs_train = []
    dfs_val = []
    for label in range(NUM_CLASSES):
        sampled = df[df["label"] == label].sample(frac=1, random_state=SEED)

        if label == 3:
            sampled = sampled.sample(n=5000, random_state=SEED)

        dfs_val.append(sampled[:NUM_IMAGES_VAL])
        dfs_train.append(
            pd.concat(
                [sampled[NUM_IMAGES_VAL:]] * upsample_multipliers[label]
            ).sample(frac=1, random_state=SEED)
        )

    dfs = {
        "train": pd.concat(dfs_train).sample(frac=1, random_state=SEED),
        "val": pd.concat(dfs_val).sample(frac=1, random_state=SEED),
    }

    paths = {"train": TFRECORDS_TRAIN_PATH, "val": TFRECORDS_VAL_PATH}

    for key, val in dfs.items():
        with tf.io.TFRecordWriter(
            os.path.join(paths[key], f"{key}.tfrecords")
        ) as writer:

            val.to_csv(f"{key}.csv", sep=",")

            for row, col in val.iterrows():
                image_string = open(
                    os.path.join(DATAPATH, "train_images", col["image_id"]),
                    "rb",
                ).read()

                feature = {
                    "image_name": _bytes_feature(
                        col["image_id"].encode("utf-8")
                    ),
                    "image": _bytes_feature(image_string),
                    "target": _int64_feature(col["label"]),
                }
                example = tf.train.Example(
                    features=tf.train.Features(feature=feature)
                )
                writer.write(example.SerializeToString())

        print(f"{key} has been written")

import os
import tensorflow as tf

from augs import data_augment
from config import SIZE, NUM_CLASSES, NUM_IMAGES


_feature_columns_train = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
    "image_name": tf.io.FixedLenFeature([], tf.string),
}

_feature_columns_test = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
}


def _parse_example(raw_record):
    """
    Depricated. Parses records from unnmapped dataset.
    """
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    return example


def _parse_image_function_train(example_proto):
    """
    Parses the input tf.train.Example proto using the dictionary provided.
    """
    features = tf.io.parse_single_example(example_proto, _feature_columns_train)
    image = tf.io.decode_jpeg(features["image"], channels=3)
    target = tf.cast(features["target"], tf.uint8)
    return image, target


def _parse_image_function_test(example_proto):
    """
    Parses the input tf.train.Example proto using the dictionary provided.
    """
    features = tf.io.parse_single_example(example_proto, _feature_columns_train)
    image = tf.io.decode_jpeg(features["image"], channels=3)
    return image


def init_dataset(
    path: str,
    is_target=True,
    shuffle=False,
    augment=False,
):
    # upload TFRecords files
    read_obj = (
        [os.path.join(path, x) for x in os.listdir(path)]
        if os.path.isdir
        else path
    )

    # init tf.data API
    dataset = tf.data.TFRecordDataset(read_obj)

    # parse dataset records
    if is_target:
        dataset = dataset.map(_parse_image_function_train)
    else:
        dataset = dataset.map(_parse_image_function_test)

    if shuffle:
        dataset.shuffle(buffer_size=2048)

    if augment:
        dataset.map(data_augment)

    return dataset


# TODO: add stratified split to dataset


def split_dataset(
    dataset, train_size: float = 0.8, num_images: int = NUM_IMAGES
):
    assert (1 - train_size) > 0, "Train size is to large"

    train_size = int(train_size * num_images)

    dataset = dataset.shuffle(buffer_size=2048)

    train_dataset = dataset.take(train_size)
    test_dataset = dataset.skip(train_size)

    return train_dataset, test_dataset


# One-hot / categorical encoding
def input_preprocess(image, label):
    label = tf.cast(label, tf.int32)
    image = tf.image.resize(image, SIZE)
    label = tf.one_hot(label, NUM_CLASSES, dtype=tf.uint8)

    image, label = data_augment(image, label)
    return image, label

import tensorflow as tf
import math
import tensorflow.keras.backend as K

from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.models import Sequential

from config import WIDTH, HEIGHT, CHANNELS, SEED

img_augmentation_sequential = Sequential(
    [
        preprocessing.RandomRotation(factor=0.15, seed=SEED),
        preprocessing.RandomTranslation(height_factor=0.1, width_factor=0.1),
        preprocessing.RandomFlip(),
        preprocessing.RandomContrast(factor=0.1),
        preprocessing.CenterCrop(height=HEIGHT, width=WIDTH),
        preprocessing.RandomZoom(
            height_factor=(0.2, 0.5),
            width_factor=(-0.3, 0.3),  # None to preserve ratio
            fill_mode="constant",
            seed=SEED,
        ),
    ],
    name="img_augmentation",
)


def image_augmentation_functional(image, label):

    # Image Adjustments
    image = tf.image.random_brightness(image=image, max_delta=0.3, seed=SEED)
    image = tf.image.random_contrast(
        image=image, lower=0.2, upper=0.5, seed=SEED
    )
    image = tf.image.random_saturation(
        image=image, lower=0.2, upper=0.5, seed=SEED
    )
    image = tf.image.random_hue(
        image=image, max_delta=0.3, seed=SEED
    )  # max_delta must be in the interval [0, 0.5]

    # Flipping, Rotating and Transposing
    image = tf.image.random_flip_left_right(image=image, seed=SEED)
    image = tf.image.random_flip_up_down(image=image, seed=SEED)

    return image, label


# TODO: add TTA

# data augmentation @cdeotte kernel: https://www.kaggle.com/cdeotte/rotation-augmentation-gpu-tpu-0-96
def transform_rotation(image, height, rotation):
    # input image - is one image of size [dim,dim,3] not a batch of [b,dim,dim,3]
    # output - image randomly rotated
    DIM = height
    XDIM = DIM % 2  # fix for size 331

    rotation = rotation * tf.random.uniform([1], dtype="float32")
    # CONVERT DEGREES TO RADIANS
    rotation = math.pi * rotation / 180.0

    # ROTATION MATRIX
    c1 = tf.math.cos(rotation)
    s1 = tf.math.sin(rotation)
    one = tf.constant([1], dtype="float32")
    zero = tf.constant([0], dtype="float32")
    rotation_matrix = tf.reshape(
        tf.concat([c1, s1, zero, -s1, c1, zero, zero, zero, one], axis=0),
        [3, 3],
    )

    # LIST DESTINATION PIXEL INDICES
    x = tf.repeat(tf.range(DIM // 2, -DIM // 2, -1), DIM)
    y = tf.tile(tf.range(-DIM // 2, DIM // 2), [DIM])
    z = tf.ones([DIM * DIM], dtype="int32")
    idx = tf.stack([x, y, z])

    # ROTATE DESTINATION PIXELS ONTO ORIGIN PIXELS
    idx2 = K.dot(rotation_matrix, tf.cast(idx, dtype="float32"))
    idx2 = K.cast(idx2, dtype="int32")
    idx2 = K.clip(idx2, -DIM // 2 + XDIM + 1, DIM // 2)

    # FIND ORIGIN PIXEL VALUES
    idx3 = tf.stack(
        [
            DIM // 2 - idx2[0],
            DIM // 2 - 1 + idx2[1],
        ]
    )
    d = tf.gather_nd(image, tf.transpose(idx3))

    return tf.reshape(d, [DIM, DIM, 3])


def transform_shear(image, height, shear):
    # input image - is one image of size [dim,dim,3] not a batch of [b,dim,dim,3]
    # output - image randomly sheared
    DIM = height
    XDIM = DIM % 2  # fix for size 331

    shear = shear * tf.random.uniform([1], dtype="float32")
    shear = math.pi * shear / 180.0

    # SHEAR MATRIX
    one = tf.constant([1], dtype="float32")
    zero = tf.constant([0], dtype="float32")
    c2 = tf.math.cos(shear)
    s2 = tf.math.sin(shear)
    shear_matrix = tf.reshape(
        tf.concat([one, s2, zero, zero, c2, zero, zero, zero, one], axis=0),
        [3, 3],
    )

    # LIST DESTINATION PIXEL INDICES
    x = tf.repeat(tf.range(DIM // 2, -DIM // 2, -1), DIM)
    y = tf.tile(tf.range(-DIM // 2, DIM // 2), [DIM])
    z = tf.ones([DIM * DIM], dtype="int32")
    idx = tf.stack([x, y, z])

    # ROTATE DESTINATION PIXELS ONTO ORIGIN PIXELS
    idx2 = K.dot(shear_matrix, tf.cast(idx, dtype="float32"))
    idx2 = K.cast(idx2, dtype="int32")
    idx2 = K.clip(idx2, -DIM // 2 + XDIM + 1, DIM // 2)

    # FIND ORIGIN PIXEL VALUES
    idx3 = tf.stack(
        [
            DIM // 2 - idx2[0],
            DIM // 2 - 1 + idx2[1],
        ]
    )
    d = tf.gather_nd(image, tf.transpose(idx3))

    return tf.reshape(d, [DIM, DIM, 3])


def data_augment(image, label):
    p_rotation = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    p_spatial = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    p_rotate = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    p_pixel_1 = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    p_pixel_2 = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    p_pixel_3 = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    p_shear = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    p_crop = tf.random.uniform([], 0, 1.0, dtype=tf.float32)

    # Shear
    if p_shear > 0.2:
        if p_shear > 0.6:
            image = transform_shear(image, HEIGHT, shear=20.0)
        else:
            image = transform_shear(image, HEIGHT, shear=-20.0)

    # Rotation
    if p_rotation > 0.2:
        if p_rotation > 0.6:
            image = transform_rotation(image, HEIGHT, rotation=45.0)
        else:
            image = transform_rotation(image, HEIGHT, rotation=-45.0)

    # Flips
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    if p_spatial > 0.75:
        image = tf.image.transpose(image)

    # Rotates
    if p_rotate > 0.75:
        image = tf.image.rot90(image, k=3)  # rotate 270º
    elif p_rotate > 0.5:
        image = tf.image.rot90(image, k=2)  # rotate 180º
    elif p_rotate > 0.25:
        image = tf.image.rot90(image, k=1)  # rotate 90º

    # Pixel-level transforms
    if p_pixel_1 >= 0.4:
        image = tf.image.random_saturation(image, lower=0.7, upper=1.3)
    if p_pixel_2 >= 0.4:
        image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    if p_pixel_3 >= 0.4:
        image = tf.image.random_brightness(image, max_delta=0.1)

    # Crops
    if p_crop > 0.7:
        if p_crop > 0.9:
            image = tf.image.central_crop(image, central_fraction=0.6)
        elif p_crop > 0.8:
            image = tf.image.central_crop(image, central_fraction=0.7)
        else:
            image = tf.image.central_crop(image, central_fraction=0.8)
    elif p_crop > 0.4:
        crop_size = tf.random.uniform(
            [], int(HEIGHT * 0.6), HEIGHT, dtype=tf.int32
        )
        image = tf.image.random_crop(
            image, size=[crop_size, crop_size, CHANNELS]
        )

    image = tf.image.resize(image, size=[HEIGHT, WIDTH])

    return image, label


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from dataset import ds_train

    for image, label in ds_train.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            aug_img = img_augmentation_sequential(tf.expand_dims(image, axis=0))
            plt.imshow(aug_img[0].numpy().astype("uint8"))
            plt.title("{}".format(str(label.numpy())))
            plt.axis("off")

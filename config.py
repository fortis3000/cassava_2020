import os

WORKDIR = "."

MODELS_FOLDER = os.path.join(WORKDIR, "models")
CHECKPOINT_FOLDER = os.path.join(WORKDIR, "chkp")

if not os.path.exists(MODELS_FOLDER):
    os.makedirs(MODELS_FOLDER)
    os.makedirs(CHECKPOINT_FOLDER)

DATAPATH = os.path.join(WORKDIR, "data", "raw")

if os.path.exists("/kaggle/input/cassava-leaf-disease-classification"):
    DATAPATH = "/kaggle/input/cassava-leaf-disease-classification"

TFRECORDS_TRAIN_PATH = os.path.join(WORKDIR, "train_tfrecords_upsampled")
TFRECORDS_VAL_PATH = os.path.join(WORKDIR, "val_tfrecords_upsampled")

if not os.path.exists(TFRECORDS_TRAIN_PATH):
    os.makedirs(TFRECORDS_TRAIN_PATH)
    os.makedirs(TFRECORDS_VAL_PATH)


MODEL_KIND = "EfficientNetB0"

batch_sizes = {
    "EfficientNetB0": 32,  # checked (local)
    "EfficientNetB1": 16,
    "EfficientNetB2": 16,
    "EfficientNetB3": 16,
    "EfficientNetB4": 16,
    "EfficientNetB5": 8,  # checked (for Kaggle)
    "EfficientNetB6": 8,
    "EfficientNetB7": 8,
}

sizes = {
    "EfficientNetB0": 224,
    "EfficientNetB1": 240,
    "EfficientNetB2": 260,
    "EfficientNetB3": 300,
    "EfficientNetB4": 380,
    "EfficientNetB5": 456,
    "EfficientNetB6": 528,
    "EfficientNetB7": 600,
}

WIDTH = HEIGHT = IMG_SIZE = sizes[MODEL_KIND]

CHANNELS = 3
SIZE = (IMG_SIZE, IMG_SIZE)
NUM_CLASSES = 5
NUM_IMAGES = 21397  # all images

TRAIN_SIZE = 0.8

SEED = 42

LEARNING_RATE = 1e-4
LR_ALPHA = 1e-3
BATCH_SIZE = batch_sizes[MODEL_KIND]
EPOCHS = 60  # max 100

# https://datascience.stackexchange.com/questions/13490/how-to-set-class-weights-for-imbalanced-classes-in-keras
WEIGHTS = [1] * 5

# uniform weights
# [1] * 5
# normalized weights
# [0.40707, 0.20214, 0.18545, 0.03363, 0.17171]
# multiples weights
# [12.10488, 6.01096, 5.51467, 1.0, 5.10594]
# sklearn weights
# [3.93689, 1.95496 , 1.79355, 0.32523 , 1.66061]
# sklearn upsampled
# [3, 1.5, 1.5, 1, 1.5]

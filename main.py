import os
import datetime as dt
import json
import random
import warnings

import numpy as np
import tensorflow as tf

from dataset import init_dataset, input_preprocess, split_dataset
from model import build_model_transfer
from config import (
    DATAPATH,
    NUM_CLASSES,
    SEED,
    BATCH_SIZE,
    EPOCHS,
    MODELS_FOLDER,
    TRAIN_SIZE,
    WEIGHTS,
    MODEL_KIND
)

import matplotlib.pyplot as plt

# def cosine_schedule(epoch, lr):
#   if epoch < 10:
#     return lr
#   else:
#     return lr * tf.math.exp(-0.1)


def get_dt_str():
    """Returns current date and time in string format"""
    return str(dt.datetime.now()).split(".")[0].replace(" ", "_")


def plot_hist(hist, filename: str = None):
    plt.plot(hist.history["categorical_accuracy"])
    plt.plot(hist.history["val_categorical_accuracy"])
    plt.plot(hist.history["f1_macro"])
    plt.plot(hist.history["val_f1_macro"])
    plt.plot(hist.history["f1_macro_median"])
    plt.plot(hist.history["val_f1_macro_median"])
    plt.plot(hist.history["f1_macro_weighted"])
    plt.plot(hist.history["val_f1_macro_weighted"])
    plt.plot(hist.history["f1_micro"])
    plt.plot(hist.history["val_f1_micro"])
    plt.title("model metrics")
    plt.ylabel("percentage")
    plt.xlabel("epoch")
    plt.legend(
        [
            "categorical_accuracy",
            "val_categorical_accuracy",
            "f1_macro",
            "val_f1_macro",
            "f1_macro_median",
            "val_f1_macro_median",
            "f1_macro_weighted",
            "val_f1_macro_weighted",
            "f1_micro",
            "val_f1_micro",
        ],
        loc="upper left",
    )

    if filename:
        plt.savefig(filename)
    else:
        plt.show()


def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"


seed_everything(SEED)
warnings.filterwarnings("ignore")

if __name__ == "__main__":

    DATESTRING = get_dt_str()

    with open(
        os.path.join(DATAPATH, "label_num_to_disease_map.json"), "r"
    ) as f:
        labels = json.load(f)

    # Dataset
    dataset = init_dataset(
        os.path.join(DATAPATH, "train_tfrecords"),
        is_target=True,
        shuffle=True,
    )

    ds_train, ds_test = split_dataset(dataset, train_size=TRAIN_SIZE)
    ds_train = ds_train.map(
        input_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    ds_train = ds_train.batch(batch_size=BATCH_SIZE, drop_remainder=True)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

    ds_test = ds_test.map(input_preprocess)
    ds_test = ds_test.batch(batch_size=BATCH_SIZE, drop_remainder=True)

    ds_train.prefetch(2)
    ds_test.prefetch(2)

    model = build_model_transfer(num_classes=NUM_CLASSES)

    print("Start training")

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=10),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(
                MODELS_FOLDER, "model.{epoch:02d}-{val_loss:.2f}.h5"
            )
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join("logs", DATESTRING)
        ),
        # tf.keras.callbacks.LearningRateScheduler(schedule=cosine_schedule)
    ]

    hist = model.fit(
        ds_train,
        epochs=EPOCHS,
        validation_data=ds_test,
        callbacks=callbacks,
        verbose=1,
        class_weight={key: val for key, val in enumerate(WEIGHTS)},
    )

    plot_hist(hist, "hist.jpg")

    # Saving trained model
    model.save(os.path.join("models", DATESTRING + MODEL_KIND))

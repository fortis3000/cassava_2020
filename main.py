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
    TRAIN_SIZE,
    WEIGHTS,
    MODEL_KIND,
    LEARNING_RATE,
    MODELS_FOLDER,
    CHECKPOINT_FOLDER,
    LR_ALPHA,
    TFRECORDS_TRAIN_PATH,
    TFRECORDS_VAL_PATH,
)

import matplotlib.pyplot as plt


def get_dt_str():
    """Returns current date and time in string format"""
    return str(dt.datetime.now()).split(".")[0].replace(" ", "_")


def plot_hist(hist, filename: str = None):
    plt.plot(hist.history["loss"])
    plt.plot(hist.history["val_loss"])
    plt.plot(hist.history["f1_macro"])
    plt.plot(hist.history["val_f1_macro"])
    plt.plot(hist.history["f1_macro_median"])
    plt.plot(hist.history["val_f1_macro_median"])
    plt.plot(hist.history["f1_macro_weighted"])
    plt.plot(hist.history["val_f1_macro_weighted"])
    plt.title("model metrics")
    plt.ylabel("percentage")
    plt.xlabel("epoch")
    plt.legend(
        [
            "loss",
            "val_loss",
            "f1_macro",
            "val_f1_macro",
            "f1_macro_median",
            "val_f1_macro_median",
            "f1_macro_weighted",
            "val_f1_macro_weighted",
        ],
        loc="upper left",
    )

    if filename:
        plt.savefig(filename)
    else:
        plt.show()


def plot_hist_lr(hist, filename: str = None):
    nb_epoch = len(hist.history["loss"])
    learning_rate = hist.history["lr"]
    xc = range(nb_epoch)
    plt.figure(3, figsize=(7, 5))
    plt.plot(xc, learning_rate)
    plt.xlabel("num of Epochs")
    plt.ylabel("learning rate")
    plt.title("Learning rate")
    plt.grid(True)
    plt.style.use(["seaborn-ticks"])

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


if __name__ == "__main__":

    seed_everything(SEED)
    warnings.filterwarnings("ignore")

    DATESTRING = get_dt_str()

    with open(
        os.path.join(DATAPATH, "label_num_to_disease_map.json"), "r"
    ) as f:
        labels = json.load(f)

    # Dataset
    # dataset = init_dataset(
    #     os.path.join(DATAPATH, "train_tfrecords"),
    #     is_target=True,
    #     shuffle=True,
    # )
    #
    # ds_train, ds_test = split_dataset(dataset, train_size=TRAIN_SIZE)

    ds_train = init_dataset(
        os.path.join(TFRECORDS_TRAIN_PATH),
        is_target=True,
        shuffle=True,
        augment=True
    )
    ds_train = ds_train.map(
        input_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    ds_train = ds_train.batch(batch_size=BATCH_SIZE, drop_remainder=True)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

    # ValueError: When providing an infinite dataset, you must specify the number
    # of steps to run (if you did not intend to create an infinite dataset,
    # make sure to not call `repeat()` on the dataset).
    # ds_train = ds_train.repeat()

    ds_test = init_dataset(
        os.path.join(TFRECORDS_VAL_PATH),
        is_target=True,
        shuffle=False,
        augment=False
    )
    ds_test = ds_test.map(input_preprocess)
    ds_test = ds_test.batch(batch_size=BATCH_SIZE, drop_remainder=True)
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

    model = build_model_transfer(num_classes=NUM_CLASSES)

    # cosine decay schedule
    # on which step scheduler is supposed to reach alpha * LEARNING_RATE
    # !!! DECAY_STEPS == EPOCHS !!!
    decay_steps = EPOCHS  # * int(NUM_IMAGES * TRAIN_SIZE) // BATCH_SIZE
    scheduler = tf.keras.experimental.CosineDecay(
        LEARNING_RATE,
        decay_steps,
        alpha=LR_ALPHA,
        name=None,  # 0.0 * LEARNING_RATE
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=10),
        tf.keras.callbacks.LearningRateScheduler(scheduler),
        # tf.keras.callbacks.ReduceLROnPlateau(
        #     monitor="val_loss",
        #     factor=0.1,
        #     patience=20,
        #     verbose=1,
        #     mode="auto",
        #     min_delta=1e-4,
        #     cooldown=0,
        #     min_lr=1e-7,
        # ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(
                CHECKPOINT_FOLDER, "model.{epoch:02d}-{val_loss:.2f}.h5"
            )
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join("logs", DATESTRING)
        ),
    ]

    hist = model.fit(
        ds_train,
        epochs=EPOCHS,
        validation_data=ds_test,
        callbacks=callbacks,
        verbose=1,
        # class_weight={key: val for key, val in enumerate(WEIGHTS)},
    )

    # Saving trained model
    model.save(
        os.path.join(MODELS_FOLDER, DATESTRING + "_" + MODEL_KIND),
    )

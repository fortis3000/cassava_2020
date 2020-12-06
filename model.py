"""
https://arxiv.org/pdf/1911.03347.pdf
"""
from tensorflow.keras.applications import *
from tensorflow.keras import layers
import tensorflow.keras as k
import tensorflow as tf

from augs import img_augmentation
from config import (
    IMG_SIZE,
    BATCH_SIZE,
    WEIGHTS,
    EPOCHS,
    LEARNING_RATE,
    NUM_IMAGES,
    TRAIN_SIZE,
    MODEL_KIND,
)

from f1_score_tf import f1_score_binary, _f1_score_formula, _get_median

models_dict = {
    "EfficientNetB0": EfficientNetB0,
    "EfficientNetB1": EfficientNetB1,
    "EfficientNetB2": EfficientNetB2,
    "EfficientNetB3": EfficientNetB3,
    "EfficientNetB4": EfficientNetB4,
    "EfficientNetB5": EfficientNetB5,
    "EfficientNetB6": EfficientNetB6,
    "EfficientNetB7": EfficientNetB7,
}


def f1_micro(y_true, y_pred):
    """
    Global F1 score is the sum of per batch f1 scores.

    Calculates global f1 score from mean global precision and recall.
    Not the best option for imbalanced multiclass classification.
    """
    _, precisions, recalls = f1_score_binary(
        y_true=y_true, y_pred=y_pred, out_precision_recall=True
    )
    global_precision = tf.reduce_mean(precisions)
    global_recall = tf.reduce_mean(recalls)
    f1_score = _f1_score_formula(
        precision=global_precision, recall=global_recall
    )

    return f1_score


def f1_macro(y_true, y_pred):
    """
    Global F1 score is the sum of per batch f1 scores.
    Args:
        y_true: Tensor("Cast_1:0", shape=(16, 5), dtype=float32)
        y_pred: Tensor("EfficientNet/pred/Softmax:0", shape=(16, 5), dtype=float32)

    Where 16 means batch size and 5 means the number of classes.
    """
    f1_scores = f1_score_binary(
        y_true=y_true, y_pred=y_pred, out_precision_recall=False
    )
    return tf.reduce_mean(f1_scores, axis=-1)


def f1_macro_median(y_true, y_pred):
    """
    Global F1 score is the sum of per batch f1 scores.
    Args:
        y_true: Tensor("Cast_1:0", shape=(16, 5), dtype=float32)
        y_pred: Tensor("EfficientNet/pred/Softmax:0", shape=(16, 5), dtype=float32)

    Where 16 means batch size and 5 means the number of classes.
    """
    f1_scores = f1_score_binary(
        y_true=y_true, y_pred=y_pred, out_precision_recall=False
    )
    return _get_median(f1_scores)


def f1_macro_weighted(y_true, y_pred):
    """
    Global F1 score is the sum of per batch f1 scores.
    Args:
        y_true: Tensor("Cast_1:0", shape=(16, 5), dtype=float32)
        y_pred: Tensor("EfficientNet/pred/Softmax:0", shape=(16, 5), dtype=float32)

    Where 16 means batch size and 5 means the number of classes.
    """
    f1_scores = f1_score_binary(
        y_true=y_true, y_pred=y_pred, out_precision_recall=False
    )
    f1_scores = tf.multiply(f1_scores, tf.constant(WEIGHTS, dtype=tf.float32))
    return tf.reduce_mean(f1_scores, axis=-1)


# https://github.com/keras-team/keras/issues/2115#issuecomment-204060456
# from itertools import product
# from functools import partial
# import numpy as np
# import tensorflow.keras.backend as K
#
#
# def w_categorical_crossentropy(y_true, y_pred, weights):
#     nb_cl = len(weights)
#     final_mask = K.zeros_like(y_pred[:, 0])
#     y_pred_max = K.max(y_pred, axis=1)
#     y_pred_max = K.reshape(y_pred_max, (K.shape(y_pred)[0], 1))
#     y_pred_max_mat = K.equal(y_pred, y_pred_max)
#     for c_p, c_t in product(range(nb_cl), range(nb_cl)):
#         final_mask += (weights[c_t, c_p] * y_pred_max_mat[:, c_p] * y_true[:, c_t])
#     return K.categorical_crossentropy(y_pred, y_true) * final_mask
#
# w_array = np.ones((10,10))
# w_array[1, 7] = 1.2
# w_array[7, 1] = 1.2
#
# ncce = partial(w_categorical_crossentropy, weights=np.array(WEIGHTS))


def build_model_transfer(num_classes):
    inputs = k.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = img_augmentation(inputs)
    # x = inputs

    model = models_dict[MODEL_KIND](
        include_top=False, input_tensor=x, weights="imagenet"
    )

    print("model initialized")
    # freeze model
    model.trainable = False

    # Rebuild top (head)
    x = k.layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = k.layers.BatchNormalization()(x)

    top_dropout_rate = 0.2
    x = k.layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    # TODO: add layers with swish
    #  https://medium.com/the-artificial-impostor/more-memory-efficient-swish
    #  -activation-function-e07c22c12a76
    # https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection
    # /discussion/111292
    outputs = layers.Dense(num_classes, activation="softmax", name="pred")(x)

    print("head added")
    # Compile
    model = k.Model(inputs, outputs, name="EfficientNet")

    # cosine decay schedule
    decay_steps = EPOCHS * int(NUM_IMAGES * TRAIN_SIZE) // BATCH_SIZE
    scheduler = tf.keras.experimental.CosineDecay(
        LEARNING_RATE, decay_steps, alpha=0.0, name=None
    )

    optimizer = k.optimizers.Adam(learning_rate=scheduler)

    metrics = [
        k.metrics.CategoricalAccuracy(),
        # k.metrics.Precision(),
        # k.metrics.Recall(),
        f1_macro,
        f1_macro_median,
        f1_macro_weighted,
        f1_micro,
    ]

    print("start compiling")
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",  # TODO: weighted cross-entropy
        metrics=metrics,
    )

    print("end compiling")
    #     print(model.summary())
    return model


def build_model(num_classes):
    # TODO: add model for TPU training
    pass

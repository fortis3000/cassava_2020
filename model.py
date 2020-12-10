"""
https://arxiv.org/pdf/1911.03347.pdf
"""
from tensorflow.keras.applications import *
from tensorflow.keras import layers
import tensorflow.keras as k
import tensorflow as tf

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


def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr

    return lr


def build_model_transfer(num_classes):
    inputs = k.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    # x = img_augmentation(inputs)
    x = inputs

    model = models_dict[MODEL_KIND](
        weights="imagenet",  # f"{MODEL_KIND.lower()}_notop.h5",
        include_top=False,
        input_tensor=x,
    )

    print("model initialized")
    # freeze model
    model.trainable = True

    # freeze some layers
    # for layer in model.layers[-40:]:
    #     if not isinstance(layer, layers.BatchNormalization):
    #         layer.trainable = True

    # Rebuild top (head)
    x = k.layers.GlobalAveragePooling2D(name="avg_pool_head")(model.output)

    outputs = layers.Dense(
        num_classes, activation="softmax", name="softmax_head"
    )(x)

    print("head added")
    # Compile
    model = k.Model(inputs, outputs, name="EfficientNet")

    optimizer = k.optimizers.Adam(learning_rate=LEARNING_RATE)
    loss = tf.losses.CategoricalCrossentropy(
        from_logits=False, label_smoothing=0.01, name="categorical_crossentropy"
    )

    lr_metric = get_lr_metric(optimizer)

    # displayed in Tensorboard
    metrics = [
        k.metrics.CategoricalCrossentropy(),
        k.metrics.CategoricalAccuracy(),
        k.metrics.AUC(
            num_thresholds=200,
            curve="ROC",
            summation_method="interpolation",
            name=None,
            dtype=None,
            thresholds=None,
            multi_label=True,
            label_weights=None,
        ),
        f1_macro,
        f1_macro_median,
        f1_macro_weighted,
        f1_micro,
        lr_metric,
    ]

    print("start compiling")
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics,
    )

    print("end compiling")
    #     print(model.summary())
    return model


def build_model(num_classes):
    # TODO: add model for TPU training
    pass

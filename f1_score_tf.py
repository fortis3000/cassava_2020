"""
https://arxiv.org/pdf/1911.03347.pdf
"""

import tensorflow as tf


@tf.function
def _f1_score_formula(precision: tf.Tensor, recall: tf.Tensor) -> tf.Tensor:
    return tf.multiply(
        tf.constant(2.0, dtype=tf.float32),
        tf.divide(tf.multiply(precision, recall), tf.add(precision, recall)),
    )


@tf.function
def _get_median(v: tf.Tensor) -> tf.Tensor:
    """
    Returns 50 percentile (median) of the vector
    Note: do not use decorators here - bug
    """
    v = tf.reshape(v, [-1])
    m = v.get_shape()[0] // 2
    vals = tf.math.top_k(v, m, sorted=False).values
    median = tf.reduce_min(vals)
    return median


@tf.function
def f1_score_binary(
    y_true: tf.Tensor, y_pred: tf.Tensor, out_precision_recall: bool = False
):
    """
    Computes f1 score on binary inputs
    """

    tf.debugging.assert_equal(
        tf.rank(y_true),
        tf.rank(y_pred),
        message="y_true and y_pred are to be the same size",
    )

    # true positive = predicted(1) * ground_truth(1) = 1
    true_positive = tf.reduce_sum(tf.multiply(y_true, y_pred), axis=0)
    true_positive = tf.cast(true_positive, dtype=tf.float32)

    # false positive = predicted(1) - ground_truth(0) > 0
    diff_false_pos = tf.math.greater(tf.subtract(y_pred, y_true), 0)
    diff_false_pos = tf.cast(diff_false_pos, dtype=tf.float32)
    false_positive = tf.reduce_sum(diff_false_pos, axis=0)

    # false negative = ground_truth(1) - predicted(0) > 0
    # order change instead of tf.math.less due to non-negative dtype
    diff_false_neg = tf.math.greater(tf.subtract(y_true, y_pred), 0)
    diff_false_neg = tf.cast(diff_false_neg, dtype=tf.float32)
    false_negative = tf.reduce_sum(diff_false_neg, axis=0)

    precision = tf.divide(true_positive, tf.add(true_positive, false_positive))
    recall = tf.divide(true_positive, tf.add(true_positive, false_negative))

    # in case of nan (batch computing)
    precision = tf.where(
        tf.math.is_nan(precision), tf.zeros_like(precision), precision
    )
    recall = tf.where(tf.math.is_nan(recall), tf.zeros_like(recall), recall)

    # 2*((precision * recall) / (precision+recall))
    f1_score = _f1_score_formula(precision, recall)

    # fill nans
    f1_score = tf.where(
        tf.math.is_nan(f1_score), tf.zeros_like(f1_score), f1_score
    )

    if out_precision_recall:
        return f1_score, precision, recall

    return f1_score


@tf.function
def f1_score_multiclass(
    classes: tf.Tensor,
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    averaging: str = "macro",
    group_func: str = "average",
    weights: tf.Tensor = None,
    **kwargs,
) -> tf.Tensor:
    """
    Calculates F1 score in multiclass tasks.

    Extracts classes from y_true
    weights is only applicable in macro averaging.

    macro F1 score - averaged F1 score
    weighted macro F1 score - weighted averaged F1 score
    micro F1 score - F1 score of averages
    """

    assert (
        y_true.shape[0] == y_pred.shape[0]
    ), "Provide y_true and y_pred tensors of equal length"

    assert weights is None or classes.get_shape()[0] == weights.shape[0], (
        f"Provide weights tensor with the length equals to classes length:"
        f" classes length {len(classes)}, weights length: {weights.shape[0]}"
    )

    # raise Exception("Here")
    num_examples = tf.shape(y_pred)[0]
    num_classes = tf.shape(classes)[0]

    print(num_examples, num_classes)

    # masking y_true
    y_true_temp = tf.cast(tf.expand_dims(y_true, axis=1), dtype=tf.int32)
    ones = tf.ones((num_examples, num_classes), dtype=tf.int32)

    y_true_2d = tf.multiply(
        ones,
        y_true_temp,
    )

    rows, cols = tf.meshgrid(
        tf.range(y_true_2d.shape[0], dtype=tf.int32),
        tf.range(y_true_2d.shape[1], dtype=tf.int32),
    )

    mask_true = tf.equal(y_true_2d, tf.transpose(cols))
    mask_true = tf.cast(mask_true, dtype=tf.int32)

    # masking y_pred
    y_pred = tf.cast(tf.expand_dims(y_pred, axis=1), dtype=tf.int32)
    y_pred_2d = tf.multiply(
        tf.ones((num_examples, num_classes), dtype=tf.int32),
        y_pred,
    )

    rows, cols = tf.meshgrid(
        tf.range(y_pred_2d.shape[0], dtype=tf.int32),
        tf.range(y_pred_2d.shape[1], dtype=tf.int32),
    )

    mask_pred = tf.equal(y_pred_2d, tf.transpose(cols))
    mask_pred = tf.cast(mask_pred, dtype=tf.int32)

    f1_scores, precisions, recalls = f1_score_binary(
        y_true=mask_true, y_pred=mask_pred, out_precision_recall=True
    )

    # calculate metrics using generalized precision and recall
    if averaging == "micro":
        if group_func_name == "median":
            global_precision = _get_median(precisions)
            global_recall = _get_median(recalls)
        else:
            global_precision = tf.reduce_mean(precisions)
            global_recall = tf.reduce_mean(recalls)

        return _f1_score_formula(
            precision=global_precision, recall=global_recall
        )

    # calculate metrics for each label
    elif averaging == "macro":

        if weights is not None:
            f1_scores = tf.multiply(f1_scores, weights)

        if group_func == "median":
            return _get_median(f1_scores)
        else:
            return tf.reduce_mean(f1_scores)
    else:
        raise Exception("Provide micro or macro averaging key")


if __name__ == "__main__":

    # Binary case
    y_true_bin = tf.constant([0, 0, 1, 1, 0, 1], dtype=tf.int8)
    y_pred_bin = tf.constant([1, 0, 1, 0, 1, 1], dtype=tf.int8)
    f1_score = f1_score_binary(y_true_bin, y_pred_bin)
    print(f"Binary TF: {f1_score}")

    # Multiclass case
    classes = tf.constant([0, 1, 2, 3, 4])
    y_true = tf.constant(
        [0, 1, 2, 3, 4, 4, 3, 2, 1, 4, 1, 1, 2, 3, 4], dtype=tf.int32
    )
    y_pred = tf.constant(
        [0, 2, 2, 3, 3, 4, 3, 1, 1, 1, 4, 2, 2, 2, 2], dtype=tf.int32
    )  # 1: 3FP, 3FN, 2: 5FP, 1FN

    averaging = "macro"  # micro, macro
    group_func_name = "mean"  # median or smth else for mean

    from sklearn.metrics import f1_score

    print(f"Binary sklearn: {f1_score(y_true=y_true_bin, y_pred=y_pred_bin)}")

    for averaging in ["macro", "micro"]:
        global_f1_score = f1_score_multiclass(
            classes=classes,
            y_true=y_true,
            y_pred=y_pred,
            averaging=averaging,
            group_func=group_func_name,
            weights=None,
        )
        print(f"Multiclass TF {averaging}: {global_f1_score}")

        print(
            f"Multiclass sklearn {averaging}: "
            f"{f1_score(y_true=y_true, y_pred=y_pred, average=averaging)}"
        )

    weights = tf.constant(
        [0.40707, 0.20214, 0.18545, 0.03363, 0.17171], dtype=tf.float32
    )
    global_f1_score_weighted = f1_score_multiclass(
        classes=classes,
        y_true=y_true,
        y_pred=y_pred,
        averaging=averaging,
        group_func=group_func_name,
        weights=weights,
    )
    print(f"Multiclass weighted TF: {global_f1_score_weighted}")

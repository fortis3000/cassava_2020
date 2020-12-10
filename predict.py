import os
import tensorflow as tf

from config import DATAPATH, HEIGHT, WIDTH, MODELS_FOLDER


def parse_image(filename):
    """
    Parses each record of dataset into (image, image_id) pair
    """

    image_id = tf.strings.split(filename, os.sep)[-1]

    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [HEIGHT, WIDTH])
    image = tf.expand_dims(image, 0)
    return image, image_id


def predict_image(model, image):
    pred = tf.argmax(model(image), axis=-1)
    return pred


def predict_batch(model, dataset: tf.data.Dataset):
    pred_probs = []
    labels_pred = []
    labels_true = []

    for image, label in dataset.take(-1):
        pred = model(image)
        pred_prob = tf.reduce_max(pred, axis=-1)
        pred_label = tf.argmax(pred, axis=-1)
        truth_label = tf.argmax(label, axis=-1)

        pred_probs.append(pred_prob)
        labels_pred.append(pred_label)
        labels_true.append(truth_label)

    probas = tf.concat(labels_true, axis=-1).numpy()
    labels_true = tf.concat(labels_true, axis=-1).numpy()
    labels_pred = tf.concat(labels_pred, axis=-1).numpy()

    return labels_true, labels_pred, probas


def make_submission(
    model, image_ds: tf.data.Dataset, filename: str = "submission.csv"
):

    f = open(filename, "w")
    f.write("image_id,label\n")

    for image, image_id in image_ds.take(-1):
        pred = tf.argmax(model(image), axis=-1)
        f.write(f"{image_id.numpy().decode('utf-8')},{pred[0]}\n")

    f.close()


if __name__ == "__main__":

    last_model = sorted(os.listdir(MODELS_FOLDER))[-1]

    TEST_MODEL_FOLDER = os.path.join(MODELS_FOLDER, last_model)

    model = tf.saved_model.load(export_dir=TEST_MODEL_FOLDER)

    #####
    # SUBMITTING
    #####
    # tf.Data API
    test_ds = tf.data.Dataset.list_files(
        os.path.join(DATAPATH, r"test_images/*")
    )
    image_ds = test_ds.map(parse_image)
    image_ds.prefetch(tf.data.experimental.AUTOTUNE)

    make_submission(model=model, image_ds=image_ds, filename="submission.csv")

    #####
    # BATCH PREDICTION
    #####
    import pickle
    from dataset import init_dataset, input_preprocess
    from config import TFRECORDS_VAL_PATH, TFRECORDS_TRAIN_PATH
    from sklearn.metrics import confusion_matrix, classification_report

    ds_val = init_dataset(
        os.path.join(TFRECORDS_VAL_PATH),
        is_target=True,
        shuffle=False,
        augment=False,
    )
    ds_val = ds_val.map(input_preprocess)
    ds_val = ds_val.batch(batch_size=1, drop_remainder=True)
    ds_val = ds_val.prefetch(tf.data.experimental.AUTOTUNE)

    labels_true, labels_pred, _ = predict_batch(model, ds_val)

    print(classification_report(y_true=labels_true, y_pred=labels_pred))
    print(confusion_matrix(y_true=labels_true, y_pred=labels_pred))

    with open(os.path.join(TEST_MODEL_FOLDER, "report.pickle"), "wb") as f:
        pickle.dump(
            classification_report(y_true=labels_true, y_pred=labels_pred), f
        )
    with open(os.path.join(TEST_MODEL_FOLDER, "matrix.pickle"), "wb") as f:
        pickle.dump(confusion_matrix(y_true=labels_true, y_pred=labels_pred), f)

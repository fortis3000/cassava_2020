import os
import tensorflow as tf

from config import DATAPATH, HEIGHT, WIDTH, TEST_MODEL_FOLDER


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


if __name__ == "__main__":

    model = tf.saved_model.load(export_dir=TEST_MODEL_FOLDER)

    # tf.Data API
    test_ds = tf.data.Dataset.list_files(
        os.path.join(DATAPATH, r"test_images/*")
    )

    image_ds = test_ds.map(parse_image)
    image_ds.prefetch(tf.data.experimental.AUTOTUNE)

    out = "submission.csv"
    f = open(out, "w")
    f.write("image_id,label\n")

    for image, image_id in image_ds.take(-1):
        pred = tf.argmax(model(image), axis=-1)
        f.write(f"{image_id.numpy().decode('utf-8')},{pred[0]}\n")

    f.close()

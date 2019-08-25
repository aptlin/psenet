import tensorflow as tf
from psenet import config
from psenet.model import build_model
import argparse


def export_saved_model(FLAGS):
    params = tf.contrib.training.HParams(
        kernel_num=7,
        backbone_name=FLAGS.backbone_name,
        encoder_weights="imagenet",
    )
    model = build_model(params)
    latest_checkpoint = tf.train.latest_checkpoint(FLAGS.source_dir)
    model.load_weights(latest_checkpoint)
    tf.keras.experimental.export_saved_model(model, FLAGS.target_dir)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument(
        "--source-dir",
        help="The directory with the model",
        default=config.MODEL_DIR,
        type=str,
    )
    PARSER.add_argument(
        "--target-dir",
        help="The directory where the saved model is going to be saved.",
        default=config.SAVED_MODEL_DIR,
        type=str,
    )
    PARSER.add_argument(
        "--backbone-name",
        help="""The name of the FPN backbone. Must be one of the following:
                - 'inceptionresnetv2'
                - 'inceptionv3'
                - 'resnext50'
                - 'resnext101'
                - 'mobilenet'
                - 'mobilenetv2'
                - 'efficientnetb0'
                - 'efficientnetb1'
                - 'efficientnetb2'
                - 'efficientnetb3'
                - 'efficientnetb4'
                - 'efficientnetb5'
        """,
        default=config.BACKBONE_NAME,
        type=str,
    )

    FLAGS, _ = PARSER.parse_known_args()
    export_saved_model(FLAGS)

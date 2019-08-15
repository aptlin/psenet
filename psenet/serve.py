import tensorflow as tf
from psenet import config
from psenet.model import model_fn
import argparse


def build_eval_estimator(FLAGS):
    params = tf.contrib.training.HParams(
        kernel_num=7, backbone_name=FLAGS.backbone_name, encoder_weights=None
    )

    estimator = tf.estimator.Estimator(
        model_fn=model_fn, model_dir=FLAGS.source_dir, params=params
    )
    return estimator


def export_saved_model(FLAGS):
    features = {
        config.IMAGE: tf.compat.v1.placeholder(
            dtype=tf.float32, shape=[None, None, None, 3], name=config.IMAGE
        )
    }
    estimator = build_eval_estimator(FLAGS)
    estimator.export_saved_model(
        FLAGS.target_dir,
        tf.estimator.export.build_raw_serving_input_receiver_fn(
            features, default_batch_size=None
        ),
    )


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
                - 'inceptionresnetv2',
                - 'inceptionv3',
                - 'resnext50',
                - 'resnext101',
                - 'mobilenet',
                - 'mobilenetv2',
                - 'efficientnetb0',
                - 'efficientnetb1',
                - 'efficientnetb2',
                - 'efficientnetb3',
                - 'efficientnetb4',
                - 'efficientnetb5'
        """,
        default=config.BACKBONE_NAME,
        type=str,
    )

    FLAGS, _ = PARSER.parse_known_args()
    export_saved_model(FLAGS)

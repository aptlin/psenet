import numpy as np
from psenet import config
import cv2
import tensorflow as tf


def random_flip(image, prob=0.5, dim=1):
    random_value = tf.random_uniform([])
    flipped = tf.reverse_v2(image, [dim])
    is_flipped = tf.less_equal(random_value, prob)
    output = tf.cond(is_flipped, flipped, image)
    return output


def random_rotate(image, prob=0.5, max_angle=10):
    random_value = tf.random_uniform([])
    is_rotated = tf.less_equal(random_value, prob)
    angle = tf.random_uniform([]) * 2 * max_angle - max_angle
    height, width = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D(
        (height / 2, width / 2), angle, 1
    )
    rotated = cv2.warpAffine(image, rotation_matrix, (height, width))
    output = tf.cond(is_rotated, rotated, image)
    return output


def scale(image, resize_length=config.RESIZE_LENGTH):
    height, width = image.shape[:2]
    ratio = 1.0
    if max(height, width) > resize_length:
        ratio = resize_length * 1.0 / max(height, width)

    def get_scaling_factor(side):
        rounded_side = np.round(side * ratio)
        return (
            rounded_side
            if rounded_side % 32 == 0
            else (np.floor(rounded_side / 32) + 1) * 32
        ) / side

    x_scale = get_scaling_factor(width)
    y_scale = get_scaling_factor(height)
    image = cv2.resize(image, dsize=None, fx=x_scale, fy=y_scale)
    return image


def preprocess_image(image):
    return image

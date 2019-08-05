import cv2
import numpy as np
import Polygon as plg
import pyclipper
import tensorflow as tf

from psenet import config


def random_flip(images, prob=0.5, dim=1):
    random_value = tf.random.uniform([])

    def flip():
        flipped = []
        for tensor in images:
            flipped.append(tf.reverse(tensor, [dim]))
        return flipped

    is_flipped = tf.less_equal(random_value, prob)
    outputs = tf.cond(is_flipped, flip, lambda: images)
    return outputs


def rotate(image):
    max_angle = config.MAX_ROTATION_ANGLE
    angle = tf.random.uniform([]) * 2 * max_angle - max_angle
    image = np.asarray(image, dtype="uint8")
    height, width = np.asarray(image.shape).astype("uint64")[:2]
    rotation_matrix = cv2.getRotationMatrix2D(
        (height / 2, width / 2), angle, 1
    )
    image = cv2.warpAffine(image, rotation_matrix, (height, width))
    return image


def random_rotate(images, prob=0.5):
    random_value = tf.random.uniform([])
    is_rotated = tf.less_equal(random_value, prob)
    rotated_images = []
    for image in images:
        image = tf.cond(
            is_rotated,
            lambda: tf.py_function(func=rotate, inp=[image], Tout=tf.uint8),
            lambda: image,
        )
        rotated_images.append(image)
    return rotated_images


def background_random_crop(
    images, text, crop_size=config.CROP_SIZE, prob=3 / 8
):
    random_value = tf.random.uniform([])
    text_shape = tf.shape(text)
    height = text_shape[0]
    height = tf.cast(height, tf.int64)
    width = text_shape[1]
    width = tf.cast(width, tf.int64)
    should_not_crop = tf.logical_and(
        tf.equal(width, crop_size), tf.equal(height, crop_size)
    )
    should_search = tf.logical_and(
        tf.greater(random_value, prob), tf.greater(tf.count_nonzero(text), 0)
    )

    non_zero_text_locs = tf.where(tf.math.greater(text, 0))
    left_texts = tf.math.reduce_min(non_zero_text_locs, axis=1) - crop_size

    left_texts = tf.where(
        tf.math.less(left_texts, 0), tf.zeros_like(left_texts), left_texts
    )
    right_background = tf.math.reduce_max(non_zero_text_locs, axis=1)
    right_background = tf.where(
        tf.math.less(right_background, 0),
        tf.zeros_like(right_background),
        right_background,
    )
    right_background_y = tf.math.minimum(
        right_background[0], tf.math.abs(height - crop_size)
    )
    right_background_x = tf.math.minimum(
        right_background[1], tf.math.abs(width - crop_size)
    )
    start_y = tf.cond(
        tf.not_equal(left_texts[0], right_background_y),
        lambda: tf.random.uniform(
            [], minval=left_texts[0], maxval=right_background_y, dtype=tf.int64
        ),
        lambda: left_texts[0],
    )
    start_x = tf.cond(
        tf.not_equal(left_texts[1], right_background_x),
        lambda: tf.random.uniform(
            [], minval=left_texts[1], maxval=right_background_x, dtype=tf.int64
        ),
        lambda: left_texts[1],
    )
    default_start_y = tf.cond(
        tf.greater(height, crop_size),
        lambda: tf.random.uniform(
            [], minval=0, maxval=height - crop_size, dtype=tf.int64
        ),
        lambda: tf.cast(0, tf.int64),
    )
    default_start_x = tf.cond(
        tf.greater(width, crop_size),
        lambda: tf.random.uniform(
            [], minval=0, maxval=width - crop_size, dtype=tf.int64
        ),
        lambda: tf.cast(0, tf.int64),
    )
    start_y = tf.cond(should_search, lambda: start_y, lambda: default_start_y)
    start_x = tf.cond(should_search, lambda: start_x, lambda: default_start_x)

    end_y = start_y + crop_size
    end_x = start_x + crop_size
    new_images = []
    for idx in range(len(images)):
        image = images[idx][start_y:end_y, start_x:end_x]
        new_images.append(image)

    output = tf.cond(should_not_crop, lambda: images, lambda: new_images)
    return output


def scale(image, resize_length=config.RESIZE_LENGTH):
    image_shape = tf.shape(image)
    resize_length = tf.cast(resize_length, tf.float32)
    height = tf.cast(image_shape[0], tf.float32)
    width = tf.cast(image_shape[1], tf.float32)
    max_side = tf.math.maximum(height, width)

    should_scale = tf.greater(max_side, resize_length)
    ratio = tf.cond(
        should_scale,
        lambda: tf.math.divide(resize_length, max_side),
        lambda: 1.0,
    )

    def get_scaling_factor(side):
        rounded_side = tf.round(side * ratio)
        is_div_by_32 = tf.equal(tf.math.mod(rounded_side, 32), 0.0)
        output = (
            tf.cond(
                is_div_by_32,
                lambda: rounded_side,
                lambda: (tf.floor(rounded_side / 32.0) + 1.0) * 32.0,
            )
            / side
        )
        return output

    x_scale = get_scaling_factor(width)
    y_scale = get_scaling_factor(height)
    output = tf.image.resize(
        image,
        [
            tf.cast(tf.round(y_scale * height), tf.int64),
            tf.cast(tf.round(x_scale * width), tf.int64),
        ],
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
    )
    return output


def random_scale(image, prob=0.5, resize_length=config.RESIZE_LENGTH):
    image = scale(image, resize_length)
    random_value = tf.random.uniform([])
    random_scale_factor = tf.random.uniform(
        [], minval=0.5, maxval=3.0, dtype=tf.float32
    )
    image_shape = tf.shape(image)
    height = tf.cast(image_shape[0], tf.float32)
    width = tf.cast(image_shape[1], tf.float32)
    min_side = tf.minimum(width, height)
    max_side = tf.maximum(width, height)
    should_limit_scale = tf.less_equal(
        min_side * random_scale_factor, config.MIN_SIDE
    )
    random_scale_factor = tf.cond(
        should_limit_scale,
        lambda: (max_side + 10.0) / min_side,
        lambda: random_scale_factor,
    )
    should_resize = tf.less_equal(random_value, prob)
    output = tf.cond(
        should_resize,
        lambda: tf.image.resize(
            image,
            [
                tf.cast(tf.round(random_scale_factor * height), tf.int64),
                tf.cast(tf.round(random_scale_factor * width), tf.int64),
            ],
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
        ),
        lambda: image,
    )
    return output


def dist(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


def perimeter(bbox):
    peri = 0.0
    for i in range(bbox.shape[0]):
        peri += dist(bbox[i], bbox[(i + 1) % bbox.shape[0]])
    return peri


def shrink(bboxes, rate, max_shr=20):
    rate = rate * rate
    shrinked_bboxes = []
    for bbox in bboxes:
        area = plg.Polygon(bbox).area()
        peri = perimeter(bbox)

        pco = pyclipper.PyclipperOffset()
        pco.AddPath(bbox, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        offset = min((int)(area * (1 - rate) / (peri + 0.001) + 0.5), max_shr)

        shrinked_bbox = pco.Execute(-offset)
        if len(shrinked_bbox) == 0:
            shrinked_bboxes.append(bbox)
            continue

        shrinked_bbox = np.array(shrinked_bbox)[0]
        if shrinked_bbox.shape[0] <= 2:
            shrinked_bboxes.append(bbox)
            continue

        shrinked_bboxes.append(shrinked_bbox)

    return np.array(shrinked_bboxes)


def normalize(image):
    image = tf.cast(image, tf.float32)
    image /= 255
    offset = tf.constant(config.MEAN_RGB, shape=[1, 1, 3])
    image -= offset

    scale = tf.constant(config.STDDEV_RGB, shape=[1, 1, 3])
    image /= scale
    return image

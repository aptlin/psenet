import cv2
import numpy as np
import Polygon as plg
import pyclipper
import tensorflow as tf

import psenet.config as config


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


def random_clip(location, side_length, crop_size):
    return tf.cond(
        tf.greater(side_length, crop_size),
        lambda: tf.math.minimum(
            tf.random.uniform(
                [], minval=0, maxval=side_length - crop_size, dtype=tf.int64
            ),
            location,
        ),
        lambda: tf.cast(0, tf.int64),
    )


def random_background_crop(
    images, reference_index=1, crop_size=config.CROP_SIZE, prob=3 / 8
):
    random_value = tf.random.uniform([])
    text = images[reference_index]
    text_shape = tf.shape(text)
    height = text_shape[0]
    height = tf.cast(height, tf.int64)
    width = text_shape[1]
    width = tf.cast(width, tf.int64)
    should_not_crop = tf.logical_and(
        tf.less_equal(width, crop_size), tf.less_equal(height, crop_size)
    )
    should_search = tf.logical_and(
        tf.greater(random_value, prob), tf.greater(tf.count_nonzero(text), 0)
    )

    def search_for_the_background():
        text_locations = tf.where(tf.math.greater(text, 0))

        left_borders = tf.math.reduce_min(text_locations, axis=1)
        right_borders = tf.math.reduce_max(text_locations, axis=1)

        start_y = tf.cond(
            tf.not_equal(left_borders[0], right_borders[0]),
            lambda: tf.random.uniform(
                [],
                minval=left_borders[0],
                maxval=right_borders[0],
                dtype=tf.int64,
            ),
            lambda: left_borders[0],
        )
        start_x = tf.cond(
            tf.not_equal(left_borders[1], right_borders[1]),
            lambda: tf.random.uniform(
                [],
                minval=left_borders[1],
                maxval=right_borders[1],
                dtype=tf.int64,
            ),
            lambda: left_borders[1],
        )

        start_y = random_clip(start_y, height, crop_size)
        start_x = random_clip(start_x, width, crop_size)

        return start_x, start_y

    def get_default_coords():
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
        return default_start_x, default_start_y

    def crop_background():
        start_x, start_y = tf.cond(
            should_search, search_for_the_background, get_default_coords
        )
        end_y = start_y + crop_size
        end_x = start_x + crop_size

        new_images = []
        for idx in range(len(images)):
            image = images[idx][start_y:end_y, start_x:end_x]
            new_images.append(image)
        return new_images

    output = tf.cond(should_not_crop, lambda: images, crop_background)
    return output


def adjust_side(side, divisor=config.MIN_SIDE, min_side=config.MIN_SIDE):
    if tf.is_tensor(side):
        new_side = tf.math.maximum(
            tf.math.floor(tf.math.floor(side + divisor / 2) / divisor)
            * divisor,
            min_side,
        )
        should_enlarge = tf.greater(0.9 * side, new_side)
        new_side = tf.cond(
            should_enlarge, lambda: new_side + divisor, lambda: new_side
        )
    else:
        new_side = max(min_side, int(side + divisor / 2) // divisor * divisor)
        if new_side < 0.9 * side:
            new_side += divisor
    return new_side


def scale(image, resize_length=config.RESIZE_LENGTH, min_side=config.MIN_SIDE):
    if tf.is_tensor(image):
        image_shape = tf.shape(image)
        resize_length = tf.cast(resize_length, tf.float32)

        height = tf.cast(image_shape[0], tf.float32)
        width = tf.cast(image_shape[1], tf.float32)
        max_side = tf.math.maximum(height, width)

        should_scale = tf.greater(max_side, resize_length)
        scaling_factor = tf.cond(
            should_scale,
            lambda: tf.math.divide(resize_length, max_side),
            lambda: 1.0,
        )

        output = tf.image.resize(
            image,
            [
                tf.cast(
                    adjust_side(tf.round(scaling_factor * height)), tf.int64
                ),
                tf.cast(
                    adjust_side(tf.round(scaling_factor * width)), tf.int64
                ),
            ],
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
        )
    else:
        height, width = image.shape[:2]
        max_side = max(height, width)
        ratio = 1.0
        if max_side > resize_length:
            ratio = resize_length / max_side
        return cv2.resize(
            image,
            (
                adjust_side(round(height * ratio)),
                adjust_side(round(width * ratio)),
            ),
            interpolation=cv2.INTER_NEAREST,
        )

    return output


def random_scale(
    image,
    prob=0.5,
    resize_length=config.RESIZE_LENGTH,
    crop_size=config.CROP_SIZE,
):
    image = scale(image, resize_length=resize_length)
    random_value = tf.random.uniform([])
    random_scaling_factor = tf.random.uniform(
        [], minval=0.5, maxval=3.0, dtype=tf.float32
    )
    image_shape = tf.shape(image)
    height = tf.cast(image_shape[0], tf.float32)
    width = tf.cast(image_shape[1], tf.float32)
    min_side = tf.math.minimum(width, height)
    new_min_side = min_side * random_scaling_factor
    should_adjust_scale = tf.less_equal(new_min_side, crop_size)
    random_scaling_factor = tf.cond(
        should_adjust_scale,
        lambda: (crop_size + 10.0) / min_side,
        lambda: random_scaling_factor,
    )
    should_resize = tf.less_equal(random_value, prob)
    output = tf.cond(
        should_resize,
        lambda: tf.image.resize(
            image,
            [
                tf.cast(
                    adjust_side(tf.round(random_scaling_factor * height)),
                    tf.int64,
                ),
                tf.cast(
                    adjust_side(tf.round(random_scaling_factor * width)),
                    tf.int64,
                ),
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


def check_image_validity(
    inputs, divisor=config.MIN_SIDE, min_side=config.MIN_SIDE
):
    def is_valid(side):
        return tf.logical_and(
            tf.math.greater_equal(side, min_side),
            tf.math.equal(tf.math.floormod(side, divisor), 0),
        )

    image = inputs[config.IMAGE]
    image_shape = tf.shape(image)
    height = image_shape[0]
    width = image_shape[1]
    return tf.logical_and(is_valid(height), is_valid(width))


def check_numpy_image_validity(
    inputs, divisor=config.MIN_SIDE, min_side=config.MIN_SIDE
):
    def is_valid(side):
        return (side >= min_side) and ((side % min_side) == 0)

    image = inputs[config.IMAGE]
    height, width = image.shape[:2]
    return is_valid(height) and is_valid(width)

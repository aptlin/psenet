import cv2
import numpy as np
from psenet import config
import tensorflow as tf
import Polygon as plg
import pyclipper


def random_flip(images, prob=0.5, dim=1):
    random_value = tf.random.uniform([])

    def flip():
        flipped = []
        for tensor in images:
            flipped.append(tf.reverse(tensor, [dim]))
        return flipped

    is_flipped = tf.less_equal(random_value, prob)
    outputs = tf.cond(is_flipped, flip, lambda: images)
    if not isinstance(outputs, (list, tuple)):
        outputs = [outputs]
    return outputs


def random_rotate(images, prob=0.5, max_angle=10):
    random_value = tf.random.uniform([])
    is_rotated = tf.less_equal(random_value, prob)
    angle = tf.random.uniform([]) * 2 * max_angle - max_angle

    def rotate():
        rotated = []
        for image in images:
            image = np.asarray(image, dtype="uint8")
            height, width = np.asarray(image.shape).astype("uint64")[:2]
            rotation_matrix = cv2.getRotationMatrix2D(
                (height / 2, width / 2), angle, 1
            )
            rotated.append(
                cv2.warpAffine(image, rotation_matrix, (height, width))
            )
        return rotated

    output = tf.cond(is_rotated, rotate, lambda: images)
    return output


def background_random_crop(
    images, text, crop_size=config.CROP_SIZE, prob=3 / 8
):
    random_value = tf.random.uniform([])
    height, width = np.asarray(text.shape).astype("uint64")[:2]
    if width == crop_size and height == crop_size:
        return images

    if random_value > prob and np.max(text) > 0:
        tl = np.min(np.where(text > 0), axis=1) - crop_size
        tl[tl < 0] = 0
        br = np.max(np.where(text > 0), axis=1) - crop_size
        br[br < 0] = 0
        br[0] = min(br[0], abs(height - crop_size))
        br[1] = min(br[1], abs(width - crop_size))

        if tl[0] != br[0]:
            start_y = tf.random.uniform(
                [], minval=tl[0], maxval=br[0], dtype=tf.int64
            )
        else:
            start_y = tl[0]
        if tl[1] != br[1]:
            start_x = tf.random.uniform(
                [], minval=tl[1], maxval=br[1], dtype=tf.int64
            )
        else:
            start_x = tl[1]
    else:
        if height > crop_size:
            start_y = tf.random.uniform(
                [], minval=0, maxval=height - crop_size, dtype=tf.int64
            )
        else:
            start_y = 0
        if width > crop_size:
            start_x = tf.random.uniform(
                [], minval=0, maxval=width - crop_size, dtype=tf.int64
            )
        else:
            start_x = 0
    for idx in range(len(images)):
        if len(images[idx].shape) == 3:
            images[idx] = images[idx][
                start_y : start_y + crop_size, start_x : start_x + crop_size, :
            ]
        else:
            images[idx] = images[idx][
                start_y : start_y + crop_size, start_x : start_x + crop_size
            ]
    return images


def scale(image, resize_length=config.RESIZE_LENGTH):
    image_shape = tf.shape(image)
    resize_length = tf.cast(resize_length, tf.float32)
    height = tf.cast(image_shape[0], tf.float32)
    width = tf.cast(image_shape[1], tf.float32)
    max_side = tf.maximum(height, width)

    is_greater = tf.greater(max_side, resize_length)
    ratio = tf.cond(
        is_greater,
        lambda: 1.0,
        lambda: tf.math.divide(resize_length, max_side),
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
            tf.cast(tf.round(y_scale * height), tf.uint64),
            tf.cast(tf.round(x_scale * width), tf.uint64),
        ],
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
    )
    return output


def random_scale(image, prob=0.5):
    random_value = tf.random.uniform([])
    random_scale_factor = tf.random.uniform([], minval=0.5, maxval=3)
    image_shape = tf.shape(image)
    height = tf.cast(image_shape[0], tf.float32)
    width = tf.cast(image_shape[1], tf.float32)
    min_side = tf.minimum(width, height)
    max_side = tf.maximum(width, height)
    should_limit_scale = tf.less_equal(
        min_side * random_scale_factor, max_side
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
                tf.cast(tf.round(random_scale_factor * height), tf.uint64),
                tf.cast(tf.round(random_scale_factor * width), tf.uint64),
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
    offset = tf.constant(config.MEAN_RGB, shape=[1, 1, 3])
    image -= offset

    scale = tf.constant(config.STDDEV_RGB, shape=[1, 1, 3])
    image /= scale
    return image

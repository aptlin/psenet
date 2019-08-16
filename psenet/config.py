import argparse

BACKBONE_NAME = "mobilenetv2"
BASE_DATA_DIR = "./dist/mlt/tfrecords"
BATCH_SIZE = 1
BBOX_SIZE = 8
BBOXES = "bboxes"
CROP_SIZE = 320
EPSILON = 1e-4
EVAL_DATA_DIR = BASE_DATA_DIR + "/eval"
EVAL_START_DELAY_SECS = 10
EVAL_THROTTLE_SECS = 120
GPU_PER_WORKER = 0
GRADIENT_CLIPPING_NORM = 9.0
HEIGHT = "height"
IMAGE = "image"
IMAGE_NAME = "image_name"
IMAGES_DIR = "images"
KEEP_CHECKPOINT_EVERY_N_HOURS = 0.5
KERNEL_METRICS = "kernel-metrics"
KERNEL_NUM = 7
KERNELS = "kernels"
KERNELS_LOSS_WEIGHT = 0.3
LABEL = "label"
LABELS_DIR = "labels"
LEARNING_RATE = 1e-3
LEARNING_RATE_DECAY_FACTOR = 0.1
LEARNING_RATE_DECAY_STEPS = 200
MASK = "mask"
MAX_ROTATION_ANGLE = 10
MIN_SCALE = 0.4
MIN_SIDE = 32
MIRRORED_STRATEGY = "mirrored"
MODEL_DIR = "./dist/psenet"
MOMENTUM = 0.99
MULTIWORKER_MIRRORED_STRATEGY = "multi-worker-mirrored"
N_EPOCHS = 600
N_EVAL_STEPS = 5
NUM_BATCHES_TO_SHUFFLE = 4
NUM_READERS = 1
NUMBER_OF_BBOXES = "number_of_bboxes"
PREFETCH = 1
RAW_EVAL_DATA_DIR = "./dist/mlt/eval"
RAW_TRAINING_DATA_DIR = "./dist/mlt/train"
REGULARIZATION_WEIGHT_DECAY = 5e-4
RESIZE_LENGTH = 1280
SAVE_CHECKPOINTS_STEPS = 5
SAVE_SUMMARY_STEPS = 1
SAVED_MODEL_DIR = "./dist/psenet/saved_model"
TAGS = "tags"
TEXT = "text"
TEXT_LOSS_WEIGHT = 0.7
TEXT_METRICS = "text-metrics"
TRAINING_DATA_DIR = BASE_DATA_DIR + "/train"
WARM_CHECKPOINT = "./dist/warm/segmentation_filters_128"
WIDTH = "width"


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

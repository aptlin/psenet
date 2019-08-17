from psenet import config
from .processed import build as build_processed_dataset
from .raw import build as build_raw_dataset

DATASETS = {
    config.RAW_DATA_LABEL: build_raw_dataset,
    config.PROCESSED_DATA_LABEL: build_processed_dataset,
}


def build_input_fn(FLAGS):
    dataset_label = FLAGS.dataset
    if dataset_label not in DATASETS:
        raise ValueError(
            "The dataset {} is not supported. Try one out of {}.".format(
                dataset_label, list(DATASETS.keys())
            )
        )
    input_fn = DATASETS[dataset_label](FLAGS)
    return input_fn

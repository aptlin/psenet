from math import ceil
from tqdm import tqdm
import numpy as np
from pathlib import Path
from multiprocessing.pool import ThreadPool

import requests

BASE_URL = "https://dataset-bj.cdn.bcebos.com/art"

ART_TRAIN = ["train_images.tar.gz", "train_labels.json"]
ART_TEST = ["test_part1_images.tar.gz", "test_part2_images.tar.gz"]


def fetch_url(dirname, filename, base=BASE_URL):
    if Path(dirname).exists():
        r = requests.get(f"{base}/{filename}", stream=True)
        total_size = int(r.headers.get("content-length", 0))
        wrote = 0
        block_size = 1024
        if r.status_code == 200:
            with open(Path(dirname) / filename, "wb") as f:
                for chunk in tqdm(
                    r.iter_content(block_size),
                    total=total_size,
                    unit="KB",
                    unit_scale=True,
                ):
                    wrote = wrote + len(chunk)
                    f.write(chunk)
            if total_size != 0 and wrote != total_size:
                raise RuntimeError("The download was incomplete!")
        else:
            return False
    return True


if __name__ == "__main__":
    current_dir = Path.cwd()
    train_dir = Path(f"{current_dir}/dist/train")
    test_dir = Path(f"{current_dir}/dist/test")
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    train_results = ThreadPool(4).imap_unordered(
        lambda filename: fetch_url(train_dir, filename), ART_TRAIN
    )
    test_results = ThreadPool(4).imap_unordered(
        lambda filename: fetch_url(test_dir, filename), ART_TEST
    )
    if not all(train_results):
        print("Train fails:", np.where(train_results)[0])
    if not all(test_results):
        print("Test fails:", np.where(test_results)[0])

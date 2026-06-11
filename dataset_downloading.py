import gzip
import os
import struct
import requests
import numpy as np
from PIL import Image

BASE_URL = "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/"

FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}

RAW_DIR = "fashion_raw"
DATA_DIR = "data/fashion_mnist"


def download_file(filename):
    os.makedirs(RAW_DIR, exist_ok=True)
    path = os.path.join(RAW_DIR, filename)

    if os.path.exists(path):
        print(f"Already downloaded: {filename}")
        return path

    url = BASE_URL + filename
    print(f"Downloading {filename}...")

    response = requests.get(url)
    response.raise_for_status()

    with open(path, "wb") as f:
        f.write(response.content)

    return path


def load_images(path):
    with gzip.open(path, "rb") as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(num_images, rows, cols)


def load_labels(path):
    with gzip.open(path, "rb") as f:
        magic, num_labels = struct.unpack(">II", f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8)


def save_split(images, labels, split_name):
    for label in range(10):
        os.makedirs(f"{DATA_DIR}/{split_name}/{label}", exist_ok=True)

    for idx, (img, label) in enumerate(zip(images, labels)):
        Image.fromarray(img).save(f"{DATA_DIR}/{split_name}/{label}/{idx}.png")

    print(f"Saved {split_name}: {len(images)} images")


if __name__ == "__main__":
    paths = {name: download_file(filename) for name, filename in FILES.items()}

    x_train = load_images(paths["train_images"])
    y_train = load_labels(paths["train_labels"])
    x_test = load_images(paths["test_images"])
    y_test = load_labels(paths["test_labels"])

    # last 10,000 training images become dev set
    x_dev = x_train[-10000:]
    y_dev = y_train[-10000:]

    x_train = x_train[:-10000]
    y_train = y_train[:-10000]

    save_split(x_train, y_train, "train")
    save_split(x_dev, y_dev, "dev")
    save_split(x_test, y_test, "test")

    print("Fashion-MNIST is ready.")
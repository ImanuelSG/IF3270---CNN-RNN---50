import os
import tarfile
import urllib.request
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models


def download_and_extract_cifar10(data_dir="./cifar10"):
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    filepath = os.path.join(data_dir, filename)

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if not os.path.exists(filepath):
        print("Downloading CIFAR-10...")
        urllib.request.urlretrieve(url, filepath)
        print("Download complete.")

    extract_path = os.path.join(data_dir, "cifar-10-batches-py")
    if not os.path.exists(extract_path):
        print("Extracting...")
        with tarfile.open(filepath, "r:gz") as tar:
            tar.extractall(path=data_dir)
        print("Extraction complete.")

    return extract_path

# ---------- Load One Batch ----------
def load_batch(batch_path):
    with open(batch_path, 'rb') as file:
        batch = pickle.load(file, encoding='bytes')
        data = batch[b'data']
        labels = batch[b'labels']
        data = data.reshape(-1, 3, 32, 32)
        labels = np.array(labels).reshape(-1, 1)
        return data, labels

# ---------- Load Full Dataset ----------
def load_cifar10_custom(data_dir="./cifar10"):
    path = download_and_extract_cifar10(data_dir)
    x_train = []
    y_train = []
    for i in range(1, 6):
        data, labels = load_batch(os.path.join(path, f"data_batch_{i}"))
        x_train.append(data)
        y_train.append(labels)
    x_train = np.concatenate(x_train)
    y_train = np.concatenate(y_train)
    x_test, y_test = load_batch(os.path.join(path, "test_batch"))
    return (x_train, y_train), (x_test, y_test)

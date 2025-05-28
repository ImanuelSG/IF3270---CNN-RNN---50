import os
import tarfile
import urllib.request
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# ---------- Download & Extract ----------
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

# ---------- Custom Callback for F1 ----------
class F1Callback(tf.keras.callbacks.Callback):
    def __init__(self, val_data):
        super().__init__()
        self.val_data = val_data

    def on_epoch_end(self, epoch, logs=None):
        val_x, val_y = self.val_data
        y_pred = self.model.predict(val_x, verbose=0)
        y_pred_labels = np.argmax(y_pred, axis=1)
        val_y = val_y.flatten()
        f1 = f1_score(val_y, y_pred_labels, average='macro')
        print(f"\nEpoch {epoch+1} - val_macro_f1: {f1:.4f}")

# ---------- Load and Preprocess ----------



# ---------- Model ----------
model = models.Sequential([
    layers.Conv2D(96, (5, 5), activation='relu', padding='same', input_shape=(32, 32, 3)),
    layers.MaxPooling2D(pool_size=(2, 2), strides=2),

    layers.Conv2D(80, (5, 5), activation='relu', padding='same'),
    layers.MaxPooling2D(pool_size=(2, 2), strides=2),

    layers.Conv2D(96, (3, 3), activation='relu', padding='same'),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),

    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dense(10)  # logits
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# ---------- Training ----------
history = model.fit(
    x_train, y_train,
    epochs=10,
    validation_data=(x_val, y_val),
    callbacks=[F1Callback((x_val, y_val))]
)

# ---------- Evaluation ----------
test_logits = model.predict(x_test)
y_pred = np.argmax(test_logits, axis=1)

macro_f1 = f1_score(y_test, y_pred, average='macro')
print(f"\nTest Macro F1 Score: {macro_f1:.4f}")

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Test Accuracy: {test_acc:.4f}")

from src.models.cnn import CNNModel
from src.layers.cnn.conv import Conv2D
from src.layers.cnn.dense import DenseLayer
from src.layers.cnn.flatten import Flatten
from src.layers.cnn.pooling import Pooling
from src.test.cnn.load_cnn_data import load_cifar10_custom
from src.utils.loss import CategoricalCrossEntropyLoss
import numpy as np
from sklearn.metrics import f1_score

(x_train, y_train), (x_test, y_test) = load_cifar10_custom()
model = CNNModel(
    layers=[
        Conv2D(32, kernel_size=3, activation="relu"),   # 1st Conv Layer
        Pooling(pool_size=2),                           # Pooling

        Flatten(),                                      # Flatten before Dense
        DenseLayer(64, activation="relu", init_method="glorot_uniform"),  # Hidden Dense
        DenseLayer(10, activation="softmax", init_method="glorot_uniform")  # Output Layer
    ],
    loss_fn=CategoricalCrossEntropyLoss()
)

# model.load_weights(weights_to_load)

y_pred_probs = model.predict(x_test[:10])
y_pred = np.argmax(y_pred_probs.data, axis=1)
macro_f1 = f1_score(y_test[:10], y_pred, average='macro')
print(f"\n📊 Macro F1-Score on Test Set: {macro_f1:.4f}")
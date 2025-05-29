from src.models.cnn_normal import CNNModel
from src.layers.cnn.usenumpy.conv import Conv2D
from src.layers.cnn.usenumpy.dense import DenseLayer
from src.layers.cnn.usenumpy.flatten import Flatten
from src.layers.cnn.usenumpy.pooling import Pooling
from src.test.cnn.load_cnn_data import load_cifar10_custom
from src.utils.loss import CategoricalCrossEntropyLoss
import numpy as np
from sklearn.metrics import f1_score

(x_train, y_train), (x_test, y_test) = load_cifar10_custom()


model = CNNModel(
    layers=[
        Conv2D(64, kernel_size=3, activation="relu", padding="same"),   
        Conv2D(64, kernel_size=3, activation="relu", padding="same"),   
        Pooling(pool_size=2),       

        Conv2D(128, kernel_size=3, activation="relu", padding="same"),   
        Conv2D(128, kernel_size=3, activation="relu", padding="same"),   
        Pooling(pool_size=2),         

        Conv2D(256, kernel_size=3, activation="relu", padding="same"),   
        Conv2D(256, kernel_size=3, activation="relu", padding="same"),   
        Pooling(pool_size=2),                                                               

        Flatten(),                                      
        DenseLayer(512, activation="relu", init_method="glorot_uniform"),  
        DenseLayer(10, activation="softmax", init_method="glorot_uniform")
    ],
    loss_fn=CategoricalCrossEntropyLoss()
)

keras_weights = "./src/test/cnn/keras_cnn_weights.npz"

weights_np = np.load(keras_weights, allow_pickle=True)
keys = weights_np.files


weights_list = [weights_np[k] for k in keys]
model.load_weights(weights_list)

y_pred_probs = model.predict(x_test[:10])
y_pred = np.argmax(y_pred_probs.data, axis=1)
macro_f1 = f1_score(y_test, y_pred, average='macro')
print(f"\n📊 Macro F1-Score on Test Set: {macro_f1:.4f}")
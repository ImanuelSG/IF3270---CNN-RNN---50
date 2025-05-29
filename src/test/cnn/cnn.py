from src.models.cnn_normal import CNNModel 
from src.layers.cnn.usenumpy.conv import Conv2D
from src.layers.cnn.usenumpy.dense import DenseLayer
from src.layers.cnn.usenumpy.pooling import Pooling
from src.layers.cnn.usenumpy.flatten import Flatten
from src.utils.autodiff import Value
from src.utils.manualloss import CategoricalCrossEntropyLoss
import numpy as np
import os
import sys

model = CNNModel(loss_fn=CategoricalCrossEntropyLoss(), epochs=10)

# Create layers
layer1 = Conv2D(filters=2, kernel_size=2, padding=0, activation='relu')
layer2 = Pooling(pool_size=3, pool_type='max', padding=0, strides=1)
layer3 = Flatten()
layer4 = DenseLayer(output_shape=2, activation='relu')
layer5 = DenseLayer(output_shape=5, activation='softmax')

# Assign predefined weights (as regular Python lists or NumPy arrays)
layer1.weights = np.array([[
    [[-8, 1], [4, 7]]
], [
    [[2, -4], [-3, 5]]
]], dtype=np.float64)
layer1.bias = np.array([0, 0], dtype=np.float64)

layer4.weights = np.array([[-1, 7], [3, -8]], dtype=np.float64)
layer4.bias = np.array([0, 0], dtype=np.float64)

layer5.weights = np.array([
    [0.01, 0.05],
    [0.02, 0.04],
    [0.03, 0.03],
    [0.04, 0.02],
    [0.05, 0.01]
], dtype=np.float64)
layer5.bias =np.array( [0, 0, 0, 0, 0], dtype=np.float64)

# Add layers to the model
model.add(layer1)
model.add(layer2)
model.add(layer3)
model.add(layer4)
model.add(layer5)

# Prepare input and label (plain Python nested lists or NumPy arrays)
input_val =np.array( [[[
    [2, 1, 3, 5],
    [4, 5, 7, 2],
    [3, 2, 8, 2],
    [1, 6, 3, 7]
]]], dtype=np.float64)
target = np.array([0], dtype=np.float64)  # class label

model.fit(X=input_val, Y=target)



from tensorflow.keras import layers, models, optimizers, losses
import numpy as np


input_val = np.array([
    [
        [[2, 1, 3, 5],
         [4, 5, 7, 2],
         [3, 2, 8, 2],
         [1, 6, 3, 7]]
    ]
], dtype=np.float32).transpose(0, 2, 3, 1)  # reshape to (1, 4, 4, 1) for Keras

y_train = np.array([0])  # class label

model = models.Sequential([
    layers.Conv2D(
        filters=2, kernel_size=(2, 2),
        activation='relu',
        padding='valid',
        input_shape=(4, 4, 1),
        kernel_regularizer=None,
        bias_regularizer=None,    
        activity_regularizer=None 
    ),
    layers.MaxPooling2D(pool_size=(3, 3)),
    layers.Flatten(),
    layers.Dense(2, activation='relu', kernel_regularizer=None, bias_regularizer=None),
    layers.Dense(5, activation='softmax', kernel_regularizer=None, bias_regularizer=None),
])

# Manually set weights
# Conv2D weights: shape = (kernel_h, kernel_w, in_channels, out_channels)
weight1 = np.array([
    [[[-8, 2]], [[1, -4]]],
    [[[4, -3]], [[7, 5]]]
], dtype=np.float32)

bias1 = np.array([0.0, 0.0], dtype=np.float32)

weight_4 = np.array([[-1, 3], [7, -8]], dtype=np.float32)
bias_4 = np.array([0.0, 0.0], dtype=np.float32)

weight_5 = np.array([
    [0.01, 0.02, 0.03, 0.04, 0.05],
    [0.05, 0.04, 0.03, 0.02, 0.01]
], dtype=np.float32)
bias_5 = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

# Set weights manually
model.layers[0].set_weights([weight1, bias1])
model.layers[3].set_weights([weight_4, bias_4])
model.layers[4].set_weights([weight_5, bias_5])

# Compile model
model.compile(
    optimizer=optimizers.Adam(learning_rate=0.01),
    loss=losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

# # Train
model.fit(input_val, y_train, epochs=10, verbose=1)

# print(model.predict(input_val))
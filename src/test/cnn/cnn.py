from src.models.cnn import CNNModel
from src.layers.cnn.conv import Conv2D
from src.layers.cnn.dense import DenseLayer
from src.layers.cnn.pooling import Pooling
from src.layers.cnn.flatten import Flatten
from src.utils.autodiff import Value
from src.utils.loss import CategoricalCrossEntropyLoss
import os
import sys


model = CNNModel(loss_fn=CategoricalCrossEntropyLoss(), epochs=10)
layer1 = Conv2D(filters=2, kernel_size=2, padding=0, activation='relu')
layer2 = Pooling(pool_size=3, pool_type='max', padding=0, strides=1)
layer3 = Flatten()
layer4 = DenseLayer(activation='relu', output_shape=2)
layer5 = DenseLayer(activation='softmax', output_shape=5)
weight1 = [
    [
    [[-8,1], [4,7]]
    
    ],
    [[[2,-4], [-3,5]]]
    ]
biases = [0,0]
weight_4 = [[-1,7],[3,-8]]
bias_4 = [0,0]
weight_5 = [[0.01,0.05],[0.02,0.04],[0.03,0.03],[0.04,0.02],[0.05,0.01]]

bias_5 = [0,0,0,0,0]
input_val = [
    [
        [[2,1,3,5],
         [4,5,7,2],
         [3,2,8,2],
         [1,6,3,7]
         ]
    ]
]
layer1.weights = Value(weight1, requires_grad=True)
layer1.bias = Value(biases, requires_grad=True)
layer1.initialized = True
layer4.weights = Value(weight_4, requires_grad=True)
layer4.bias = Value(bias_4, requires_grad=True)
layer5.weights = Value(weight_5, requires_grad=True)
layer5.bias = Value(bias_5, requires_grad=True)

model.add(layer1)
model.add(layer2)
model.add(layer3)
model.add(layer4)
model.add(layer5)
# model.forward(Value(input_val, requires_grad=False))
# print(layer1.forward(Value(input_val, requires_grad=False)))
model.fit(
    X=Value(input_val, requires_grad=False),
    Y=Value([0], requires_grad=False)
)


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
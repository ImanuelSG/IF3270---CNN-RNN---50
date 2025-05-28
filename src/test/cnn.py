from src.models.cnn import CNNModel
from src.layers.cnn.conv import Conv2D
from src.layers.cnn.dense import DenseLayer
from src.layers.cnn.pooling import Pooling
from src.layers.cnn.flatten import Flatten
from src.utils.autodiff import Value
from src.utils.loss import CategoricalCrossEntropyLoss
import os
import sys


model = CNNModel(loss_fn=CategoricalCrossEntropyLoss())
layer1 = Conv2D(filters=2, kernel_size=2, padding=0, activation='relu')
layer2 = Pooling(pool_size=3, pool_type='global', padding=0, strides=1)
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






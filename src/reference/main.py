from engine.autodiff import Value
from nn.layer import Layer
from nn.ffnn import FFNN
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state
from sklearn.preprocessing import StandardScaler
import numpy as np
from nn.initializers import XavierInit


data = np.load("mnist_data.npz")
X_train = data["X_train"]
X_test = data["X_test"]
y_train = data["y_train"]
y_test = data["y_test"]

X_train_val = Value(X_train)
y_train_val = Value(y_train)
X_test_val = Value(X_test)
y_test_val = Value(y_test)

xavier = XavierInit(42, uniform=True)
h1 = Layer(128, activation="relu", init_method=xavier, input_shape=784)
h2 = Layer(64, activation="relu", init_method=xavier)
out = Layer(10, activation="softmax", init_method=xavier)

# ourmodel = FFNN(learning_rate=0.5, epochs=5, layers_list=[h1, h2, out], random_seed=42, verbose=1, batch_size=200)

# ourmodel.compile("CCE", "L2")
# ourmodel.fit(X_train_val, y_train_val, X_test_val, y_test_val)

# print(ourmodel.score(X_test_val, y_test_val))
# ourmodel.save("model.pkl")


# ourmodel_new = FFNN.load("models/model.pkl")
# print(ourmodel_new.score(X_test_val, y_test_val))

# ourmodel_new.compile("CCE", "L2")
# ourmodel_new.fit(X_train_val, y_train_val, X_test_val, y_test_val)
# print(ourmodel_new.score(X_test_val, y_test_val))
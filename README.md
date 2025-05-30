# IF3270---CNN-RNN---50
# CNN, RNN, LSTM from Scratch

## General Description

This project implements **Convolutional Neural Network (CNN), Recurrent Neural Network (RNN), Long Short-Term Memory (LSTM)** from scratch. It features:

- CNN Implementation from scratch, with Forward and Backward Propagation ✅
- RNN Implementation from scratch, with Forward Propagation ✅
- LSTM Implementation from scratch, with Forward Propagation ✅
- The implementation of the forward and backward propagations are made to have the same results to Keras' implemetation ✅
- Batch Size Implementation for CNN, RNN, and LSTM ✅

## Project Structure
```
.
├── README.md
├── doc
│   └── Tubes2_IF3270_CNN_RNN_LSTM.pdf
├── main.py
├── requirements.txt
└── src
    ├── __init__.py
    ├── data
    │   ├── test.csv
    │   ├── train.csv
    │   └── valid.csv
    ├── layers
    │   ├── __init__.py
    │   ├── cnn
    │   │   ├── __init__.py
    │   │   ├── conv.py
    │   │   ├── dense.py
    │   │   ├── flatten.py
    │   │   ├── pooling.py
    │   │   └── usenumpy
    │   │       ├── conv.py
    │   │       ├── dense.py
    │   │       ├── flatten.py
    │   │       └── pooling.py
    │   ├── dense.py
    │   ├── dropout.py
    │   ├── embedding.py
    │   ├── layer.py
    │   ├── lstm
    │   │   ├── bidirectionalLSTM.py
    │   │   └── unidirectionalLSTM.py
    │   └── rnn
    │       ├── bidirectionalRNN.py
    │       └── unidirectionalRNN.py
    ├── main.py
    ├── model_scratch
    │   ├── __init__.py
    │   ├── cnn.py
    │   ├── lstm.py
    │   ├── model.py
    │   └── rnn.py
    ├── models
    │   ├── __init__.py
    │   ├── cnn.py
    │   ├── cnn_normal.py
    │   └── model.py
    ├── reference
    │   ├── engine
    │   │   ├── __init__.py
    │   │   └── autodiff.py
    │   ├── main.ipynb
    │   ├── main.py
    │   └── nn
    │       ├── __init__.py
    │       ├── ffnn.py
    │       ├── initializers.py
    │       ├── layer.py
    │       ├── loss.py
    │       └── module.py
    ├── test
    │   ├── __init__.py
    │   ├── cnn
    │   │   ├── cnn.py
    │   │   ├── cnn_compare.py
    │   │   ├── cnn_normal_compare.py
    │   │   ├── debug.py
    │   │   ├── keras.py
    │   │   ├── keras_cnn_weights.npz
    │   │   └── load_cnn_data.py
    │   └── cnn_backprop.py
    ├── trainLSTM.ipynb
    ├── trainRNN.ipynb
    ├── utils
    │   ├── __init__.py
    │   ├── activations.py
    │   ├── autodiff.py
    │   ├── evaluate.py
    │   ├── initialize_weights.py
    │   ├── loss.py
    │   ├── manualloss.py
    │   └── visualize.py
    └── weights
        ├── LSTM
        │   ├── cell16.weights.h5
        │   ├── cell24.weights.h5
        │   ├── cell8.weights.h5
        │   ├── layer1.weights.h5
        │   ├── layer1Bi.weights.h5
        │   ├── layer1Uni.weights.h5
        │   ├── layer2.weights.h5
        │   └── layer3.weights.h5
        └── RNN
            ├── cell16.weights.h5
            ├── cell24.weights.h5
            ├── cell8.weights.h5
            ├── layer1.weights.h5
            ├── layer1Bi.weights.h5
            ├── layer1Uni.weights.h5
            ├── layer2.weights.h5
            └── layer3.weights.h5
```

## Installation
### Step 1: Create and Activate Virtual Environment
```sh
python -m venv .venv  # Create virtual environment
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate  # Windows
```
Make sure to choose the appropriate Python environment for the Jupyter kernel.

### Step 2: Install Dependencies
```sh
pip install -r requirements.txt
```

### Step 3: Run the Program
Run the `main.py` or open `trainRNN.ipynb` and `trainLSTM.ipynb`. Uncomment and adjust the appropriate lines based on your testing needs.

### Step 4: Freeze Dependencies
```sh
pip freeze > requirements.txt
```

## Work Division

| Items                        | Name (NIM)                                  |
|------------------------------|---------------------------------------------|
| CNN                          | Imanuel Sebastian Girsang (13522058)        |
| RNN                          | Fedrianz Dharma (13522090)                  |
| LSTM                         | Aurelius Justin Philo Fanjaya (13522020)    |
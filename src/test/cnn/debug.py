import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score

# === Your Custom Modules ===
from src.models.cnn_normal import CNNModel
from src.layers.cnn.usenumpy.conv import Conv2D
from src.layers.cnn.usenumpy.dense import DenseLayer
from src.layers.cnn.usenumpy.flatten import Flatten
from src.layers.cnn.usenumpy.pooling import Pooling
from src.test.cnn.load_cnn_data import load_cifar10_custom
from src.utils.loss import CategoricalCrossEntropyLoss

# === Step 1: Load the data ===
(x_train, y_train), (x_test, y_test) = load_cifar10_custom()
x_input = x_test[:10]  # Use one sample for comparison

x_input_keras = x_input.transpose(0, 2, 3, 1)  # Convert from NCHW to NHWC for Keras

# === Step 2: Load weights ===
keras_weights_path = "./src/test/cnn/keras_cnn_weights.npz"
weights_np = np.load(keras_weights_path, allow_pickle=True)
weights_list = [weights_np[k] for k in weights_np.files]

# === Step 3: Custom model with intermediate output ===
class CNNModelWithIntermediates(CNNModel):
    def predict_with_intermediates(self, x):
        intermediates = []
        for layer in self.layers:
            x = layer(x)
            intermediates.append(x)
        return intermediates

model = CNNModelWithIntermediates(
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

model.load_weights(weights_list)

# === Step 4: Build Keras model ===
def build_keras_model():
    input_layer = tf.keras.Input(shape=(32, 32, 3))
    x = tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same', name='conv1')(input_layer)
    x = tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same', name='conv2')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2,2), name='pool1')(x)

    x = tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same', name='conv3')(x)
    x = tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same', name='conv4')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2,2), name='pool2')(x)

    x = tf.keras.layers.Conv2D(256, (3,3), activation='relu', padding='same', name='conv5')(x)
    x = tf.keras.layers.Conv2D(256, (3,3), activation='relu', padding='same', name='conv6')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2,2), name='pool3')(x)

    x = tf.keras.layers.Flatten(name='flatten')(x)
    x = tf.keras.layers.Dense(512, activation='relu', name='dense1')(x)
    x = tf.keras.layers.Dense(10, activation='softmax', name='output')(x)

    model = tf.keras.Model(inputs=input_layer, outputs=x)
    return model


def load_weights_to_keras_model(keras_model, weights_list):

    
    # Get only the layers that have weights (Conv2D and Dense layers)
    trainable_layers = [layer for layer in keras_model.layers if len(layer.get_weights()) > 0]
    
    print(f"Found {len(trainable_layers)} trainable layers in Keras model")
    print(f"Found {len(weights_list)} weight arrays in weights_list")
    
    # Map weights to layers
    weight_idx = 0
    for i, layer in enumerate(trainable_layers):
        layer_weights = layer.get_weights()
        num_params = len(layer_weights)  # Usually 2 (weights + bias) for Conv2D/Dense
        
        if weight_idx >= len(weights_list):
            print(f"Warning: Not enough weights for layer {layer.name}")
            break
            
        new_weights = []
        
        if isinstance(layer, tf.keras.layers.Conv2D):
            # Conv2D layer: expects [height, width, input_channels, output_channels]
            # Your custom format might be [output_channels, input_channels, height, width]
            kernel_weights = weights_list[weight_idx]
            
            
            new_weights.append(kernel_weights)
            weight_idx += 1
            
            # Add bias if present
            if num_params > 1 and weight_idx < len(weights_list):
                bias_weights = weights_list[weight_idx]
                new_weights.append(bias_weights)
                weight_idx += 1
                
        elif isinstance(layer, tf.keras.layers.Dense):
            # Dense layer: expects [input_features, output_features]
            dense_weights = weights_list[weight_idx]
            
            # Check if we need to transpose
            if len(dense_weights.shape) == 2:
                # Your custom format might be [output_features, input_features]
                # Keras expects [input_features, output_features]
                if dense_weights.shape[0] == layer.units:
                    dense_weights = dense_weights.T
            
            new_weights.append(dense_weights)
            weight_idx += 1
            
            # Add bias if present
            if num_params > 1 and weight_idx < len(weights_list):
                bias_weights = weights_list[weight_idx]
                new_weights.append(bias_weights)
                weight_idx += 1
        
        # Set the weights for this layer
        try:
            layer.set_weights(new_weights)
            print(f"✅ Loaded weights for {layer.name}: {[w.shape for w in new_weights]}")
        except Exception as e:
            print(f"❌ Failed to load weights for {layer.name}: {e}")
            print(f"   Expected shapes: {[w.shape for w in layer_weights]}")
            print(f"   Got shapes: {[w.shape for w in new_weights]}")

# Load weights into Keras model

keras_model = build_keras_model()

load_weights_to_keras_model(keras_model, weights_list)

# === Step 5: Create Keras intermediate model ===
intermediate_model = tf.keras.Model(
    inputs=keras_model.input,
    outputs=[layer.output for layer in keras_model.layers if 'input' not in layer.name]
)


evaluation_size = 1000
x_eval_custom = x_test[:evaluation_size]  
x_eval_keras = x_eval_custom.transpose(0, 2, 3, 1)  



y_pred_custom = model.predict(x_eval_custom)
y_pred_keras = keras_model.predict(x_eval_keras)

y_pred_custom = np.argmax(y_pred_custom.data, axis=1)
y_pred_keras = np.argmax(y_pred_keras, axis=1)

f1_custom = f1_score(y_test[:evaluation_size], y_pred_custom, average='macro')
f1_keras = f1_score(y_test[:evaluation_size], y_pred_keras, average='macro')

print(f"🔧 Custom Model F1 Score (macro) [For 1k data]: {f1_custom:.4f}")
print(f"🤖 Keras  Model F1 Score (macro) [For 1k data]: {f1_keras:.4f}")
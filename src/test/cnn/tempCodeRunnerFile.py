import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

(x_train_full, y_train_full), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

x_train_full = x_train_full.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0


x_train, x_val, y_train, y_val = train_test_split(
    x_train_full, y_train_full, test_size=0.2, random_state=42
)

model = models.Sequential([
    layers.Conv2D(96, (5, 5), activation='relu', padding='same', input_shape=(32, 32, 3)),
    layers.MaxPooling2D(pool_size=(2, 2), strides=2),

    layers.Conv2D(96, (5, 5), activation='relu', padding='same'),
    layers.MaxPooling2D(pool_size=(2, 2), strides=2),

    layers.Conv2D(80, (5, 5), activation='relu', padding='same'),
    layers.Conv2D(64, (5, 5), activation='relu', padding='same'),
    layers.Conv2D(64, (5, 5), activation='relu', padding='same'),
    layers.Conv2D(96, (5, 5), activation='relu', padding='same'),

    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dense(10)  # No activation for logits (from_logits=True used below)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train model
history = model.fit(
    x_train, y_train,
    epochs=10,
    validation_data=(x_val, y_val)
)

# Evaluate on test set
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\nTest accuracy: {test_acc:.4f}")

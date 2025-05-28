from tensorflow.keras import layers, models

def final_cnn_model(input_shape=(32, 32, 3), num_classes=10):
    model = models.Sequential()

    # Block 1: Conv + Conv + MaxPool
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # Block 2: Conv + Conv + MaxPool
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # Block 3: Conv + Conv + MaxPool
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # Flatten + Dense layers
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))  # final softmax for classification

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
    
model = final_cnn_model()
history = model.fit(
    x_train, y_train,
    epochs=15,
    validation_data=(x_val, y_val),
    batch_size=64,
    verbose=2
)

# Plot Loss
plot_history(history)

# 5. Evaluate on Test Set
y_pred_probs = model.predict(x_test)
y_pred = np.argmax(y_pred_probs, axis=1)
macro_f1 = f1_score(y_test, y_pred, average='macro')
print(f"\n📊 Macro F1-Score on Test Set: {macro_f1:.4f}")

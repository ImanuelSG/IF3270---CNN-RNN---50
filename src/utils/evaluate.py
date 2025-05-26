import numpy as np
from sklearn.metrics import f1_score
import torch

from utils.autodiff import Value
def evaluate_model(model, test, test_labels):
    # Evaluate the model on the test set
    # 1. Prediksi model
    y_pred_probs = model.predict(test)

    # 2. Ubah probabilitas ke label prediksi
    # Untuk output softmax (multi-class)
    # Convert to numpy array
    if isinstance(y_pred_probs, Value):
        y_pred_probs = y_pred_probs.data.numpy()
    elif isinstance(y_pred_probs, torch.Tensor):
        y_pred_probs = y_pred_probs.detach().numpy()

    # Get predicted labels
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Untuk output sigmoid (binary classification), gunakan ini:
    # y_pred = (y_pred_probs > 0.5).astype(int).flatten()

    # 3. Ambil label asli
    y_true = test_labels

    # 4. Hitung F1 score (macro average)
    f1 = f1_score(y_true, y_pred, average='macro')

    print(f"Macro F1 Score: {f1:.4f}")
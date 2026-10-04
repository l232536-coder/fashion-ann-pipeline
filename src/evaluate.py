import json
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.metrics import ConfusionMatrixDisplay

x_test = np.load("data/processed/x_test.npy")
y_test = np.load("data/processed/y_test.npy")

model = keras.models.load_model("models/model.h5")
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)

y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.savefig("confusion_matrix.png")

with open("metrics.json", "w") as f:
    json.dump({"test_loss": float(loss), "test_accuracy": float(accuracy)}, f, indent=4)

print(f"Test accuracy: {accuracy:.4f}")
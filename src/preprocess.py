import os
import yaml
import numpy as np
from sklearn.model_selection import train_test_split

with open("params.yaml") as f:
    params = yaml.safe_load(f)["preprocess"]

x_train = np.load("data/raw/x_train.npy")
y_train = np.load("data/raw/y_train.npy")
x_test = np.load("data/raw/x_test.npy")
y_test = np.load("data/raw/y_test.npy")

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train,
    test_size=params["test_size"],
    random_state=params["seed"]
)

os.makedirs("data/processed", exist_ok=True)

np.save("data/processed/x_train.npy", x_train)
np.save("data/processed/y_train.npy", y_train)
np.save("data/processed/x_val.npy", x_val)
np.save("data/processed/y_val.npy", y_val)
np.save("data/processed/x_test.npy", x_test)
np.save("data/processed/y_test.npy", y_test)

print(f"Processed data saved: {x_train.shape[0]} train, {x_val.shape[0]} val, {x_test.shape[0]} test")
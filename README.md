# fashion-ann-pipeline

A simple ML pipeline that trains a fully-connected neural network on
Fashion-MNIST. Data, model and metrics are versioned with DVC and
stored on Google Drive.

## Pipeline
prepare -> preprocess -> train -> evaluate

## Setup
```
pip install tensorflow "dvc[gdrive]" pyyaml scikit-learn matplotlib pandas
```

## Run
```
dvc pull
dvc repro
```
Hyperparameters are set in `params.yaml`.

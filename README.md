# ECG Anomaly Detection with LSTM Autoencoder

![Python](https://img.shields.io/badge/python-3.x-blue)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-blue)
![pandas](https://img.shields.io/badge/pandas-1.x-blue)
![NumPy](https://img.shields.io/badge/NumPy-1.x-blue)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.x-blue)
![Seaborn](https://img.shields.io/badge/Seaborn-0.x-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-0.x-blue)
![ARFF2Pandas](https://img.shields.io/badge/arff2pandas-0.x-blue)

## Overview

This project utilizes an LSTM-based autoencoder for ECG (electrocardiogram) anomaly detection. The goal is to train a model to detect abnormal heartbeats based on time series data and classify them into categories such as 'Normal', 'PVC', 'Anomaly', etc. The main idea is to use normal heartbeats to train the autoencoder and detect anomalies based on reconstruction error.

### Key Features:
- **LSTM Autoencoder**: The model employs LSTM layers to capture the temporal dependencies in ECG time series data.
- **Anomaly Detection**: After training on normal heartbeats, the model identifies anomalies by measuring reconstruction loss.
- **Visualization**: Visualizations of ECG time series for different heartbeat classes.
- **Thresholding**: The model uses a threshold to classify a heartbeat as either normal or abnormal based on the reconstruction error.

---

## Setup

### Requirements

- Python 3.x
- PyTorch
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- arff2pandas

You can install the dependencies using pip:

```bash
pip install torch pandas numpy matplotlib seaborn scikit-learn arff2pandas
```

## Model Description
## Architecture
The model is an autoencoder with the following components:

- **Encoder:** The encoder uses two LSTM layers to compress the input time series data into a lower-dimensional representation.
- **Decoder:** The decoder reconstructs the input data from the compressed representation using two more LSTM layers.
- **Reconstruction Loss:** The model minimizes L1 loss to reduce the error between the input and reconstructed sequences.

## Code Overview

- **Data Preprocessing:** Data is loaded from ARFF files, cleaned, and split into normal and anomaly examples.
- **Model Training:** The model is trained on normal ECG data, and its performance is validated using a separate validation set.
- **Prediction:** After training, the model predicts the reconstruction error on both normal and anomaly test data.
- **Evaluation:** A threshold is set based on the reconstruction error to classify heartbeats as either normal or abnormal.

## Training the Model
To train the model, simply run the training function:

```python
model, history = train_model(
  model,
  train_dataset,
  val_dataset,
  n_epochs=150
)
```
The training process will log the loss for each epoch, and once finished, the model will be saved as model.pth.

## Evaluation and Inference

After training, you can evaluate the model using the test data:

```python
predictions, pred_losses = predict(model, test_normal_dataset)
```
You can visualize the reconstruction loss distributions for both normal and anomaly test datasets.


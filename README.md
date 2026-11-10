# PJM Hourly Energy Load Forecasting with PyTorch

A deep learning time-series project predicting PJM regional hourly electricity consumption (in Megawatts) using a Multi-Layer Perceptron (ANN) built with PyTorch.

## Overview
Accurate load forecasting is essential for power grid stability, resource allocation, and preventing energy shortfalls. This project models non-linear demand patterns, seasonal trends, and temporal dependencies using cyclic time features and historical lag variables.

## Project Structure
```text
├── data/
│   └── PJME_hourly.csv           # Raw PJM hourly energy load dataset
├── notebooks/
│   └── 01_exploratory_data_analysis.ipynb  # EDA, model prototyping & experiments
├── src/
│   ├── dataset.py                # Preprocessing, cyclic encoding, lag creation & scaling
│   ├── model.py                  # PyTorch ANN architecture definition
│   ├── train.py                  # Training loop with validation & checkpoint saving
│   └── evaluate.py               # Metric evaluation & plot generation
├── actual_vs_predicted.png       # Generated forecast visualization
├── best_model.pt                 # Saved model weights checkpoint
└── README.md                     # Project documentation
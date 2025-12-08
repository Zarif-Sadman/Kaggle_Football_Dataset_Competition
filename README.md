# Match Outcome Prediction

This repository contains machine learning and deep learning models designed to predict match outcomes (Home Win, Draw, Away Win). The project compares traditional tabular models against deep sequence models and includes tools for performance visualization.

## Project Structure

### 1. Tabular Models

**File:** `tabular_LR_RF_XGB_MLP.py`

This script handles the training and evaluation of models based on engineered tabular features. It includes the following algorithms:

- **Logistic Regression (LR)**
- **Random Forest (RF)**
- **XGBoost (XGB)**
- **Multi-Layer Perceptron (MLP)**

### 2. Deep Sequence Models

**File:** `deep_sequence_models_gru_cnn.py`

This script implements deep learning architectures designed to learn from historical sequence data:

- **Gated Recurrent Unit (GRU)**: For capturing temporal dependencies in match history.
- **Convolutional Neural Network (CNN)**: For extracting local patterns from sequence windows.

### 3. Visualization

**File:** `Plot.py`

A utility script used to generate performance visualizations, including:

- Calibration curves (reliability diagrams)
- Confusion matrices
- Metric comparison plots

## Installation

1. Clone the repository.

2. Install the required dependencies using the provided requirements file:

   ```bash
   pip install -r requirements.txt

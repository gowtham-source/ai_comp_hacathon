# TENZERO
---
# Heart Rate Prediction Project

![Envision_a_closeup_of_a_persons_chest_their_h (1)](https://github.com/gowtham-source/ai_comp_hacathon/assets/75899822/eb7e623f-d5ca-4432-82de-5f9fa0ef900a)

## Overview

This project aims to predict heart rates based on various physiological features using machine learning models. The goal is to explore different models, analyze their performance, and provide insights into the predictive capabilities for heart rate estimation.

## Analysis and Experimentation

### Data Analysis

The dataset used for this project does not contain any null or nan values. It consists of several physiological features, including VLF, LF, HF, TP, and various statistical measures. The analysis involves exploring the characteristics of the dataset.

### Model Selection

Three models were selected for experimentation: Linear Regression, Naive Bayes, and Random Forest. The performance of each model was assessed, with the Linear Regression achieving an accuracy of 96.28%. Naive Bayes showed a marginal improvement, but further evaluation is needed for real-world applications. The Random Forest model demonstrated robustness in capturing unseen patterns and probabilistic values.

### Model Accuracy Comparison

| Model Name         | Accuracy     |
|--------------------|--------------|
| Linear Regression  | 96.28%       |
| Naive Bayes        | 96.35%       |
| Random Forest      | 99.95%%      |
| Optimized LSTM     | 99.86%       |
| XGBoost            | 99.75%       |

### Future Work

Future work may involve optimizing the LSTM network, exploring other neural network architectures, and refining feature engineering to enhance model performance.

## Running the Code

### Prerequisites

- Python 3
- Required libraries (scikit-learn, pandas, xgboost, keras)

### Usage

1. Clone the repository.
2. Install required dependencies: `pip install -r requirements.txt`
3. Run the main script: `python run.py <path_to_csv> <model_name>`

Replace `<path_to_csv>` with the path to the input CSV file and `<model_name>` with the desired model (default is 'lstm').

```python run.py test_data.csv```

### Model Loading Comments

```python
# Load models
rf_model = joblib.load('random_forest_model.joblib')
xg_model = joblib.load('xgboost_model.joblib')
lstm_model = load_model('lstm_model.h5')

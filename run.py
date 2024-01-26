import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBRegressor
import joblib

# Command line arguments
csv_file_path = sys.argv[1]  # Input CSV file path
model_name = sys.argv[2] if len(sys.argv) > 2 else 'random_forest'  # Model name (default: random_forest)

# Load CSV file
data = pd.read_csv(csv_file_path)

# Extract features
X_columns = ['VLF', 'VLF_PCT', 'LF', 'LF_PCT', 'LF_NU', 'HF', 'HF_PCT',
          'HF_NU', 'TP', 'LF_HF', 'HF_LF', 'SD1', 'SD2', 'sampen', 'higuci',
          'datasetId', 'condition', 'MEAN_RR', 'MEDIAN_RR', 'SDRR', 'RMSSD',
          'SDSD', 'SDRR_RMSSD', 'pNN25', 'pNN50', 'KURT', 'SKEW',
          'MEAN_REL_RR', 'MEDIAN_REL_RR', 'SDRR_REL_RR', 'RMSSD_REL_RR',
          'SDSD_REL_RR', 'SDRR_RMSSD_REL_RR', 'KURT_REL_RR', 'SKEW_REL_RR']

# Preprocess data based on the specified model
if model_name in ['random_forest', 'xg_boost']:
    # Random Forest and XGBoost preprocessing
    X_test = data[X_columns]

    # One-hot encode the 'condition' column
    encoder = OneHotEncoder(drop='first', sparse=False)
    X_encoded = pd.concat([X_test, pd.DataFrame(encoder.fit_transform(X_test[['condition']]), columns=encoder.get_feature_names_out(['condition']))], axis=1)

    # Drop the original 'condition' column
    X_encoded = X_encoded.drop(['condition'], axis=1)

    # Standardize the features using StandardScaler
    scaler = StandardScaler()
    X_test_scaled = scaler.fit_transform(X_encoded)

# Load models
if model_name == 'random_forest':
    model = joblib.load('random_forest_model.joblib')
elif model_name == 'xg_boost':
    model = joblib.load('xgboost_model.joblib')

# Initialize lists to store predictions and uuids
predictions = []
uuids = []

# Iterate through each row of the input CSV file
for i in range(len(X_test_scaled)):
    X_row = np.expand_dims(X_test_scaled[i], axis=0)  # Reshape the data for prediction

    # Make predictions based on the specified model
    y_pred = model.predict(X_row)

    # Append predictions and corresponding uuids to the lists
    predictions.append(y_pred[0]-8.6)
    uuids.append(data['uuid'].iloc[i])

# Create DataFrame with uuid and predicted HR values
output_df = pd.DataFrame({'uuid': uuids, 'HR': predictions})

# Save the output DataFrame to 'output.csv'
output_df.to_csv('result.csv', index=False)

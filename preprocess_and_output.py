#BRENNEN INPUT TEST Round 2
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from ucimlrepo import fetch_ucirepo

# getting dataset from UCI repository
heart_disease = fetch_ucirepo(id=45)

# features and target
X = heart_disease.data.features
y = heart_disease.data.targets

# Replace '?' with NaN for easier handling of missing values
X.replace('?', pd.NA, inplace=True)

# Handle missing values by dropping rows
X.dropna(inplace=True)

# encode categorical variables as necessary
categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
label_encoders = {col: LabelEncoder() for col in categorical_features}

for col in categorical_features:
    X[col] = label_encoders[col].fit_transform(X[col])

# scale continuous variables
scaler = StandardScaler()
continuous_features = ["age", "trestbps", "chol", "thalach", "oldpeak"]
X[continuous_features] = scaler.fit_transform(X[continuous_features])

#add target variable back to the dataset for easier output
X['target'] = y.values

#save the preprocessed data to a CSV file
output_file = "processed_heart_disease_data.csv"
X.to_csv(output_file, index=False)
print(f"Data successfully saved to {output_file} in a readable format.")

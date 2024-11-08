import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo
from sklearn.metrics import accuracy_score

# dataset from UCI repository
heart_disease = fetch_ucirepo(id=45) 

# Separate features and target
X = heart_disease.data.features 
y = heart_disease.data.targets

# Scale continuous features 
scaler = StandardScaler()
continuous_features = ["age", "trestbps", "chol", "thalach", "oldpeak"]
X[continuous_features] = scaler.fit_transform(X[continuous_features])

# Splitting the dataset 50/50 for training 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Training Naive Bayes model, (model fitting)
model = GaussianNB()
model.fit(X_train, y_train)

# predicting and calculating accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)

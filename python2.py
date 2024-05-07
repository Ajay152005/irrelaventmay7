import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb

# Generate synthetic image dataset
X, y = make_classification(n_samples=1000, n_features=100, n_classes= 2, random_state=42)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define XGBoost model
model = xgb.XGBClassifier()

# Define hyperparameters grid for tuning
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.1, 0.01, 0.001]
}

# Perform grid search cross-validation for hyperparameter tuning
grid_search = GridSearchCV(estimator=model, param_grid= param_grid, cv = 3, n_jobs = -1)

grid_search.fit(X_train, y_train)

# Get the best hyperparameters

best_params = grid_search.best_params_
print("Best Hyperparameters: ", best_params)

# Train the model with the best hyperparameters
best_model = xgb.XGBClassifier(**best_params)
best_model.fit(X_train, y_train)

# Predictions on the test set
y_pred = best_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: ", accuracy)

# Classification report
print("\nClassification Report: ")
print(classification_report(y_test, y_pred))
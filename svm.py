# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, f1_score
import joblib

# Load the dataset
csv_path = "fingerprint_features.csv"
data = pd.read_csv(csv_path)

# Separate features (X) and labels (y)
X = data.drop("label", axis=1).values
y = data["label"].values

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standardize the feature values (zero mean, unit variance)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit on train, transform train
X_test_scaled = scaler.transform(X_test)        # Transform test using same scaler

# Initialize SVM model with RBF kernel
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)

# Train the SVM model
svm_model.fit(X_train_scaled, y_train)

# Predict the labels for the test set
y_pred = svm_model.predict(X_test_scaled)

# Evaluate the model
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Calculate and print additional metrics
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Recall:   {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Save the trained model and scaler to disk
joblib.dump(svm_model, "svm_rbf_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("Model and scaler saved.")

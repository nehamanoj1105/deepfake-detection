import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import numpy as np

df = pd.read_csv("fingerprint_features.csv")
X = df.drop("label", axis=1)
y = df["label"]

# Train-test split (keeping for validation purposes)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# XGBoost model with regularization parameters
model = XGBClassifier(
    eval_metric='logloss',
    max_depth=6,              # Controls the complexity of the model
    min_child_weight=2,       # Controls the minimum sum of instance weight (hessian) in a child
    subsample=0.8,            # Prevents overfitting by randomly sampling the training data
    colsample_bytree=0.8,     # Prevents overfitting by randomly sampling the features
    use_label_encoder=False
)

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')

print(f"Cross-validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Train the model on the entire training data
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model on the test set
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1-Score:", f1_score(y_test, y_pred))

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


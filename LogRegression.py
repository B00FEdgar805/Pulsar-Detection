import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

# Load the CSV.
df = pd.read_csv("HTRU_2.csv", header=None)

# Assign meaningful column names
df.columns = [
    "profile_mean",
    "profile_std",
    "profile_kurtosis",
    "profile_skewness",
    "dm_snr_mean",
    "dm_snr_std",
    "dm_snr_kurtosis",
    "dm_snr_skewness",
    "target"
]

# Separate input features and target
X = df.drop(columns="target")
y = df["target"]

# Split the data while preserving the class proportions
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# Create a preprocessing and model pipeline
model = Pipeline([
    ("scaler", StandardScaler()),
    ("logistic_regression", LogisticRegression(
        class_weight="balanced",
        solver='liblinear',
        penalty='l1',
        C=0.1,
        max_iter=1000,
        random_state=42
    ))
])

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_probability = model.predict_proba(X_test)[:, 1]

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Balanced accuracy:", balanced_accuracy_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_probability))

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    display_labels=["Non-pulsar", "Pulsar"],
    cmap="Blues"
)

plt.title("Logistic Regression Confusion Matrix")
plt.show()

print("\nClassification report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Non-pulsar", "Pulsar"]
))
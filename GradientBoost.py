import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from xgboost import XGBClassifier


df = pd.read_csv("HTRU_2.csv", header=None)

# Assign meaningful column names
df.columns = [
    "profile_mean",
    "profile_std",
    "profile_kurtosis",
    "profile_skewness",
    "dm-snr_mean",
    "dm-snr_std",
    "dm-snr_kurtosis",
    "dm-snr_skewness",
    "target"
]

# Separate input features and target
X = df.drop(columns="target")
y = df["target"]

# Keep the proportion of pulsars similar in both sets.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y,
)

# Calculate the class weight using the training data only.
negative_count = (y_train == 0).sum()
positive_count = (y_train == 1).sum()
class_weight = negative_count / positive_count

print(f"Training examples: {len(X_train)}")
print(f"Test examples: {len(X_test)}")
print(f"Positive class weight: {class_weight:.2f}")

# Create and train the model.
model = XGBClassifier(
    objective="binary:logistic",
    n_estimators=200,
    max_depth=6,
    learning_rate=0.01,
    subsample=0.7,
    min_child_weight=1,
    colsample_bytree=0.9,
    scale_pos_weight=3,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train, y_train)

# Class predictions and scores for the positive class.
y_pred = model.predict(X_test)
y_scores = model.predict_proba(X_test)[:, 1]

print("\nTest accuracy:", f"{accuracy_score(y_test, y_pred):.4f}")
print("ROC-AUC:", f"{roc_auc_score(y_test, y_scores):.4f}")
print("Balanced accuracy:", balanced_accuracy_score(y_test, y_pred))
print("Average precision:", f"{average_precision_score(y_test, y_scores):.4f}")

print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))


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
    target_names=["Non-pulsar (0)", "Pulsar (1)"],
    zero_division=0,
))
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from xgboost import XGBClassifier


# The CSV does not have a header, so provide column names.
column_names = [
    "profile_mean",
    "profile_std",
    "profile_kurtosis",
    "profile_skewness",
    "dm_snr_mean",
    "dm_snr_std",
    "dm_snr_kurtosis",
    "dm_snr_skewness",
    "target",
]

df = pd.read_csv("HTRU_2.csv", header=None, names=column_names)

# First eight columns are features; the last column is the class.
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
    max_depth=3,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=class_weight,
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
print("Average precision:", f"{average_precision_score(y_test, y_scores):.4f}")

print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Non-pulsar (0)", "Pulsar (1)"],
    zero_division=0,
))
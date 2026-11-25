import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
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
import seaborn as sns



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

# Visualize the class distribution
print(df["target"].value_counts())
sns.countplot(data=df, x="target")
plt.xticks([0, 1], ["Non-pulsar", "Pulsar"])
plt.title("Class Distribution")
plt.xlabel("Class")
plt.ylabel("Number of candidates")
plt.tight_layout()
plt.show()


# Visualize the distribution of profile_mean by class
sns.boxplot(data=df, x="target", y="profile_mean")
plt.xticks([0, 1], ["Non-pulsar", "Pulsar"])
plt.title("Profile Mean by Class")
plt.tight_layout()
plt.show()

# Split the data while preserving the class proportions
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# Create a preprocessing and model pipeline
# This is the best model found after hyperparameter tuning using RandomizedSearchCV.
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

# Plot the coefficients from the tuned model
coefficients = model.named_steps["logistic_regression"].coef_[0]
coefficient_series = pd.Series(coefficients, index=X.columns)
coefficient_series = coefficient_series.sort_values()

colors = ["#d62728" if value < 0 else "#1f77b4" for value in coefficient_series]
coefficient_series.plot(
    kind="barh",
    figsize=(9, 6),
    color=colors,
)
plt.axvline(0, color="black", linewidth=0.8)
plt.title("Logistic Regression Coefficient Plot")
plt.xlabel("Coefficient value")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# Trace how each coefficient changes with the regularization strength
C_values = np.logspace(-4, 2, 25)
coefficient_paths = []

for C in C_values:
    path_model = Pipeline([
        ("scaler", StandardScaler()),
        ("logistic_regression", LogisticRegression(
            class_weight="balanced",
            solver="liblinear",
            penalty="l1",
            C=C,
            max_iter=1000,
            random_state=42,
        )),
    ])
    path_model.fit(X_train, y_train)
    coefficient_paths.append(
        path_model.named_steps["logistic_regression"].coef_[0]
    )

coefficient_paths = np.asarray(coefficient_paths)

fig, ax = plt.subplots(figsize=(9, 6))
for feature_index, feature_name in enumerate(X.columns):
    ax.plot(
        C_values,
        coefficient_paths[:, feature_index],
        label=feature_name,
    )

ax.set_xscale("log")
ax.axvline(0.1, color="black", linestyle="--", linewidth=1, label="Selected C")
ax.set_title("Coefficient Curve")
ax.set_xlabel("Regularization strength C (log scale)")
ax.set_ylabel("Coefficient value")
ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
plt.tight_layout()
plt.show()

# Compare cross-validated balanced accuracy across regularization strengths
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_scores = []
validation_scores = []

for C in C_values:
    regularization_model = Pipeline([
        ("scaler", StandardScaler()),
        ("logistic_regression", LogisticRegression(
            class_weight="balanced",
            solver="liblinear",
            penalty="l1",
            C=C,
            max_iter=1000,
            random_state=42,
        )),
    ])
    cv_results = cross_validate(
        regularization_model,
        X_train,
        y_train,
        cv=cv,
        scoring="balanced_accuracy",
        return_train_score=True,
        n_jobs=-1,
    )
    train_scores.append(cv_results["train_score"].mean())
    validation_scores.append(cv_results["test_score"].mean())

fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(C_values, train_scores, marker="o", label="Training score")
ax.plot(C_values, validation_scores, marker="o", label="Cross-validation score")
ax.set_xscale("log")
ax.axvline(0.1, color="black", linestyle="--", linewidth=1, label="Selected C")
ax.set_title("Regularization Curve")
ax.set_xlabel("Regularization strength C (log scale)")
ax.set_ylabel("Balanced accuracy")
ax.legend()
plt.tight_layout()
plt.show()

# Make predictions
y_pred = model.predict(X_test)
y_probability = model.predict_proba(X_test)[:, 1]

# Evaluate the model
print("\nTest accuracy:", accuracy_score(y_test, y_pred))
print("Balanced accuracy:", balanced_accuracy_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_probability))

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
    target_names=["Non-pulsar", "Pulsar"],
    zero_division=0,

))
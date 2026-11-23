import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
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
    "target",
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
    stratify=y,
)

# Train and evaluate the gradient boosting model
negative_count = (y_train == 0).sum()
positive_count = (y_train == 1).sum()
class_weight = negative_count / positive_count

gradient_model = XGBClassifier(
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

gradient_model.fit(X_train, y_train)

gradient_pred = gradient_model.predict(X_test)
gradient_scores = gradient_model.predict_proba(X_test)[:, 1]

print("\nGradient Boosting")
print("Test accuracy:", f"{accuracy_score(y_test, gradient_pred):.4f}")
print("ROC-AUC:", f"{roc_auc_score(y_test, gradient_scores):.4f}")
print("Balanced accuracy:", balanced_accuracy_score(y_test, gradient_pred))
print(
    "Average precision:",
    f"{average_precision_score(y_test, gradient_scores):.4f}",
)

print("\nConfusion matrix:")
print(confusion_matrix(y_test, gradient_pred))

ConfusionMatrixDisplay.from_predictions(
    y_test,
    gradient_pred,
    display_labels=["Non-pulsar", "Pulsar"],
    cmap="Blues",
)
plt.title("Gradient Boosting Confusion Matrix")
plt.show()

print("\nClassification report:")
print(
    classification_report(
        y_test,
        gradient_pred,
        target_names=["Non-pulsar", "Pulsar"],
        zero_division=0,
    )
)

# Train and evaluate the logistic regression model
logistic_model = Pipeline(
    [
        ("scaler", StandardScaler()),
        (
            "logistic_regression",
            LogisticRegression(
                class_weight="balanced",
                solver="liblinear",
                penalty="l1",
                C=0.1,
                max_iter=1000,
                random_state=42,
            ),
        ),
    ]
)

logistic_model.fit(X_train, y_train)

logistic_pred = logistic_model.predict(X_test)
logistic_scores = logistic_model.predict_proba(X_test)[:, 1]

print("\nLogistic Regression")
print("Test accuracy:", accuracy_score(y_test, logistic_pred))
print("Balanced accuracy:", balanced_accuracy_score(y_test, logistic_pred))
print("ROC-AUC:", roc_auc_score(y_test, logistic_scores))

print("\nConfusion matrix:")
print(confusion_matrix(y_test, logistic_pred))

ConfusionMatrixDisplay.from_predictions(
    y_test,
    logistic_pred,
    display_labels=["Non-pulsar", "Pulsar"],
    cmap="Blues",
)
plt.title("Logistic Regression Confusion Matrix")
plt.show()

print("\nClassification report:")
print(
    classification_report(
        y_test,
        logistic_pred,
        target_names=["Non-pulsar", "Pulsar"],
        zero_division=0,
    )
)

from sklearn.metrics import PrecisionRecallDisplay

fig, ax = plt.subplots(figsize=(7, 5))

PrecisionRecallDisplay.from_estimator(
    logistic_model,
    X_test,
    y_test,
    name="Logistic Regression",
    ax=ax
)

PrecisionRecallDisplay.from_estimator(
    gradient_model,
    X_test,
    y_test,
    name="Gradient Boosting",
    ax=ax
)

# Positive-class proportion: a useful baseline for the PR curve.
ax.axhline(
    y=y_test.mean(),
    color="gray",
    linestyle="--",
    label="Pulsar proportion"
)

ax.set_title("Precision–Recall Curves")
ax.legend()
plt.tight_layout()
plt.show()


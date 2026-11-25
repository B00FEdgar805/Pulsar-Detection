import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.inspection import PartialDependenceDisplay
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
    stratify=y,
)

# Keep the test set untouched while reserving data for validation.


# Calculate the class weight using the training data only.
negative_count = (y_train == 0).sum()
positive_count = (y_train == 1).sum()
class_weight = negative_count / positive_count

# Create and train the model.
# This is the best model found after hyperparameter tuning using RandomizedSearchCV.
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

# Plot training and validation log loss for each boosting round.
loss_history = model.evals_result()
fig, ax = plt.subplots(figsize=(8, 5))
for dataset_name, metrics in loss_history.items():
    label = "Training" if dataset_name.endswith("0") else "Validation"
    ax.plot(range(1, len(metrics["logloss"]) + 1), metrics["logloss"], label=label)
ax.set_title("Training and Validation Loss")
ax.set_xlabel("Boosting round")
ax.set_ylabel("Log loss")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Plot feature importance, sorted from least to most important.
feature_importance = pd.Series(
    model.feature_importances_,
    index=X.columns,
).sort_values()
fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(feature_importance.index, feature_importance.values, color="steelblue")
ax.set_title("Feature Importance")
ax.set_xlabel("Importance")
plt.tight_layout()
plt.show()



y_pred = model.predict(X_test)
y_scores = model.predict_proba(X_test)[:, 1]

# Evaluate the model
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

plt.title("Gradient Boosting Confusion Matrix")
plt.show()

print("\nClassification report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Non-pulsar", "Pulsar"],
    zero_division=0,
))
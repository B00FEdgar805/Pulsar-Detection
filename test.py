import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score,
    roc_auc_score
)

# Load data
df = pd.read_csv("HTRU_2.csv", header=None)

X = df.iloc[:, :8]
y = df.iloc[:, 8]

# Keep the final test set untouched
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# Scaling and model
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=5000))
])

# Values to test
param_grid = [
    {
        "model__solver": ["liblinear"],
        "model__penalty": ["l1", "l2"],
        "model__C": [0.001, 0.01, 0.1, 1, 10, 100],
        "model__class_weight": [None, "balanced"]
    },
    {
        "model__solver": ["lbfgs"],
        "model__penalty": ["l2"],
        "model__C": [0.001, 0.01, 0.1, 1, 10, 100],
        "model__class_weight": [None, "balanced"]
    }
]

# Preserve class proportions in every fold
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="average_precision",  # PR-AUC; useful for imbalanced data
    cv=cv,
    n_jobs=-1,
    refit=True
)

search.fit(X_train, y_train)

print("Best parameters:")
print(search.best_params_)

print("\nBest cross-validation PR-AUC:")
print(search.best_score_)

# Evaluate once on the untouched test set
y_probability = search.predict_proba(X_test)[:, 1]
y_pred = search.predict(X_test)

print("\nTest PR-AUC:", average_precision_score(y_test, y_probability))
print("Test ROC-AUC:", roc_auc_score(y_test, y_probability))

print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Non-pulsar", "Pulsar"]
))
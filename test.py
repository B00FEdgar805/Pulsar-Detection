import pandas as pd

from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from xgboost import XGBClassifier


# ----------------------------------------
# 1. Load the dataset
# ----------------------------------------

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

X = df.drop(columns="target")
y = df["target"]


# ----------------------------------------
# 2. Split into training and test sets
# ----------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y,
)


# ----------------------------------------
# 3. Create the base model
# ----------------------------------------

model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    random_state=42,
    n_jobs=1,
)


# ----------------------------------------
# 4. Define settings to try
# ----------------------------------------

param_distributions = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [2, 3, 4, 5, 6],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 3, 5, 10],
    "scale_pos_weight": [1, 3, 6, 10],
}


# ----------------------------------------
# 5. Set up cross-validation
# ----------------------------------------

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_distributions,
    n_iter=30,
    scoring="average_precision",
    cv=cv,
    random_state=42,
    n_jobs=-1,
    verbose=2,
    refit=True,
)


# ----------------------------------------
# 6. Run the search
# ----------------------------------------

print("Starting hyperparameter search...")

search.fit(X_train, y_train)

print("\nBest parameters:")
for parameter, value in search.best_params_.items():
    print(f"  {parameter}: {value}")

print(
    "\nBest mean cross-validation average precision:",
    f"{search.best_score_:.4f}",
)

# refit=True retrains the best settings on all training data.
best_model = search.best_estimator_


# ----------------------------------------
# 7. Evaluate the best model on test data
# ----------------------------------------

y_pred = best_model.predict(X_test)
y_scores = best_model.predict_proba(X_test)[:, 1]

print("\n--- Test set results ---")
print(f"Accuracy:          {accuracy_score(y_test, y_pred):.4f}")
print(f"ROC-AUC:           {roc_auc_score(y_test, y_scores):.4f}")
print(f"Average precision: {average_precision_score(y_test, y_scores):.4f}")

print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Non-pulsar (0)", "Pulsar (1)"],
        zero_division=0,
    )
)
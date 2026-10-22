
import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

# 1. Load both datasets
customers = pd.read_csv("archive/customer_data.csv")
payments = pd.read_csv("archive/payment_data.csv")

print("Customers:", customers.shape)
print("Payment records:", payments.shape)

# 2. Aggregate payment history per customer
# Exclude identifiers and date columns from aggregation.
payment_features = [
    "OVD_t1",
    "OVD_t2",
    "OVD_t3",
    "OVD_sum",
    "pay_normal",
    "prod_limit",
    "new_balance",
    "highest_balance"
]

# Keep only columns that actually exist
payment_features = [
    col for col in payment_features if col in payments.columns
]

for col in payment_features:
    payments[col] = pd.to_numeric(
        payments[col], errors="coerce"
    )

# Summarize multiple products/payment records per customer
payment_summary = payments.groupby("id")[payment_features].agg(
    ["sum", "mean", "max"]
)

# Flatten column names
payment_summary.columns = [
    f"{col}_{stat}" for col, stat in payment_summary.columns
]
payment_summary = payment_summary.reset_index()

# 3. Merge with customer data
df = customers.merge(
    payment_summary,
    on="id",
    how="left",
    validate="one_to_one"
)

print("Merged dataset:", df.shape)

# 4. Prepare target and features
df = df.dropna(subset=["label"])
df = df[df["label"].isin([0, 1])].copy()

y = df["label"].astype(int)

# ID is not a predictive feature
X = df.drop(columns=["label", "id"])

# Convert features to numeric and handle infinities
X = X.apply(pd.to_numeric, errors="coerce")
X = X.replace([np.inf, -np.inf], np.nan)

# Remove columns with no usable values
X = X.dropna(axis=1, how="all")
feature_names = X.columns.tolist()

print("\nClass distribution:")
print(y.value_counts())

# 5. Split before fitting any preprocessing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 6. Train two simple models
models = {
    "Logistic Regression": make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=42
        )
    ),
    "Random Forest": make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
            min_samples_leaf=3
        )
    )
}

os.makedirs("models", exist_ok=True)

best_name = None
best_model = None
best_auc = -1

# 7. Evaluate models
for name, model in models.items():
    print("\n" + "=" * 45)
    print(name)
    print("=" * 45)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    print(
        "Accuracy:",
        round(accuracy_score(y_test, predictions), 4)
    )
    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Low Risk", "High Risk"],
        zero_division=0
    ))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))

    auc = roc_auc_score(y_test, probabilities)
    print("ROC-AUC:", round(auc, 4))

    if auc > best_auc:
        best_auc = auc
        best_name = name
        best_model = model

# 8. Save the best model by ROC-AUC
joblib.dump(best_model, "models/credit_risk_model.pkl")
joblib.dump(feature_names, "models/features.pkl")
joblib.dump(best_name, "models/model_name.pkl")

print("\nSelected model:", best_name)
print("Selected model ROC-AUC:", round(best_auc, 4))
print("Model and feature names saved successfully!")

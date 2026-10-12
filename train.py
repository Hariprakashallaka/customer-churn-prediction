
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# --------------------------------------------------
# 1. Define project paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR /"data" / "customer_churning_dataset.csv"
MODEL_PATH = BASE_DIR / "churn_model.pkl"


# --------------------------------------------------
# 2. Load the dataset
# --------------------------------------------------

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}\n"
        "Place customer_churning_dataset.csv in the project folder."
    )

df = pd.read_csv(DATA_PATH)

# Remove accidental spaces from column names
df.columns = df.columns.str.strip()

print("\nDataset loaded successfully!")
print("Dataset shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())


# --------------------------------------------------
# 3. Validate the target column
# --------------------------------------------------

TARGET = "Churn"

if TARGET not in df.columns:
    raise ValueError(
        f"Target column '{TARGET}' was not found. "
        f"Available columns: {df.columns.tolist()}"
    )

# Remove rows without a target value
df = df.dropna(subset=[TARGET]).copy()

# Separate input features and target
X = df.drop(columns=[TARGET, "CustomerID"], errors="ignore")
y = df[TARGET]

if X.shape[1] == 0:
    raise ValueError("No input features were found.")

if y.nunique() < 2:
    raise ValueError(
        "The target must contain at least two different classes."
    )

if y.value_counts().min() < 2:
    raise ValueError(
        "Each target class needs at least two records "
        "for a stratified train/test split."
    )

print("\nInput features:", X.columns.tolist())
print("\nTarget distribution:")
print(y.value_counts())


# --------------------------------------------------
# 4. Identify numerical and categorical columns
# --------------------------------------------------

numerical_columns = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_columns = X.select_dtypes(
    exclude=["number"]
).columns.tolist()

print("\nNumerical columns:", numerical_columns)
print("Categorical columns:", categorical_columns)


# --------------------------------------------------
# 5. Build preprocessing pipelines
# --------------------------------------------------

numerical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent"),
        ),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore"),
        ),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numerical", numerical_pipeline, numerical_columns),
        ("categorical", categorical_pipeline, categorical_columns),
    ],
    remainder="drop",
)


# --------------------------------------------------
# 6. Create the complete ML pipeline
# --------------------------------------------------

model_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                class_weight="balanced",
                n_jobs=-1,
            ),
        ),
    ]
)


# --------------------------------------------------
# 7. Split the data into training and testing sets
# --------------------------------------------------

class_counts = y.value_counts()
use_stratify = (
    class_counts.min() >= 2
    and len(y) * 0.20 >= y.nunique()
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y if use_stratify else None,
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# --------------------------------------------------
# 8. Train the model
# --------------------------------------------------

print("\nTraining Random Forest model...")

model_pipeline.fit(X_train, y_train)

print("Training completed!")


# --------------------------------------------------
# 9. Evaluate the model on unseen test data
# --------------------------------------------------

y_pred = model_pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n========== MODEL EVALUATION ==========")
print(f"Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0,
    )
)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ROC-AUC is applicable when the target is binary
if y.nunique() == 2:
    y_proba = model_pipeline.predict_proba(X_test)[:, 1]

    try:
        roc_auc = roc_auc_score(
            y_test,
            y_proba,
        )
        print(f"\nROC-AUC: {roc_auc:.4f}")
    except ValueError as error:
        print("\nROC-AUC could not be calculated:", error)


# --------------------------------------------------
# 10. Save the entire fitted pipeline
# --------------------------------------------------

joblib.dump(model_pipeline, MODEL_PATH)

print("\nModel saved successfully!")
print("Model path:", MODEL_PATH)
print("\nNext step: use churn_model.pkl in app.py")
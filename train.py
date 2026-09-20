import os
import joblib

from sklearn.preprocessing import LabelEncoder

from data_preprocessing import (
    load_data,
    split_data,
    create_preprocessor
)

from sklearn.svm import SVC
from xgboost import XGBClassifier


# ============================================
# CONFIGURATION
# ============================================

DATA_PATH = "ai_job_impact.csv"
MODEL_DIR = "models"


# ============================================
# LOAD DATA
# ============================================

X, y = load_data(DATA_PATH)

print("Data loaded successfully.")
print(f"Number of rows: {X.shape[0]}")
print(f"Number of features: {X.shape[1]}")


# ============================================
# ENCODE TARGET VARIABLE
# ============================================

label_encoder = LabelEncoder()

y = label_encoder.fit_transform(y)

print("Target classes:")
print(label_encoder.classes_)


# ============================================
# TRAIN / TEST SPLIT
# ============================================

X_train, X_test, y_train, y_test = split_data(X, y)

print("Train/Test split completed.")
print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")


# ============================================
# CREATE PREPROCESSOR
# ============================================

preprocessor = create_preprocessor()


# ============================================
# FIT PREPROCESSOR ON TRAINING DATA
# ============================================

X_train_processed = preprocessor.fit_transform(X_train)

X_test_processed = preprocessor.transform(X_test)

print("Data preprocessing completed.")


# ============================================
# TRAIN LINEAR SVM
# ============================================

linear_svm = SVC(
    kernel="linear",
    random_state=0,
    probability=True
)

linear_svm.fit(
    X_train_processed,
    y_train
)

print("Linear SVM training completed.")


# ============================================
# TRAIN RBF SVM
# ============================================

rbf_svm = SVC(
    kernel="rbf",
    random_state=0,
    probability=True,
    class_weight="balanced"
)

rbf_svm.fit(
    X_train_processed,
    y_train
)

print("RBF SVM training completed.")


# ============================================
# TRAIN XGBOOST
# ============================================

xgboost_model = XGBClassifier(
    objective="multi:softprob",
    num_class=len(label_encoder.classes_),
    eval_metric="mlogloss",
    random_state=42
)

xgboost_model.fit(
    X_train_processed,
    y_train
)

print("XGBoost training completed.")


# ============================================
# CREATE MODEL DIRECTORY
# ============================================

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================
# SAVE MODELS
# ============================================

joblib.dump(
    linear_svm,
    f"{MODEL_DIR}/linear_svm.pkl"
)

joblib.dump(
    rbf_svm,
    f"{MODEL_DIR}/rbf_svm.pkl"
)

joblib.dump(
    xgboost_model,
    f"{MODEL_DIR}/xgboost_model.pkl"
)


# ============================================
# SAVE PREPROCESSOR
# ============================================

joblib.dump(
    preprocessor,
    f"{MODEL_DIR}/preprocessor.pkl"
)


# ============================================
# SAVE LABEL ENCODER
# ============================================

joblib.dump(
    label_encoder,
    f"{MODEL_DIR}/label_encoder.pkl"
)


# ============================================
# SAVE TEST DATA
# ============================================

joblib.dump(
    X_test,
    f"{MODEL_DIR}/X_test.pkl"
)

joblib.dump(
    y_test,
    f"{MODEL_DIR}/y_test.pkl"
)


print("\n============================================")
print("TRAINING COMPLETED SUCCESSFULLY")
print("============================================")

print("\nSaved files:")

print("models/linear_svm.pkl")
print("models/rbf_svm.pkl")
print("models/xgboost_model.pkl")
print("models/preprocessor.pkl")
print("models/label_encoder.pkl")
print("models/X_test.pkl")
print("models/y_test.pkl")

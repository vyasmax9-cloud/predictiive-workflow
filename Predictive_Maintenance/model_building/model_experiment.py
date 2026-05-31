import os
import json
import joblib
import pandas as pd
from datasets import load_dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from huggingface_hub import HfApi, create_repo

# ==========================================
# 1. ENVIRONMENT SETUP & REPOSITORY CONFIG
# ==========================================
TOKEN = os.getenv("HF_TOKEN")
DATASET_REPO = "vyasmax9/predictive-maintenance-engine"
MODEL_REPO = "vyasmax9/predictive-maintenance-model"
api = HfApi(token=TOKEN)

MODEL_PATH = os.path.join(os.getcwd(), "Predictive_Maintenance", "models")
os.makedirs(MODEL_PATH, exist_ok=True)

# ==========================================
# 2. LOAD TRAIN / TEST DATA SPLITS FROM HUB
# ==========================================
print("--- Loading train and test CSV files from Hugging Face Hub ---")
train_df = load_dataset(DATASET_REPO, data_files="train.csv", split="train").to_pandas()
test_df = load_dataset(DATASET_REPO, data_files="test.csv", split="train").to_pandas()

TARGET = "Engine_Condition"
X_train = train_df.drop(columns=[TARGET])
y_train = train_df[TARGET]
X_test = test_df.drop(columns=[TARGET])
y_test = test_df[TARGET]

# ==========================================
# 3. INITIALIZE CHAMPION WITH BEST PARAMETERS
# ==========================================
print("\n--- Training Champion Random Forest Classifier with Optimized Parameters ---")

# Replace these key-value pairs with the exact values printed by your Random Forest cell
best_params = {
    "n_estimators": 200,
    "max_depth": None,
    "min_samples_split": 2,
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1
}

best_model = RandomForestClassifier(**best_params)
best_model.fit(X_train, y_train)

# ==========================================
# 4. FINAL PRODUCTION EVALUATION MATRICES
# ==========================================
preds = best_model.predict(X_test)

final_metrics = {
    "Accuracy": round(accuracy_score(y_test, preds), 4),
    "Precision": round(precision_score(y_test, preds, average='weighted'), 4),
    "Recall": round(recall_score(y_test, preds, average='weighted'), 4),
    "F1-Score": round(f1_score(y_test, preds, average='weighted'), 4)
}

print("\nFinal Optimized Model Metrics:")
for metric_name, value in final_metrics.items():
    print(f" - {metric_name}: {value}")

print("\nClassification Report:")
print(classification_report(y_test, preds))

print("Confusion Matrix Array:")
print(confusion_matrix(y_test, preds))

# ==========================================
# 5. LOCAL FILE SERIALIZATION
# ==========================================
# Export parameters dictionary
params_path = os.path.join(MODEL_PATH, "best_params.json")
with open(params_path, "w") as file:
    json.dump(best_params, file, indent=4)

# Export evaluation metrics table
metrics_path = os.path.join(MODEL_PATH, "final_metrics.json")
with open(metrics_path, "w") as file:
    json.dump(final_metrics, file, indent=4)

# Export feature importance scores
if hasattr(best_model, "feature_importances_"):
    importance_df = pd.DataFrame({
        "Feature": X_train.columns,
        "Importance": best_model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    importance_path = os.path.join(MODEL_PATH, "feature_importance.csv")
    importance_df.to_csv(importance_path, index=False)

# Save the final binary weight file
model_file = os.path.join(MODEL_PATH, "best_model.pkl")
joblib.dump(best_model, model_file)
print("\n[SUCCESS] Local serialization of production artifacts completed.")

# ==========================================
# 6. REGISTRATION TO HUGGING FACE MODEL HUB
# ==========================================
print("\n--- Registering Production Model to Hugging Face Model Hub ---")
try:
    create_repo(repo_id=MODEL_REPO, repo_type="model", token=TOKEN, private=False, exist_ok=True)

    upload_list = [model_file, params_path, metrics_path]
    if 'importance_path' in locals():
        upload_list.append(importance_path)

    for file_to_upload in upload_list:
        api.upload_file(
            path_or_fileobj=file_to_upload,
            path_in_repo=os.path.basename(file_to_upload),
            repo_id=MODEL_REPO,
            repo_type="model"
        )
    print("\n[SUCCESS] Production Model Hub Repository fully synchronized on Hugging Face!")
except Exception as e:
    print(f"\n[ERROR] Problem encountered during repository file push: {e}")

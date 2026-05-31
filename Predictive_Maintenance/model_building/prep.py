from huggingface_hub import HfApi
import os

# ==========================================
# HF TOKEN
# ==========================================

TOKEN = os.getenv("HF_TOKEN")

if TOKEN is None:
    raise Exception("HF_TOKEN not found")

TOKEN = TOKEN.strip()

print("HF Token Loaded Successfully")

# ==========================================
# HF API
# ==========================================

api = HfApi()

# ==========================================
# SAVE TRAIN / TEST
# ==========================================

save_path = os.path.join(
    os.getcwd(),
    "prepared_data"
)

os.makedirs(
    save_path,
    exist_ok=True
)

train_path = os.path.join(
    save_path,
    "train.csv"
)

test_path = os.path.join(
    save_path,
    "test.csv"
)

from sklearn.model_selection import train_test_split
import pandas as pd

# Load dataset (CHANGE PATH ACCORDING TO YOUR FILE)
df = pd.read_csv("Predictive_Maintenance/data/dataset.csv")

# Train-test split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

print("Train/Test split done")

train_df.to_csv(
    train_path,
    index=False
)

test_df.to_csv(
    test_path,
    index=False
)

print("Train CSV Saved")
print("Test CSV Saved")

# ==========================================
# UPLOAD TRAIN FILE
# ==========================================

api.upload_file(
    path_or_fileobj=train_path,
    path_in_repo="train.csv",
    repo_id=repo_id,
    repo_type="dataset",
    token=TOKEN
)

print("train.csv uploaded")

# ==========================================
# UPLOAD TEST FILE
# ==========================================

api.upload_file(
    path_or_fileobj=test_path,
    path_in_repo="test.csv",
    repo_id=repo_id,
    repo_type="dataset",
    token=TOKEN
)

print("test.csv uploaded")

# ==========================================
# VERIFY FILES
# ==========================================

files = api.list_repo_files(
    repo_id=repo_id,
    repo_type="dataset",
    token=TOKEN
)

print("\nFiles Uploaded:")
print(files)

print("\nData Preparation Completed")

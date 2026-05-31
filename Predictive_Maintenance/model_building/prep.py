
# ==========================================
# IMPORT LIBRARIES
# ==========================================

import os
import pandas as pd

from datasets import load_dataset
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi


# ==========================================
# CONFIG
# ==========================================

TOKEN = os.getenv("HF_TOKEN")

repo_id = "vyasmax9/predictive-maintenance-engine"

api = HfApi(token=TOKEN)

print("Using Repo:", repo_id)


# ==========================================
# LOAD DATASET FROM HF
# ==========================================

dataset = load_dataset(

    "csv",

    data_files=f"https://huggingface.co/datasets/{repo_id}/resolve/main/engine_data.csv"

)

df = dataset["train"].to_pandas()

print("\nDataset Loaded")

print(df.shape)


# ==========================================
# COLUMN CLEANING
# ==========================================

df.columns = [

    col.strip().replace(" ", "_")

    for col in df.columns

]

print("\nColumns")

print(df.columns.tolist())


# ==========================================
# MISSING VALUES
# ==========================================

print("\nMissing Values")

print(df.isnull().sum())

df = df.dropna()


# ==========================================
# REMOVE DUPLICATES
# ==========================================

duplicates = df.duplicated().sum()

df = df.drop_duplicates()

print("\nDuplicates Removed:", duplicates)

print("Shape:", df.shape)


# ==========================================
# REMOVE UNUSED COLUMNS
# ==========================================

remove_cols = [

    "Unnamed:_0",

    "Unnamed: 0",

    "ID",

    "index"

]

existing_cols = [

    col for col in remove_cols

    if col in df.columns

]

df.drop(

    columns=existing_cols,

    inplace=True,

    errors="ignore"

)


# ==========================================
# FEATURES / TARGET
# ==========================================

target = "Engine_Condition"

X = df.drop(

    columns=[target]

)

y = df[target]


# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)


train_df = X_train.copy()

train_df[target] = y_train


test_df = X_test.copy()

test_df[target] = y_test


# ==========================================
# SAVE LOCALLY
# ==========================================

save_path = os.path.join(

    os.getcwd(),

    "Predictive_Maintenance",

    "data"

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

train_df.to_csv(

    train_path,

    index=False

)

test_df.to_csv(

    test_path,

    index=False

)


# ==========================================
# UPLOAD TRAIN / TEST
# ==========================================

api.upload_file(

    path_or_fileobj=train_path,

    path_in_repo="train.csv",

    repo_id=repo_id,

    repo_type="dataset"

)

api.upload_file(

    path_or_fileobj=test_path,

    path_in_repo="test.csv",

    repo_id=repo_id,

    repo_type="dataset"

)

print("\nUploaded train/test successfully")

print("\nData Preparation Completed")

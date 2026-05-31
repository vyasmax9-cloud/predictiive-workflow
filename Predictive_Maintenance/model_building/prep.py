from huggingface_hub import HfApi 
import os 
import pandas as pd 
from sklearn.model_selection import train_test_split


# HF TOKEN

TOKEN = os.getenv("HF_TOKEN")

if not TOKEN: raise Exception("HF_TOKEN not found")

TOKEN = TOKEN.strip()

print("HF Token Loaded Successfully")


# HF API

api = HfApi()

repo_id = "vyasmax9/predictive-maintenance-engine"

# PATHS

save_path = os.path.join( 
  os.getcwd(), 
  "Predictive_Maintenance", 
  "data" 
)

os.makedirs( save_path, exist_ok=True )

dataset_path = os.path.join( 
  save_path, 
  "engine_data.csv" 
)

print("Dataset Path:", dataset_path) 
print("Dataset Exists:", 
os.path.exists(dataset_path))

# LOAD DATASET

df = pd.read_csv(dataset_path)

print("Dataset Loaded Successfully") 
print("Shape:", df.shape)


# TRAIN TEST SPLIT

train_df, test_df = train_test_split( 
  df, test_size=0.2, 
  random_state=42 
)

print("Train Shape:", train_df.shape) 
print("Test Shape:", test_df.shape)


# SAVE FILES

train_path = os.path.join( 
  save_path,
  "train.csv" 
)

test_path = os.path.join( 
  save_path, 
  "test.csv" 
)

train_df.to_csv( train_path, index=False )

test_df.to_csv( test_path, index=False )

print("Train CSV Saved") 
print("Test CSV Saved")


# UPLOAD TRAIN FILE

api.upload_file( 
  path_or_fileobj=train_path, 
  path_in_repo="train.csv", 
  repo_id=repo_id, 
  repo_type="dataset",
  token=TOKEN
)

print("train.csv uploaded")


# UPLOAD TEST FILE

api.upload_file( 
  path_or_fileobj=test_path, 
  path_in_repo="test.csv", 
  repo_id=repo_id, 
  repo_type="dataset",
  token=TOKEN
)


# VERIFY FILES

files = api.list_repo_files( 
  repo_id=repo_id, 
  repo_type="dataset" 
  token=TOKEN
  )

print("\nFiles Uploaded:") print(files)

print("\nData Preparation Completed")


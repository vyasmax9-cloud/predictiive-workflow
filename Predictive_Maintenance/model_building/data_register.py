from huggingface_hub import (
    HfApi,
    create_repo,
    login
)

from huggingface_hub.utils import RepositoryNotFoundError

import os

# ==========================================
# CONFIGURATION
# ==========================================

DATA_PATH = os.path.join(
    os.getcwd(),
    "Predictive_Maintenance",
    "data"
)

repo_id = "vyasmax9/predictive-maintenance-engine"
repo_type = "dataset"

# ==========================================
# TOKEN CHECK
# ==========================================

TOKEN = os.getenv("HF_TOKEN")

if not TOKEN:
    raise Exception("HF_TOKEN not found")

# Remove hidden spaces/newlines
TOKEN = TOKEN.strip()

print("Token Found:", TOKEN is not None)
print("Token Length:", len(TOKEN))
print("Token Prefix:", TOKEN[:3])

# ==========================================
# LOGIN TO HUGGING FACE
# ==========================================

login(token=TOKEN)

api = HfApi()

# ==========================================
# DEBUG INFORMATION
# ==========================================

print("Current Directory:", os.getcwd())
print("Data Path:", DATA_PATH)
print("Path Exists:", os.path.exists(DATA_PATH))

# ==========================================
# CHECK DATA FOLDER
# ==========================================

if not os.path.exists(DATA_PATH):
    raise Exception(
        f"Path not found: {DATA_PATH}"
    )

# ==========================================
# CREATE DATASET REPOSITORY
# ==========================================

try:

    api.repo_info(
        repo_id=repo_id,
        repo_type=repo_type
    )

    print(
        f"Dataset Repo '{repo_id}' already exists."
    )

except RepositoryNotFoundError:

    print(
        "Creating Dataset Repository..."
    )

    create_repo(
        repo_id=repo_id,
        repo_type=repo_type,
        private=False,
        token=TOKEN
    )

    print(
        "Dataset Repository Created"
    )

except Exception as e:

    print(
        f"Repository Check Error: {e}"
    )
    raise

# ==========================================
# UPLOAD DATA FOLDER
# ==========================================

try:

    api.upload_folder(
        folder_path=DATA_PATH,
        repo_id=repo_id,
        repo_type=repo_type
    )

    print(
        "Folder Uploaded Successfully"
    )

except Exception as e:

    print(
        f"Upload Error: {e}"
    )
    raise

# ==========================================
# VERIFY FILES
# ==========================================

try:

    files = api.list_repo_files(
        repo_id=repo_id,
        repo_type=repo_type
    )

    print("\nFiles Uploaded:")
    print(files)

except Exception as e:

    print(
        f"Verification Error: {e}"
    )
    raise

print("\nDataset Registration Completed Successfully")

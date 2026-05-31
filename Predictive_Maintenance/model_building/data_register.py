
from huggingface_hub import (
    HfApi,
    create_repo
)

from huggingface_hub.utils import (
    RepositoryNotFoundError
)

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

if TOKEN is None:

    raise Exception(

        "HF_TOKEN not found"

    )


# ==========================================
# HF API
# ==========================================

api = HfApi(

    token=TOKEN

)

print(

    "Current Directory:",

    os.getcwd()

)

print(

    "Data Path:",

    DATA_PATH

)

print(

    "Path Exists:",

    os.path.exists(DATA_PATH)

)


# ==========================================
# CHECK DATA FOLDER
# ==========================================

if not os.path.exists(DATA_PATH):

    raise Exception(

        f"Path not found: {DATA_PATH}"

    )


# ==========================================
# CREATE DATASET REPO
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

        private=False

    )

    print(

        "Dataset Repository Created"

    )


# ==========================================
# UPLOAD DATA FOLDER
# ==========================================

api.upload_folder(

    folder_path=DATA_PATH,

    repo_id=repo_id,

    repo_type=repo_type

)

print(

    "Folder Uploaded Successfully"

)


# ==========================================
# VERIFY FILES
# ==========================================

files = api.list_repo_files(

    repo_id=repo_id,

    repo_type=repo_type

)

print(

    "\nFiles Uploaded:"

)

print(

    files

)


from huggingface_hub import (
    login,
    upload_folder
)

import os

HF_TOKEN = os.getenv(
    "HF_TOKEN"
)

login(
    token=HF_TOKEN
)

upload_folder(

    repo_id="vyasmax9/predictive-maintenance",

    folder_path="Predictive_Maintenance",

    repo_type="space"

)

print(
    "Deployment Uploaded Successfully"
)

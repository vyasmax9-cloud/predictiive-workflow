import os
import mlflow

from huggingface_hub import (

    HfApi,

    create_repo

)

from huggingface_hub.utils import (

    RepositoryNotFoundError

)

# =====================================
# Configure MLflow
# =====================================

mlflow.set_tracking_uri(
    "file:./mlruns"
)

mlflow.set_experiment(
    "predictive-maintenance"
)

api = HfApi()

# =====================================
# Paths
# =====================================

MODEL_PATH = "Predictive_Maintenance/models/best_model.pkl"

repo_id = "vyasmax9/predictive-maintenance-model"

# =====================================
# Verify Model Exists
# =====================================

if not os.path.exists(
    MODEL_PATH
):

    raise FileNotFoundError(

        f"Model not found: {MODEL_PATH}"

    )

# =====================================
# Start Tracking
# =====================================

with mlflow.start_run():

    mlflow.log_param(

        "model_name",

        "best_model.pkl"

    )

    mlflow.log_artifact(

        MODEL_PATH,

        artifact_path="model"

    )

# =====================================
# Create Repo If Needed
# =====================================

try:

    api.repo_info(

        repo_id=repo_id,

        repo_type="model"

    )

    print(
        "Model Repository Exists"
    )

except RepositoryNotFoundError:

    create_repo(

        repo_id=repo_id,

        repo_type="model",

        private=False

    )

    print(
        "Created New Model Repository"
    )

# =====================================
# Upload Model
# =====================================

api.upload_file(

    path_or_fileobj=MODEL_PATH,

    path_in_repo="best_model.pkl",

    repo_id=repo_id,

    repo_type="model"

)

print(
    "Model Uploaded Successfully"
)

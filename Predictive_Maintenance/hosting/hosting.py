from huggingface_hub import login, upload_folder
import os

# ==========================================
# HF TOKEN
# ==========================================

HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise Exception("HF_TOKEN not found")

HF_TOKEN = HF_TOKEN.strip()

print("HF Token Loaded Successfully")

# ==========================================
# LOGIN
# ==========================================

login(token=HF_TOKEN)

# ==========================================
# UPLOAD TO HUGGING FACE SPACE
# ==========================================

upload_folder(
    repo_id="vyasmax9/predictive-maintenance",
    folder_path="Predictive_Maintenance",
    repo_type="space",
    token=HF_TOKEN
)

print("Deployment Uploaded Successfully")

import kagglehub
import shutil
import os

# 1. Download dataset (already extracted by KaggleHub)
path = kagglehub.dataset_download("kaustubhdikshit/neu-surface-defect-database")
print("✅ Dataset is available at:", path)

# 2. Copy dataset into your project folder ./dataset
project_dataset_path = "./dataset"
os.makedirs(project_dataset_path, exist_ok=True)

# Copy only once if not already copied
if not os.listdir(project_dataset_path):
    shutil.copytree(path, project_dataset_path, dirs_exist_ok=True)
    print("✅ Dataset copied into:", project_dataset_path)
else:
    print("ℹ️ Dataset already exists in:", project_dataset_path)

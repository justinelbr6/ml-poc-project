import os
import zipfile
import kagglehub

# We use kagglehub for simple downloading.
# Ensure you have kagglehub installed: pip install kagglehub

def download_dataset():
    # Set the destination path
    dest_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    os.makedirs(dest_dir, exist_ok=True)
    
    print("Downloading ritwikb3/heart-disease-cleveland from Kaggle...")
    # Download latest version
    path = kagglehub.dataset_download("ritwikb3/heart-disease-cleveland")

    print("Path to dataset files:", path)
    
    # Files are usually downloaded to a cache dir, so we copy them to our data folder
    import shutil
    for file_name in os.listdir(path):
        full_file_name = os.path.join(path, file_name)
        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, dest_dir)
            print(f"Copied {file_name} to {dest_dir}")

if __name__ == "__main__":
    download_dataset()
    print("Download complete!")

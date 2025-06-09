from huggingface_hub import snapshot_download
import shutil
import os

#TAMIA CLUSTER
#local_dir = "/scratch/d/dferbach/fineweb/"
#MILA CLUSTER
local_dir = "../../../../scratch/fineweb/"

# Remove directory if it exists
if os.path.exists(local_dir):
    shutil.rmtree(local_dir)

folder = snapshot_download(
            "HuggingFaceFW/fineweb", 
            repo_type="dataset",
            local_dir=local_dir,
            # replace "data/CC-MAIN-2023-50/*" with "sample/100BT/*" to use the 100BT sample
            allow_patterns="sample/10BT/*",
            max_workers=6,  # Use 6 parallel downloads
            local_files_only=False)

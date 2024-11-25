import os
import requests
import zipfile
import argparse

# Please replace the API Token below by visiting https://borealisdata.ca/ and getting your API Token from the top left menue.
# e.g. run command for patio scene: python download_clutter_dataset.py patio

# Configuration
API_TOKEN = "YOUR_BOREALIS_DATAVERSE_TOKEN"
BASE_URL = "https://borealisdata.ca"
DATASET_DOI = "doi:10.5683/SP3/WOFXFT"
HEADERS = {"X-Dataverse-key": API_TOKEN}
DOWNLOAD_ROOT = "./"

# Parse command line arguments for the scene name
parser = argparse.ArgumentParser(description="Download and unzip scene files from Dataverse")
parser.add_argument("scene", help="Name of the scene to download files from")
args = parser.parse_args()
SCENE = args.scene
DOWNLOAD_DIR = f"{SCENE}"  # Directory to save downloaded files

# Step 1: Get dataset metadata
dataset_url = f"{BASE_URL}/api/datasets/:persistentId?persistentId={DATASET_DOI}"
response = requests.get(dataset_url, headers=HEADERS)
dataset = response.json()

# Step 2: Filter files in the scene directory
files_to_download = [
    (f["dataFile"], DOWNLOAD_ROOT + f.get("directoryLabel"))
    for f in dataset["data"]["latestVersion"]["files"]
    if (DOWNLOAD_DIR in f.get("directoryLabel") and DOWNLOAD_DIR + '_' not in f.get("directoryLabel"))
]
print(f"Found {len(files_to_download)} files in the {DOWNLOAD_DIR} directory")

# Step 3: Download each file using its Persistent ID
for file_info in files_to_download:
    # Create the download directory if it doesn't exist
    if not os.path.exists(file_info[1]):
        os.makedirs(file_info[1])
    
    file_id = file_info[0]["id"]
    file_name = file_info[0]["filename"]
    
    # Construct the URL for downloading using Persistent ID
    download_url = f"{BASE_URL}/api/access/datafile/{file_id}/?persistentId={file_id}"
    
    # Construct local file path
    local_file_path = os.path.join(file_info[1], file_name)
    
    # Send GET request to download the file
    download_response = requests.get(download_url, headers=HEADERS, stream=True)
    
    if download_response.status_code == 200:
        with open(local_file_path, 'wb') as f:
            for chunk in download_response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        print(f"Downloaded file: {file_name}")
        
        # Step 4: If the file is a zip, unzip it and remove the zip file
        if file_name.endswith('.zip'):
            print(f"Unzipping {file_name}...")
            unzip_dir = os.path.join(file_info[1])  # Create a directory based on the zip file name
            
            if not os.path.exists(unzip_dir):
                os.makedirs(unzip_dir)
            
            # Unzip the file
            with zipfile.ZipFile(local_file_path, 'r') as zip_ref:
                zip_ref.extractall(unzip_dir)
            
            # Remove the zip file after extraction
            os.remove(local_file_path)
            print(f"Unzipped and removed {file_name}")
    
    else:
        print(f"Failed to download file {file_name} (Status: {download_response.status_code})")

        
    

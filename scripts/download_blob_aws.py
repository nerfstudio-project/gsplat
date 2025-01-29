from azure.storage.blob import BlobServiceClient
import os, shutil

# Don't clean directory this time since we want to keep existing files
local_path = os.path.expanduser("~/new_data")
if not os.path.exists(local_path):
    os.makedirs(local_path)

connection_string = f"DefaultEndpointsProtocol=https;AccountName=odyingest;AccountKey=;EndpointSuffix=core.windows.net"
container_name = "3droutput"

# Create the blob service client
blob_service_client = BlobServiceClient.from_connection_string(connection_string)
container_client = blob_service_client.get_container_client(container_name)

# Create ALL directories up front
directories = set()
for blob in container_client.list_blobs():
    dir_path = os.path.dirname(os.path.join(local_path, blob.name))
    directories.add(dir_path)

# Create all directory paths first
for dir_path in sorted(directories):
    os.makedirs(dir_path, exist_ok=True)
    print(f"Created directory: {dir_path}")

# Now download only files that don't exist
for blob in container_client.list_blobs():
    if blob.size > 0:
        file_path = os.path.join(local_path, blob.name)
        if not os.path.exists(file_path):  # Only download if file doesn't exist
            try:
                print(f"Downloading {blob.name}...")
                with open(file_path, "wb") as file:
                    data = container_client.download_blob(blob.name).readall()
                    file.write(data)
            except Exception as e:
                print(f"Error downloading {blob.name}: {str(e)}")
        else:
            print(f"Skipping {blob.name} - already exists")

print("Download complete!")

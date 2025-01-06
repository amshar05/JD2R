import os
import requests

# Disable SSL verification globally for this script
requests.packages.urllib3.disable_warnings()

# Base URL for the Hugging Face model
base_url = "https://huggingface.co/facebook/bart-large-cnn/resolve/main/"

# Files to download
files = [
    "tokenizer.json",
    "tokenizer_config.json"
]

# Directory to save the files
model_dir = "./models/facebook/bart-large-cnn"
os.makedirs(model_dir, exist_ok=True)

# Download each file
for file_name in files:
    url = base_url + file_name
    file_path = os.path.join(model_dir, file_name)
    print(f"Downloading {url} to {file_path}...")
    response = requests.get(url, verify=False)
    if response.status_code == 200:
        with open(file_path, "wb") as f:
            f.write(response.content)
        print(f"Saved {file_name}")
    else:
        print(f"Failed to download {file_name}: {response.status_code}")

print("All files downloaded.")

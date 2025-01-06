import requests
import os

# URL of the model/config file
url = "https://huggingface.co/sshleifer/distilbart-cnn-12-6/resolve/a4f8f3e/config.json"

# Output file path
output_path = "config.json"

# Disable SSL verification to download
response = requests.get(url, verify=False)

# Check if the download was successful
if response.status_code == 200:
    with open(output_path, 'wb') as f:
        f.write(response.content)
    print(f"File downloaded successfully: {output_path}")
else:
    print(f"Failed to download file. HTTP Status: {response.status_code}")

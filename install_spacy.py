import requests
import shutil
import os
import subprocess

# Disable SSL verification and download the model
url = 'https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.4.1/en_core_web_sm-3.4.1.tar.gz'
local_filename = 'en_core_web_sm-3.4.1.tar.gz'

response = requests.get(url, stream=True, verify=False)
with open(local_filename, 'wb') as file:
    shutil.copyfileobj(response.raw, file)

print(f"Downloaded {local_filename}")

# Install the model tarball as a Python package
subprocess.run(["pip", "install", local_filename])

# Clean up the downloaded file
os.remove(local_filename)

print("Model installed successfully.")

import os
import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# We will use Model ID 1 (Eunomia) from DAMIT
MODEL_ID = 1
ASTEROID_NAME = "15_Eunomia"
DATA_DIR = f"data/{ASTEROID_NAME}"
BASE_URL = f"https://astro.troja.mff.cuni.cz/projects/damit/asteroid_models/download/{MODEL_ID}"

os.makedirs(DATA_DIR, exist_ok=True)

print(f"Downloading real DAMIT data for {ASTEROID_NAME} (Model {MODEL_ID})...")

files_to_download = [
    ("shape.txt", "shape.txt"),
    ("spin.txt", "spin.txt"),
    ("lc.txt", "lc.txt")
]

for filename, save_name in files_to_download:
    url = f"{BASE_URL}/{filename}"
    print(f"Downloading {url}...")
    try:
        response = requests.get(url, verify=False)
        response.raise_for_status()
        save_path = os.path.join(DATA_DIR, save_name)
        with open(save_path, 'w') as f:
            f.write(response.text)
        print(f"  -> Saved to {save_path} ({len(response.text)} bytes)")
    except Exception as e:
        print(f"  -> Failed to download {filename}: {e}")

print("Done setting up real asteroid data.")

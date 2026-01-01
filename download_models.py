import os
import urllib.request

MODEL_URL = "https://github.com/Vreins/Facemask_recognition/releases/download/V1/convnext_tiny_mask.pth"
MODEL_PATH = "models/convnext_tiny_mask.pth"

os.makedirs("models", exist_ok=True)

if not os.path.exists(MODEL_PATH):
    print("⬇️ Downloading ConvNeXt model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("✅ Model downloaded")
else:
    print("✅ Model already exists")

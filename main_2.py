import os
os.environ["MPLCONFIGDIR"] = "/tmp"
import platform
import pathlib
import torch
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse
from fastai.vision.all import *
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from ultralytics import YOLO
import base64
from io import BytesIO
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import timm

from torchvision import transforms
import numpy as np
from PIL import Image

DEVICE = torch.device("cpu")

# ------------------------
# ------------------------
# Load models ONCE
# ------------------------
# model = YOLO("yolov8n-face.pt")
# learner = load_learner("models/new_model.pkl")

if platform.system() != "Windows":
    pathlib.WindowsPath = pathlib.PosixPath

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🔥 Lifespan starting")
    global learner, model
    model = YOLO("yolov8n-face.pt")  # face detector
    # PyTorch ConvNeXt Tiny for mask detection
    model_mask = timm.create_model("convnext_tiny", pretrained=False, num_classes=2)
    state = torch.load(
        "models/convnext_tiny_mask.pth",
        map_location="cpu"
    )
    
    model_mask.load_state_dict(state)
    model_mask.eval()
    print("🔥 Models loaded")
    learner = model_mask
    yield


app = FastAPI(title="Face Mask Detection API", lifespan=lifespan)

app.mount("/static", StaticFiles(directory="static"), name="static")

# ------------------------
# Helper functions
# ------------------------
val_tfms = transforms.Compose([
    transforms.Resize((128,128)),  # match training
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225])
])

def predict_mask_np(image_np: np.ndarray):
    image = Image.fromarray(image_np).convert("RGB")
    image = val_tfms(image).unsqueeze(0).to(DEVICE)  # add batch dim
    with torch.no_grad():
        outputs = learner(image)  # raw logits
        probs = torch.softmax(outputs, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
    return "Mask" if pred_class == 1 else "No Mask"  # adjust based on your label_map

def image_to_base64(img: Image.Image):
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()

def annotate_image(image: Image.Image):
    img_np = np.array(image)
    results = model.predict(img_np, imgsz=320, verbose=False)

    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)

    try:
        font_size = max(20, annotated.height // 25)
        font = ImageFont.truetype("arial.ttf", size=font_size)
    except:
        font = ImageFont.load_default()

    faces = []

    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, box)

            face = annotated.crop((x1, y1, x2, y2)).resize((224, 224))
            label = predict_mask_np(np.array(face))

            color = (0, 255, 0) if label == "Mask" else (255, 0, 0)

            draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
            text_x=x1+5
            text_y=y1+5

            bbox = draw.textbbox((0,0), label, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            draw.rectangle(
                [text_x-3, text_y-3, text_x + text_w + 3, text_y + text_h + 3],
                fill=(255, 255, 255, 200)
            )
            draw.text((text_x, text_y), label, fill=color, font=font)
            # draw.text((text_x, text_y), label, fill=color, font=font)

            # draw.text((x1, y1 - 10), label, fill=color, font=font)

            faces.append({
                "bbox": [x1, y1, x2, y2],
                "label": label
            })

    return annotated, faces

# ------------------------
# API (UNCHANGED)
# ------------------------
@app.post("/detect/")
async def detect_mask(file: UploadFile = File(...)):
    try:
        image = Image.open(file.file).convert("RGB")
        annotated, faces = annotate_image(image)
        return JSONResponse({
            "image": image_to_base64(annotated),
            "faces": faces,
            "num_faces": len(faces)
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# =====================================================
# FRONTEND PAGES (STREAMLIT-LIKE)
# =====================================================

# ------------------------
# HOME PAGE
# ------------------------
@app.get("/", response_class=HTMLResponse)
async def home():
    return HTMLResponse("""
<!DOCTYPE html>
<html>
<head>
    <title>Face Mask Detection</title>
    <link rel="stylesheet"
          href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
    <style>
        body {
            padding: 50px;
            font-family: Arial, sans-serif;
        }

        .big-title {
            font-size: 48px;
            font-weight: 800;
            margin-bottom: 20px;
        }

        .caption-box {
            font-size: 18px;
            line-height: 1.7;
        }

        .home-container {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 50px;
            align-items: stretch; /* 🔥 important */
        }

        .home-container > div {
            height: 100%;
            display: flex;
            flex-direction: column;
        }

         .home-left {
            justify-content: flex-start;
        }
        .home-image {
            display: flex;
            justify-content: center;
            align-items: center;
            max-height: 500px;
        }

        .home-image img {
            max-height: 100%;
            object-fit: contain;
        }


        .start-btn {
            margin-top: 30px;
        }
    </style>
</head>
<body>

<div class="big-title">
    Face Mask Detection & COVID Safety
</div>
<div class="home-container">
    <!-- LEFT COLUMN -->
    <div>
        
        <div class="caption-box">
            <b>
                1. Face masks play a crucial role in reducing the spread of COVID-19,
                especially in public and crowded areas.<br><br>

                2. Our AI automatically detects faces and identifies whether each person
                is wearing a mask properly.<br><br>

                3. Get instant visual feedback with labeled bounding boxes showing
                <i>Mask</i> or <i>No Mask</i> for every detected face.<br><br>
            </b>
        </div>

        <a href="/detect-page" class="btn btn-primary btn-lg start-btn">
            Start Detection →
        </a>
    </div>

    <!-- RIGHT COLUMN -->
    <div class="home-image">
        <img src="static/images/Frame_4.jpg" alt="Face Mask Detection">
    </div>

</div>

</body>
</html>
""")


# ------------------------
# DETECTION PAGE
# ------------------------
@app.get("/detect-page", response_class=HTMLResponse)
async def detect_page():
    return HTMLResponse("""
<!DOCTYPE html>
<html>
<head>
    <title>Detection</title>
    <link rel="stylesheet"
          href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
    <style>
        body { margin: 30px; font-family: Arial; }
        canvas { border: 1px solid #ccc; }
        .canvas-row { display:flex; gap:20px; margin-top:20px; }
    </style>
</head>
<body>

<a href="/" class="btn btn-outline-secondary mb-3">← Home</a>

<h2>Face Mask Detection</h2>

<button class="btn btn-primary" onclick="showTab('upload')">Upload</button>
<button class="btn btn-secondary" onclick="showTab('camera')">Camera</button>

<hr>

<!-- UPLOAD -->
<div id="uploadTab">
    <input type="file" id="imageUpload" accept="image/*">
    <button class="btn btn-success" onclick="detectUpload()">Detect</button>

    <div class="canvas-row">
        <canvas id="uploadOriginal" width="640" height="480"></canvas>
        <canvas id="uploadAnnotated" width="640" height="480"></canvas>
    </div>
</div>

<!-- CAMERA -->
<div id="cameraTab" style="display:none;">
    <video id="video" width="640" height="480" autoplay muted></video><br>
    <canvas id="cameraCanvas" width="640" height="480"></canvas>
</div>

<script>
let cameraInterval = null;
const input = document.getElementById("imageUpload");

// --------------------
// Tabs
// --------------------
function showTab(tab){
    document.getElementById("uploadTab").style.display = tab==="upload"?"block":"none";
    document.getElementById("cameraTab").style.display = tab==="camera"?"block":"none";
    if(tab==="camera") startCamera();
    else stopCamera();
}

// --------------------
// Upload preview
// --------------------
input.onchange = () => {
    const img = new Image();
    img.src = URL.createObjectURL(input.files[0]);
    img.onload = () => {
        const c = uploadOriginal.getContext("2d");
        c.clearRect(0,0,640,480);
        c.drawImage(img,0,0,640,480);
    }
};

async function detectUpload(){
    const fd = new FormData();
    fd.append("file", input.files[0]);

    const res = await fetch("/detect/", {method:"POST", body:fd});
    const data = await res.json();

    const img = new Image();
    img.src = "data:image/png;base64," + data.image;
    img.onload = () => {
        const c = uploadAnnotated.getContext("2d");
        c.clearRect(0,0,640,480);
        c.drawImage(img,0,0,640,480);
    }
}

// --------------------
// Camera
// --------------------
const video = document.getElementById("video");
const camCanvas = document.getElementById("cameraCanvas");
const camCtx = camCanvas.getContext("2d");

navigator.mediaDevices.getUserMedia({video:true})
.then(stream => video.srcObject = stream);

async function detectCamera(){
    const tmp = document.createElement("canvas");
    tmp.width = 320;
    tmp.height = 240;
    tmp.getContext("2d").drawImage(video,0,0,320,240);

    tmp.toBlob(async blob => {
        const fd = new FormData();
        fd.append("file", blob, "frame.png");

        const res = await fetch("/detect/", {method:"POST", body:fd});
        const data = await res.json();

        const img = new Image();
        img.src = "data:image/png;base64," + data.image;
        img.onload = () => {
            camCtx.clearRect(0,0,640,480);
            camCtx.drawImage(img,0,0,640,480);
        }
    });
}

function startCamera(){
    if(cameraInterval) return;
    cameraInterval = setInterval(detectCamera, 1000);
}

function stopCamera(){
    clearInterval(cameraInterval);
    cameraInterval = null;
}
</script>

</body>
</html>
""")

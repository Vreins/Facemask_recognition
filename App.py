from fastai.vision.all import *
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import streamlit as st
from streamlit_option_menu import option_menu
import os
import numpy as np
import tempfile
import pygame
from ultralytics import YOLO
# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------

import pathlib
if os.name == "posix":
    pathlib.WindowsPath = pathlib.PosixPath
elif os.name == "nt":
    pathlib.PosixPath = pathlib.WindowsPath

model = YOLO("yolov8n-face.pt")  # download yolov8 face model

new_learner = load_learner("models/model.pkl")  # load FastAI model

# ---------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------
def analyze_image_for_mask_temp(image_pil):
    """Predict Mask / No Mask using FastAI learner from a PIL Image"""
    try:
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".png")
        os.close(tmp_fd)
        image_pil.save(tmp_path)

        img_fastai = PILImage.create(tmp_path)
        pred_class, pred_idx, outputs = new_learner.predict(img_fastai)

        os.remove(tmp_path)

        if str(pred_class) == "0":
            label_str = "No Mask"
        elif str(pred_class) == "1":
            label_str = "Mask"
        else:
            label_str = str(pred_class)

        return label_str, outputs

    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None

def detect_faces_yolo(image_pil):
    """
    Detect faces using YOLOv8.
    Returns a list of bounding boxes: (x1, y1, x2, y2)
    """
    # Convert PIL to numpy
    img_np = np.array(image_pil)

    # Run YOLO model
    results = model.predict(img_np, imgsz=640, verbose=False)

    boxes = []
    for r in results:
        for box in r.boxes.xyxy.cpu().numpy():  # xyxy = [x1, y1, x2, y2]
            x1, y1, x2, y2 = box.astype(int)
            boxes.append((x1, y1, x2, y2))

    return boxes

# ---------------------------------------------------
# STREAMLIT PAGE CONFIGURATION
# ---------------------------------------------------
st.set_page_config(page_title="Face Mask Detection", page_icon="😷", layout="wide")

# CSS Styling
st.markdown("""
<style>
.big-title {font-size: 48px; font-weight: 800; color:#2ecc71; margin-bottom: -15px;}
.subtle-text {font-size: 18px; color: #666;}
.stButton>button {border-radius: 8px; font-size: 18px; padding: 10px 20px; background-color: #2ecc71 !important; color: white !important; border: none;}
.image-card {border: 1px solid #eee; padding: 15px; border-radius: 12px; background-color: #fafafa;}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# TOP MENU
# ---------------------------------------------------
selected = option_menu(
    menu_title=None,
    options=["Home", "Facemask Identification"],
    icons=["house-fill", "person-bounding-box"],
    default_index=0,
    orientation="horizontal"
)

# ---------------------------------------------------
# HOME PAGE
# ---------------------------------------------------
if selected == "Home":
    col1, col2 = st.columns([1, 1.2])
    with col1:
        st.markdown('<div class="big-title">Face Mask Detection & COVID Safety</div>', unsafe_allow_html=True)
        st.caption(
        """
        <b>
        1. Face masks play a crucial role in reducing the spread of COVID-19, especially in public and crowded areas.<br><br>
        2. Our AI automatically detect faces and identifies whether each person is wearing a mask properly.<br><br>
        3. Get instant visual feedback with labeled bounding boxes showing <i>Mask</i> or <i>No Mask</i> for every detected face.<br><br>
        </b>
        """,
        unsafe_allow_html=True
    )
    with col2:
        if os.path.exists('images/Frame_4.jpg'):
            image = Image.open('images/Frame_4.jpg').resize((600, 750))
            st.image(image)

# ---------------------------------------------------
# FACEMASK IDENTIFICATION PAGE
# ---------------------------------------------------
elif selected == "Facemask Identification":
    st.markdown("### 😷 Facemask Detection")
    st.write("Upload an image or use your camera. The AI will detect faces and classify each as 'Mask' or 'No Mask'.")

    upload_option = st.radio("Choose Image Source", ('Upload Image', 'Use Camera'), horizontal=True)
    source_img = None

    col_upload, col_preview = st.columns([1, 1])
    with col_upload:
        if upload_option == 'Upload Image':
            img_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
            if img_file:
                source_img = Image.open(img_file).convert("RGB")
        else:
            camera_file = st.camera_input("Camera Capture")
            if camera_file:
                source_img = Image.open(camera_file).convert("RGB")

    with col_preview:
        st.markdown("#### 🖼 Image Preview")
        st.markdown('<div class="image-card">', unsafe_allow_html=True)
        if source_img:
            st.image(source_img, width=350)
        else:
            st.write("No image selected yet.")
        st.markdown("</div>", unsafe_allow_html=True)

    # ---------------------------------------------------
    # MASK DETECTION LOGIC
    # ---------------------------------------------------
    if 'pygame_inited' not in st.session_state:
        pygame.mixer.init()
        st.session_state.pygame_inited = True

    if source_img:
        if st.button("😷 Detect Mask", width='stretch'):
            with st.spinner("Analyzing image... please wait..."):
                try:
                    # detection with yolo
                    faces = detect_faces_yolo(source_img)

                    if len(faces) == 0:
                        st.error("No faces detected. Try another image.")
                        pygame.mixer.music.stop()
                    else:
                        annotated_img = source_img.copy()
                        draw = ImageDraw.Draw(annotated_img)
                        no_mask_found = False

                        # Load Times New Roman font
                        try:
                            font_size = max(20, annotated_img.height // 25)
                            font = ImageFont.truetype("times.ttf", size=font_size)      
                        except:
                            font = ImageFont.load_default()

                        for (x1, y1, x2, y2) in faces:
                            face_crop = annotated_img.crop((x1, y1, x2, y2)).resize((224, 224))
                            label, _ = analyze_image_for_mask_temp(face_crop)
                            color = (0, 255, 0) if label == "Mask" else (255, 0, 0)

                            # Draw rectangle
                            
                            draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=3)
                            text_x = x1 + 5
                            text_y = y1 + 5
                            # Draw label inside box (top-left inside)
                            bbox = draw.textbbox((0,0), label, font=font)
                            text_w = bbox[2] - bbox[0]
                            text_h = bbox[3] - bbox[1]
                            draw.rectangle(
                                [text_x-3, text_y-3, text_x + text_w + 3, text_y + text_h + 3],
                                fill=(255, 255, 255, 200)
                            )
                            draw.text((text_x, text_y), label, fill=color, font=font)
                            # draw.text((text_x, text_y), label, fill=color, font=font)

                            if label == "No Mask":
                                no_mask_found = True

                        st.image(annotated_img, caption="Detected Faces with Mask/No Mask", width='stretch')

                        # Play or stop audio
                        if no_mask_found:
                            st.warning("At least one person is not wearing a mask! Please wear a mask 😷")
                            if not pygame.mixer.music.get_busy():
                                pygame.mixer.music.load("audio.mp3")
                                pygame.mixer.music.play(-1)  # Loop
                        else:
                            if pygame.mixer.music.get_busy():
                                pygame.mixer.music.stop()

                except Exception as e:
                    st.error(f"Error during mask detection: {e}")

    else:
        st.warning("Please upload or capture an image first.")

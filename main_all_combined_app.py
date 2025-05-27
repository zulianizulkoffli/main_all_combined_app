# -*- coding: utf-8 -*-
"""
Created on Tue May 27 04:36:09 2025

@author: zzulk
"""

import streamlit as st
import cv2
import torch
import pandas as pd
import numpy as np
from PIL import Image
import tempfile

st.set_page_config(page_title="All-in-One YOLOv5 App", layout="wide")
st.title("🔍 All-in-One YOLOv5 Object Detection App")

# Load YOLOv5 model
@st.cache_resource
def load_model():
    return torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)

model = load_model()

# Sidebar options
mode = st.sidebar.radio(
    "Select Detection Mode:",
    ("📷 Upload Image", "🎞️ Upload Video", "🎥 Live Webcam", "📱 Phone Camera (IP Stream)")
)

# -----------------------
# 📷 Upload Image
# -----------------------
if mode == "📷 Upload Image":
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        image = Image.open(uploaded_image).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        results = model(np.array(image))
        df = results.pandas().xyxy[0]

        for _, row in df.iterrows():
            st.write(f"{row['name']} - {row['confidence']:.2f}")

        st.image(np.squeeze(results.render()), caption="Detected Objects", use_column_width=True)

# -----------------------
# 🎞️ Upload Video
# -----------------------
elif mode == "🎞️ Upload Video":
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi"])
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()
        frame_idx = 0

        while cap.isOpened() and frame_idx < 100:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)
            df = results.pandas().xyxy[0]

            for _, row in df.iterrows():
                x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
                label = f"{row['name']} {row['confidence']:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            stframe.image(frame, channels="BGR", use_column_width=True)
            frame_idx += 1

        cap.release()
        st.success("✅ Video processing complete.")

# -----------------------
# 🎥 Webcam (Terminal use only)
# -----------------------
elif mode == "🎥 Live Webcam":
    st.warning("⚠️ To use webcam detection, run this app in your terminal using:\n`streamlit run main_app.py`")
    st.code("cv2.VideoCapture(0) only works in local runtime")

# -----------------------
# 📱 IP Stream from Phone
# -----------------------
elif mode == "📱 Phone Camera (IP Stream)":
    ip_url = st.text_input("Enter phone camera stream URL (e.g., http://192.168.x.x:8080/video):")
    if st.button("Start Detection") and ip_url:
        cap = cv2.VideoCapture(ip_url)
        stframe = st.empty()
        if not cap.isOpened():
            st.error("❌ Cannot connect to IP stream.")
        else:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                results = model(frame)
                df = results.pandas().xyxy[0]

                for _, row in df.iterrows():
                    x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
                    label = f"{row['name']} {row['confidence']:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                stframe.image(frame, channels="BGR", use_column_width=True)
            cap.release()
            st.success("✅ Stream ended.")

"""
Streamlit app: License Plate Detection using YOLOv8 + EasyOCR

Dependencies:
pip install streamlit opencv-python-headless numpy ultralytics easyocr

Notes:
- Works on CPU by default; GPU is used if available.
- Supports Image (.jpg, .png) and Video (.mp4, .avi, .mov, .mkv) uploads.
- Provides annotated output and OCR text downloads.
"""

import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
import tempfile, io, os, re, time, csv
from collections import defaultdict, deque
from typing import List, Tuple

# -------------------------
# Constants
# -------------------------
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.45
DEFAULT_LANGS = ['en']
DEFAULT_FRAME_SKIP = 1  # for video
PLATE_REGEX = re.compile(r"^[A-Z]{2}[0-9]{2}[A-Z]{3}$")

# -------------------------
# Helper Functions
# -------------------------

def load_yolo(model_path: str):
    if not os.path.exists(model_path):
        st.error(f"YOLO model not found: {model_path}")
        return None
    return YOLO(model_path)

def load_easyocr(lang_list: List[str]):
    return easyocr.Reader(lang_list, gpu=False)  # CPU fallback

def correct_plate_format(ocr_text: str) -> str:
    mapping_num_to_alpha = {"0": "O", "1": "I", "5": "S", "8": "B"}
    mapping_alpha_to_num = {"O": "0", "I": "1", "Z": "2", "S": "5", "B": "8"}
    ocr_text = ocr_text.upper().replace(" ", "")
    if len(ocr_text) != 7:
        return ""
    corrected = []
    for i, ch in enumerate(ocr_text):
        if i < 2 or i >= 4:  # alpha positions
            if ch.isdigit() and ch in mapping_num_to_alpha:
                corrected.append(mapping_num_to_alpha[ch])
            elif ch.isalpha():
                corrected.append(ch)
            else:
                return ""
        else:  # numeric positions
            if ch.isalpha() and ch in mapping_alpha_to_num:
                corrected.append(mapping_alpha_to_num[ch])
            elif ch.isdigit():
                corrected.append(ch)
            else:
                return ""
    return "".join(corrected)

def recognize_plate(plate_crop, reader):
    if plate_crop is None or plate_crop.size == 0:
        return ""
    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    plate_resized = cv2.resize(thresh, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    try:
        ocr_result = reader.readtext(
            plate_resized,
            detail=0,
            allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        )
        if len(ocr_result) > 0:
            candidate = correct_plate_format(ocr_result[0])
            if candidate and PLATE_REGEX.match(candidate):
                return candidate
    except Exception:
        return ""
    return ""

def annotate_image(image, boxes, texts, confidences):
    annotated = image.copy()
    for (x1, y1, x2, y2), text, conf in zip(boxes, texts, confidences):
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)
        label = f"Plate {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw, y1), (0,255,0), -1)
        cv2.putText(annotated, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
        if text:
            cv2.putText(annotated, text, (x1, y2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    return annotated

# -------------------------
# Image Detection
# -------------------------
def detect_plates_on_image(image, model, reader, conf_thresh=DEFAULT_CONF, iou_thresh=DEFAULT_IOU):
    results = model(image, conf=conf_thresh, iou=iou_thresh, verbose=False)[0]
    boxes = []
    texts = []
    confidences = []

    for box in results.boxes:
        conf = float(box.conf.cpu())
        if conf < conf_thresh:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().tolist())
        boxes.append((x1, y1, x2, y2))
        plate_crop = image[y1:y2, x1:x2]
        text = recognize_plate(plate_crop, reader)
        texts.append(text)
        confidences.append(conf)
    annotated = annotate_image(image, boxes, texts, confidences)
    return annotated, boxes, texts, confidences

# -------------------------
# Video Detection (preserve original logic)
# -------------------------
def process_video_streamlit(input_path_or_bytes, model, reader, conf_thresh=DEFAULT_CONF, iou_thresh=DEFAULT_IOU, frame_skip=1):
    plate_history = defaultdict(lambda: deque(maxlen=10))
    plate_final = {}

    def get_box_id(x1, y1, x2, y2):
        return f"{int(x1/10)}_{int(y1/10)}_{int(x2/10)}_{int(y2/10)}"

    def get_stable_plate(box_id, new_text):
        if new_text:
            plate_history[box_id].append(new_text)
            most_common = max(set(plate_history[box_id]), key=plate_history[box_id].count)
            plate_final[box_id] = most_common
        return plate_final.get(box_id, "")

    # Handle BytesIO or path
    if isinstance(input_path_or_bytes, io.BytesIO):
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(input_path_or_bytes.read())
        tfile.close()
        video_path = tfile.name
    else:
        video_path = input_path_or_bytes

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    temp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    out = cv2.VideoWriter(temp_out.name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    frame_idx = 0
    detections_list = []
    progress_bar = st.progress(0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx +=1
        if frame_idx % frame_skip != 0:
            out.write(frame)
            continue

        results = model(frame, verbose=False)[0]
        boxes_frame = []
        texts_frame = []
        confs_frame = []

        for box in results.boxes:
            conf = float(box.conf.cpu())
            if conf < conf_thresh:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().tolist())
            x1, x2 = max(0,x1), min(width-1,x2)
            y1, y2 = max(0,y1), min(height-1,y2)
            if x2<=x1 or y2<=y1:
                continue
            plate_crop = frame[y1:y2, x1:x2]
            text = recognize_plate(plate_crop, reader)
            box_id = get_box_id(x1, y1, x2, y2)
            stable_text = get_stable_plate(box_id, text)
            # draw rectangle
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            if stable_text:
                cv2.putText(frame, stable_text, (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2)
            boxes_frame.append((x1,y1,x2,y2))
            texts_frame.append(stable_text)
            confs_frame.append(conf)
            detections_list.append([frame_idx,x1,y1,x2,y2,conf,stable_text])

        out.write(frame)
        progress_bar.progress(min(frame_idx/total_frames,1.0))

    cap.release()
    out.release()
    return temp_out.name, detections_list

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="🚗 License Plate Detection", layout="wide")
st.title("🚗 License Plate Detection System")
st.markdown("Detect vehicle number plates and read text using YOLOv8 + EasyOCR")

with st.sidebar:
    st.header("⚙️ Settings")
    model_file = st.file_uploader("Upload YOLOv8 model (.pt)", type=["pt"])
    conf_thresh = st.slider("Confidence Threshold", 0.1, 0.9, DEFAULT_CONF, 0.05)
    iou_thresh = st.slider("NMS IoU Threshold", 0.1, 0.9, DEFAULT_IOU, 0.05)
    langs = st.multiselect("OCR Language(s)", options=['en','hi','fr','de','es'], default=DEFAULT_LANGS)
    mode = st.radio("Processing Mode", ["Image", "Video"])
    frame_skip = st.slider("Frame skip (video)",1,10,DEFAULT_FRAME_SKIP)

# Load models
if model_file:
    t_model = tempfile.NamedTemporaryFile(delete=False,suffix=".pt")
    t_model.write(model_file.read())
    t_model.close()
    yolo_model = load_yolo(t_model.name)
else:
    yolo_model = load_yolo("yolov8n.pt")  # default small model

ocr_reader = load_easyocr(langs)

# Upload media
if mode=="Image":
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg","jpeg","png"])
    if uploaded_file and yolo_model:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        annotated_img, boxes, texts, confs = detect_plates_on_image(image, yolo_model, ocr_reader, conf_thresh, iou_thresh)
        st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), caption="Annotated Image", use_column_width=True)
        if boxes:
            st.subheader("Detections")
            for i,(b,t,c) in enumerate(zip(boxes,texts,confs)):
                st.write(f"Plate {i+1}: OCR='{t}' | Confidence={c:.2f}")
            # Downloads
            img_out = cv2.imencode('.png', annotated_img)[1].tobytes()
            st.download_button("Download Annotated Image", img_out, file_name="annotated_image.png")
            plates_txt = "\n".join([t for t in texts if t])
            st.download_button("Download Plates Text", plates_txt, file_name="plates.txt")
        else:
            st.warning("No plates detected.")
elif mode=="Video":
    uploaded_file = st.file_uploader("Upload a Video", type=["mp4","avi","mov","mkv"])
    if uploaded_file and yolo_model:
        st.info("Processing video...")
        annotated_video_path, detections_list = process_video_streamlit(uploaded_file, yolo_model, ocr_reader, conf_thresh, iou_thresh, frame_skip)
        st.success("Video processed!")

        # Display small preview (first frame)
        cap = cv2.VideoCapture(annotated_video_path)
        ret, frame = cap.read()
        if ret:
            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Annotated Video Sample Frame", use_column_width=True)
        cap.release()

        # Downloads
        with open(annotated_video_path, "rb") as f:
            st.download_button("Download Annotated Video", f, file_name="annotated_video.mp4")

        # Save detections as CSV
        csv_out = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        with open(csv_out.name,"w",newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["frame","x1","y1","x2","y2","confidence","ocr_text"])
            writer.writerows(detections_list)
        with open(csv_out.name,"rb") as f:
            st.download_button("Download Detections CSV", f, file_name="detections.csv")

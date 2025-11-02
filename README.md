SmartVision: AI-Powered Vehicle License Recognition System

An end-to-end number-plate detection and OCR pipeline featuring YOLOv8 for plate detection and EasyOCR for text reading. Use it in two ways:

Colab script for batch video processing with advanced India/UK plate correction.

Streamlit app for quick image/video uploads with downloadable results.

✨ Features

YOLOv8-based plate detection with confidence/IoU controls. 

ocr

EasyOCR text extraction (auto-GPU in Colab, CPU fallback in Streamlit).

Locale-aware plate normalization

UK format AA99AAA validation.

India format fixer with state code snapping, digit/letter swaps (e.g., O↔0, I↔1), and slot-wise repair. 

app

OCR robustness via grayscale, CLAHE, upscaling, adaptive/OTSU thresholds; tries multiple variants. 

app

Temporal stabilization of plate text across frames (deque/majority vote).

Rich outputs: annotated image/video overlays, extracted plate text file, and CSV of detections (frame, box, confidence, OCR). 

ocr

📦 Installation
# Core
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip

# Dependencies
pip install ultralytics easyocr opencv-python-headless numpy streamlit


Colab users can also run the inline pip install lines present in the script comments. 

app

🚀 Quick Start
A) Streamlit App (upload image/video)
streamlit run ocr.py


Upload YOLOv8 .pt (or use default yolov8n.pt), choose Image or Video, adjust Confidence/IoU, pick OCR language(s), and process. 

ocr

Download: Annotated Image/Video and detections.csv from the UI. 

ocr

B) Colab Script (video → annotated video)

Upload your YOLO .pt and input .mp4 into /content.

Run app.py cells; it auto-picks the first .pt and .mp4 and writes output_with_license.mp4. 

app

🧠 How It Works

Detection (YOLOv8) — runs on each frame/image to produce bounding boxes. Thresholds configurable. 

ocr

Preprocess for OCR — grayscale → CLAHE (Colab) / OTSU (Streamlit) → 2–3× upscaling; tries multiple binary variants to improve OCR reliability. 

app

OCR (EasyOCR) — restricted allow-list A–Z, 0–9 to reduce noise.

Post-processing & Validation

UK: strict regex ^[A-Z]{2}\d{2}[A-Z]{3}$.

India: slot-wise repair (AA NN A{1..3} N{1..4}), state aliasing (e.g., OR→OD), and digit/letter confusion maps (O/0, I/1, S/5, B/8, etc.). 

app

Temporal Stabilization — per-box deque buffers last N reads and emits the majority label to avoid flicker.

Output Rendering — draws boxes + labels; Streamlit offers downloads (image/video/CSV). 

ocr

🗂️ Project Structure
.
├── app.py   # Colab-oriented batch video pipeline (UK + India normalization)  ← recommended for datasets
├── ocr.py   # Streamlit app for quick image/video processing & downloads
└── models/  # (optional) place your YOLOv8 .pt file(s) here

⚙️ Configuration Tips

Streamlit defaults: DEFAULT_CONF=0.25, DEFAULT_IOU=0.45, frame_skip=1. Tweak in sidebar for speed/accuracy trade-off. 

ocr

Model choice: start with yolov8n.pt for CPU; use your finetuned license-plate model for best results. 

ocr

OCR langs: select additional languages in the sidebar if your plates include non-Latin scripts. 

ocr

GPU: Colab script enables GPU for EasyOCR when available; Streamlit uses CPU by default.

📈 Performance Hints

Tighter boxes → better OCR: use a plate-specific YOLO model if possible.

Increase resolution: the pipeline already upscales crops; high-res sources still help. 

app

Adjust thresholds: raise confidence to reduce false positives; fine-tune IoU to control NMS. 

ocr

Leverage stabilization for videos; keep frame_skip low for fast motion. 

ocr

🧪 Sample Commands

Run with default YOLO:

streamlit run ocr.py


Run with your custom model:

Upload .pt via the Streamlit sidebar (or hardcode path in ocr.py if you prefer). 

ocr

Colab batch processing:

Put model.pt and clip.mp4 in /content, execute app.py, collect output_with_license.mp4. 

app

🛠️ Troubleshooting

“YOLO model not found” in Streamlit: upload a .pt file from the sidebar or ensure yolov8n.pt exists in the working directory. 

ocr

Blank OCR / low accuracy: ensure plate is readable; try different thresholds or increase video resolution; verify locale regex (UK) or use the Colab script for India formats.

Slow processing: increase frame_skip, use a lighter YOLO model, or enable GPU in Colab. 

ocr

📄 License

MIT (or your preferred license).

🙏 Acknowledgements

Ultralytics YOLOv8 for detection

EasyOCR for text recognition

OpenCV for preprocessing and rendering

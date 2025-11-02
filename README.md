# SmartVision: AI-Powered Vehicle License Recognition System

An advanced AI-based vehicle license plate recognition system combining **YOLOv8** for detection and **EasyOCR** for text recognition. SmartVision accurately detects, reads, and annotates license plates from **images and videos**, offering both a **Streamlit web interface** and a **Colab-compatible batch processor**.

---

## 🚀 Key Features

* **YOLOv8 Detection:** Detects vehicle license plates with adjustable confidence and IoU thresholds.
* **EasyOCR Integration:** Reads alphanumeric plate text with GPU acceleration (if available).
* **Country-Specific Support:**

  * UK format validation: `AA99AAA`
  * India format correction with **state code normalization**, **digit-letter confusion mapping**, and **slot-wise repair logic**.
* **Advanced OCR Preprocessing:** CLAHE contrast enhancement, grayscale, adaptive and OTSU thresholding, and upscaling for improved recognition accuracy.
* **Temporal Text Stabilization:** Uses a deque-based majority voting system to stabilize plate text across video frames.
* **Interactive Streamlit App:**

  * Upload images or videos
  * View annotated results instantly
  * Download **annotated media**, **CSV reports**, and **extracted text files**
* **Colab-Compatible Script:** Batch process videos with automatic input/output handling for quick experimentation.

---

## 🧩 System Workflow

1. **Detection:** YOLOv8 identifies bounding boxes for plates in each frame.
2. **Preprocessing:** Plates are cropped, enhanced, and thresholded for OCR.
3. **OCR (EasyOCR):** Extracts textual content from plate regions.
4. **Post-Processing:**

   * Applies regex-based format validation for UK/India.
   * Performs **keyword-based corrections** (e.g., 0↔O, 1↔I, 5↔S).
5. **Stabilization:** Smooths recognition across multiple frames.
6. **Output Generation:** Annotated visuals, logs, and CSV reports are saved.

---

## 🧠 Technologies Used

| Component                 | Description                                            |
| ------------------------- | ------------------------------------------------------ |
| **YOLOv8**                | Vehicle license plate detection                        |
| **EasyOCR**               | Optical Character Recognition engine                   |
| **OpenCV**                | Image preprocessing, visualization, and video handling |
| **Streamlit**             | Interactive UI for real-time image/video testing       |
| **Python**                | Core programming language                              |
| **Regex & State Mapping** | Country-specific plate format correction               |

---

## 🧰 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/SmartVision.git
cd SmartVision

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

Alternatively, install manually:

```bash
pip install ultralytics easyocr opencv-python-headless numpy streamlit
```

---

## 💻 Usage

### Option 1: Streamlit Interface

```bash
streamlit run ocr.py
```

**Steps:**

1. Upload your **YOLOv8 model (.pt)** or use the default (`yolov8n.pt`).
2. Choose processing mode — **Image** or **Video**.
3. Adjust **confidence**, **IoU**, and **OCR language** in the sidebar.
4. Process, preview, and download annotated outputs (image/video/CSV).

### Option 2: Colab Script

1. Upload your YOLO `.pt` model and `.mp4` video file into `/content/`.
2. Run `app.py`.
3. Annotated video output will be saved as `output_with_license.mp4`.

---

## ⚙️ Configuration Options

| Parameter     | Description                         | Default |
| ------------- | ----------------------------------- | ------- |
| `CONF_THRESH` | YOLO confidence threshold           | 0.25    |
| `IOU_THRESH`  | Non-max suppression IoU threshold   | 0.45    |
| `FRAME_SKIP`  | Skip frames during video processing | 1       |
| `OCR_LANGS`   | OCR language list                   | ['en']  |

---

## 🧩 Project Structure

```
SmartVision/
├── app.py           # Colab-compatible batch processing pipeline
├── ocr.py           # Streamlit interface for image/video uploads
├── requirements.txt # Dependencies
└── models/           # Optional directory for YOLOv8 models
```

---

## 📊 Example Outputs

| Mode  | Input                  | Output                                     |
| ----- | ---------------------- | ------------------------------------------ |
| Image | Upload a car photo     | Annotated image with detected plate & text |
| Video | Upload dashcam footage | Annotated video with stabilized OCR text   |

---

## 🧠 Performance Tips

* Use **high-resolution inputs** for better OCR results.
* Optimize frame skipping to balance speed and accuracy.
* Tune YOLO confidence/IoU thresholds to reduce false positives.
* For Indian plates, prefer the **Colab script** with advanced slot-based correction.

---

## 🧪 Example Datasets

You can train or fine-tune YOLO models using:

* **OpenALPR Benchmark Dataset**
* **Indian License Plate Dataset (Kaggle)**
* **UFPR-ALPR Dataset**

---

## 🧰 Future Enhancements

* Add **multi-language OCR** for non-Latin scripts (Hindi, Arabic, etc.)
* Integrate **vehicle make/model recognition**
* Deploy a **web dashboard** for real-time monitoring
* Add **cloud deployment** support (Docker + Streamlit Cloud)

---

## 🧾 License

This project is released under the **MIT License**.

---

## 🙌 Acknowledgements

* [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
* [EasyOCR](https://github.com/JaidedAI/EasyOCR)
* [OpenCV](https://opencv.org/)
* [Streamlit](https://streamlit.io/)

---



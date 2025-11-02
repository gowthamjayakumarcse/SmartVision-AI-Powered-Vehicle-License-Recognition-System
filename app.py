# ==== Dependencies (Colab) ====
# !pip install ultralytics easyocr opencv-python-headless==4.10.0.84 --quiet

import os, re, cv2, glob, torch
import numpy as np
from ultralytics import YOLO
import easyocr
from collections import defaultdict, deque


# ================== CONFIG (no hard-coded OS paths) ==================
# Put your model (.pt) and video in /content (upload via Colab file browser or files.upload)
MODEL_PATH  = next(iter(glob.glob("/content/*.pt")),  None)  # auto-pick first .pt in /content
INPUT_VIDEO = next(iter(glob.glob("/content/*.mp4")), None)  # auto-pick first .mp4 in /content
OUTPUT_VIDEO = "/content/output_with_license.mp4"

assert MODEL_PATH is not None,  "Place your YOLO model .pt file in /content"
assert INPUT_VIDEO is not None, "Place an input .mp4 video in /content"

print("Using MODEL_PATH :", MODEL_PATH)
print("Using INPUT_VIDEO:", INPUT_VIDEO)

# ================== MODELS ==================
reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
model  = YOLO(MODEL_PATH)

# ================== PLATE HELPERS ==================
# UK: GX15OGU; India: MH20DV2366 / KA01AB1234
PLATE_REGEX_UK = re.compile(r"^[A-Z]{2}\d{2}[A-Z]{3}$")
PLATE_REGEX_IN = re.compile(r"^[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{1,4}$")

INDIA_STATES = {
    "AP","AR","AS","BR","CG","CH","DD","DL","DN","GA","GJ","HP","HR","JH","JK","KA","KL",
    "LA","LD","MH","ML","MN","MP","MZ","NL","OD","OR","PB","RJ","SK","TN","TS","TR","UK","UP","WB"
}
ALIASES = {"OR": "OD", "UA": "UK"}  # map old → current

A2N = str.maketrans({"O":"0","I":"1","L":"1","Z":"2","S":"5","B":"8","G":"6","T":"7"})
N2A = str.maketrans({"0":"O","1":"I","2":"Z","5":"S","6":"G","7":"T","8":"B"})

def closest_indian_state(code2: str) -> str:
    code2 = (code2 or "")[:2].upper()
    if code2 in INDIA_STATES: return ALIASES.get(code2, code2)
    if code2 in ALIASES:      return ALIASES[code2]
    best, best_d = None, 3
    for s in INDIA_STATES:
        d = sum(a != b for a,b in zip(code2, s))
        if d < best_d: best, best_d = s, d
    return best or code2

def fix_slots_india(raw: str) -> str:
    """
    Coerce OCR to India pattern: AA NN A{1..3} N{1..4}
    Slot-wise letter/digit corrections + state snapping + allow dropping 1 stray char.
    """
    s = re.sub(r"[^A-Z0-9]", "", raw.upper())
    if not (6 <= len(s) <= 11): return ""

    def try_parse(t: str) -> str:
        for dist_len in (1,2):
            for series_len in (1,2,3):
                for num_len in (1,2,3,4):
                    total = 2 + dist_len + series_len + num_len
                    if len(t) != total: continue
                    a  = t[0:2].translate(N2A)                    # letters
                    n1 = t[2:2+dist_len].translate(A2N)           # digits
                    a2 = t[2+dist_len:2+dist_len+series_len].translate(N2A)  # letters
                    n2 = t[-num_len:].translate(A2N)              # digits
                    a = closest_indian_state(a)
                    cand = f"{a}{n1}{a2}{n2}"
                    if PLATE_REGEX_IN.match(cand): return cand
        return ""

    # as-is
    cand = try_parse(s)
    if cand: return cand
    # whole-string ambiguity
    for v in {s.translate(A2N), s.translate(N2A)}:
        cand = try_parse(v)
        if cand: return cand
    # drop one stray char
    for i in range(len(s)):
        cand = try_parse(s[:i] + s[i+1:])
        if cand: return cand
    return ""

def join_ocr_fragments(result_detail):
    """Sort fragments left→right and join text."""
    parts = []
    for (bbox, txt, conf) in result_detail:
        xs = [p[0] for p in bbox]
        parts.append((sum(xs)/4.0, txt))
    parts.sort(key=lambda t: t[0])
    joined = "".join(p[1] for p in parts)
    return re.sub(r"[^A-Z0-9]", "", joined.upper())

def recognize_plate(plate_crop: np.ndarray) -> str:
    if plate_crop is None or plate_crop.size == 0: return ""

    # pad crop ~20% to avoid cutting characters
    h, w = plate_crop.shape[:2]
    pad = int(0.2 * max(h, w))
    y1, y2 = max(0, -pad), min(h+pad, h)  # padding after copy border
    x1, x2 = max(0, -pad), min(w+pad, w)
    crop = cv2.copyMakeBorder(plate_crop, pad, pad, pad, pad, cv2.BORDER_REPLICATE)

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(2.0, (8,8)).apply(gray)

    # upscale for OCR clarity
    fh, fw = gray.shape
    factor = max(3, 600 // max(1, fw))
    gray = cv2.resize(gray, (fw*factor, fh*factor), interpolation=cv2.INTER_CUBIC)

    variants = []
    try:
        v1 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
        variants += [v1, 255-v1]
    except: pass
    try:
        _, v2 = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        variants += [v2, 255-v2]
    except: pass
    variants.append(gray)

    for img in variants:
        try:
            res = reader.readtext(img, detail=1, paragraph=False,
                                  allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
            if not res: continue
            joined = join_ocr_fragments(res)

            # UK direct
            for cand_raw in [joined] + [joined.translate(A2N)] + [joined.translate(N2A)]:
                if PLATE_REGEX_UK.match(cand_raw): return cand_raw

            # India fixer
            fixed = fix_slots_india(joined)
            if fixed: return fixed
        except: 
            continue
    return ""

# ================== STABILIZATION ==================
plate_history = defaultdict(lambda: deque(maxlen=10))
plate_final = {}
def get_box_id(x1,y1,x2,y2): return f"{int(x1/10)}_{int(y1/10)}_{int(x2/10)}_{int(y2/10)}"
def get_stable_plate(box_id, new_text):
    if new_text:
        plate_history[box_id].append(new_text)
        best = max(set(plate_history[box_id]), key=plate_history[box_id].count)
        plate_final[box_id] = best
    return plate_final.get(box_id, "")

# ================== VIDEO IO ==================
cap = cv2.VideoCapture(INPUT_VIDEO)
assert cap.isOpened(), f"Cannot open video: {INPUT_VIDEO}"

fps = max(1, cap.get(cv2.CAP_PROP_FPS))
W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (W, H))

CONF_THRESH = 0.30
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret: break
    results = model(frame, verbose=False)[0]

    for box in results.boxes:
        conf = float(box.conf.cpu().item())
        if conf < CONF_THRESH: continue

        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().tolist())
        x1, x2 = max(0, min(x1, W-1)), max(0, min(x2, W-1))
        y1, y2 = max(0, min(y1, H-1)), max(0, min(y2, H-1))
        if x2 <= x1 or y2 <= y1: continue

        plate_crop = frame[y1:y2, x1:x2]
        text = recognize_plate(plate_crop)
        stable = get_stable_plate(get_box_id(x1,y1,x2,y2), text)

        cv2.rectangle(frame, (x1,y1),(x2,y2), (0,255,0), 3)
        label = stable or text
        if label:
            # draw label with black background for readability
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            px, py = x1, max(25, y1-10)
            cv2.rectangle(frame, (px-6, py-th-10), (px+tw+6, py+6), (0,0,0), -1)
            cv2.putText(frame, label, (px, py), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

            # optional zoom overlay above the plate
            oh, ow = 150, 400
            zoom = cv2.resize(plate_crop, (ow, oh))
            oy1, ox1 = max(0, y1 - oh - 40), x1
            oy2, ox2 = oy1 + oh, ox1 + ow
            if oy2 <= H and ox2 <= W:
                frame[oy1:oy2, ox1:ox2] = zoom

    out.write(frame)

    # Show every Nth frame in Colab to avoid spamming the output
    if frame_idx % max(1, int(fps)) == 0:
        cv2_imshow(frame)
    frame_idx += 1

cap.release()
out.release()
cv2.destroyAllWindows()
print("✅ Annotated video saved to:", OUTPUT_VIDEO)

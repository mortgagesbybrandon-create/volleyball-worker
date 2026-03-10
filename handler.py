import runpod
import cv2
import numpy as np
import base64
from ultralytics import YOLO
from PIL import Image
import io
import easyocr

print("Loading YOLO model...")
yolo_model = YOLO("yolov8m.pt")
print("Loading OCR reader...")
ocr_reader = easyocr.Reader(["en"], gpu=True)
print("Models loaded!")


def decode_frame(data_url):
    if "," in data_url:
        data_url = data_url.split(",")[1]
    img_bytes = base64.b64decode(data_url)
    img = Image.open(io.BytesIO(img_bytes))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def detect_players(frame):
    results = yolo_model(frame, classes=[0], conf=0.3, verbose=False)
    detections = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0])
            detections.append({
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "confidence": round(conf, 3),
            })
    return detections


def read_jersey_numbers(frame, detections):
    results = []
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        h = y2 - y1
        jersey_y1 = y1 + int(h * 0.1)
        jersey_y2 = y1 + int(h * 0.55)
        crop = frame[jersey_y1:jersey_y2, x1:x2]

        if crop.size == 0:
            results.append({**det, "jersey_number": None, "ocr_confidence": 0})
            continue

        ocr_results = ocr_reader.readtext(crop, allowlist="0123456789")
        best_number = None
        best_conf = 0
        for (_, text, conf) in ocr_results:
            text = text.strip()
            if text.isdigit() and 0 < int(text) <= 99:
                if conf > best_conf:
                    best_number = int(text)
                    best_conf = float(conf)

        results.append({
            **det,
            "jersey_number": best_number,
            "ocr_confidence": round(best_conf, 3),
        })
    return results


def analyze_motion(frames_data):
    if len(frames_data) < 2:
        return []
    motion_scores = []
    prev_gray = None
    for fd in frames_data:
        frame = decode_frame(fd["dataUrl"])
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (320, 180))
        if prev_gray is not None:
            diff = cv2.absdiff(prev_gray, gray)
            score = float(np.mean(diff)) / 255.0
            motion_scores.append({
                "timestamp": fd["timestamp"],
                "motion_score": round(score, 4),
            })
        prev_gray = gray
    return motion_scores


def handler(job):
    input_data = job["input"]
    action = input_data.get("action", "detect")

    if action == "detect":
        frames = input_data.get("frames", [])
        all_results = []
        for frame_data in frames:
            frame = decode_frame(frame_data["dataUrl"])
            detections = detect_players(frame)
            players = read_jersey_numbers(frame, detections)
            all_results.append({
                "timestamp": frame_data["timestamp"],
                "players": players,
                "player_count": len(players),
            })
        return {"action": "detect", "frames": all_results}

    elif action == "triage":
        frames = input_data.get("frames", [])
        motion = analyze_motion(frames)
        activity_scores = []
        for frame_data in frames:
            frame = decode_frame(frame_data["dataUrl"])
            detections = detect_players(frame)
            activity_scores.append({
                "timestamp": frame_data["timestamp"],
                "person_count": len(detections),
                "has_activity": len(detections) >= 4,
            })
        return {"action": "triage", "motion": motion, "activity": activity_scores}

    else:
        return {"error": f"Unknown action: {action}"}


runpod.serverless.start({"handler": handler})

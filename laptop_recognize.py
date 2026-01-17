#!/usr/bin/env python3
"""
Laptop-Optimized Face Recognition (Ryzen 5 5600U + Vega 7)
==========================================================
Optimizations for x86_64:
1. Full resolution detection (640x480) for maximum accuracy.
2. Prioritizes ResNet-50 (w600k_r50) over MobileFaceNet.
3. Uses DirectML (GPU) if available, falling back to AVX2 CPU instructions.
4. No sensor dependencies.
"""
import os
import time
import threading
from pathlib import Path
from collections import deque

import cv2
import numpy as np

# Try ONNX Runtime with GPU support
try:
    import onnxruntime as ort
    USE_ORT = True
except ImportError:
    USE_ORT = False
    print("Warning: onnxruntime not found. Using cv2.dnn (Slower, CPU only).")
    print("Install: pip install onnxruntime-directml")

# ================= CONFIG =================
USERS_DIR = Path("data/users")

# Laptops have better cameras and CPUs, so we can be stricter
THRESH = 0.70               # Higher confidence threshold (ResNet is more confident)
VOTE_WINDOW = 5             # Faster reaction time (less smoothing needed)
VOTE_PASS = 3               

# Resolution
CAM_W, CAM_H = 640, 480
DETECT_W, DETECT_H = 640, 480  # Full resolution detection

# Process every frame (Ryzen 5600U can handle 30 FPS easily)
SKIP_FRAMES = 0             

# Model Paths
YUNET_MODEL = "models/face_detection_yunet_2023mar.onnx"

# We prefer the larger ResNet-50 model for laptops
PREFERRED_MODELS = [
    "models/buffalo_l/w600k_r50.onnx",   # Best Accuracy (ResNet-50)
    "models/w600k_r50.onnx",             # ResNet-50 (Alt path)
    "models/buffalo_sc/w600k_mbf.onnx",  # Fast (MobileFaceNet)
    "models/w600k_mbf.onnx"              # Original Pi Model
]

ARCFACE_REF = np.array([
    [38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
    [41.5493, 92.3655], [70.7299, 92.2041]
], dtype=np.float32)

def l2_normalize(v, eps=1e-10):
    return v / (np.linalg.norm(v) + eps)

def align_face_112(frame, lm5):
    M, _ = cv2.estimateAffinePartial2D(lm5, ARCFACE_REF, method=cv2.LMEDS)
    if M is None: return None
    return cv2.warpAffine(frame, M, (112, 112), flags=cv2.INTER_LINEAR)

class ONNXEmbedder:
    def __init__(self, model_path):
        # Prioritize Vega 7 GPU (DirectML) -> CPU
        providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
        
        try:
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session = ort.InferenceSession(model_path, sess_options=opts, providers=providers)
        except Exception as e:
            print(f"GPU Provider failed, falling back to CPU: {e}")
            self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

        self.input_name = self.session.get_inputs()[0].name
        print(f"Device used: {self.session.get_providers()[0]}")
    
    def embed(self, face_aligned):
        face_rgb = cv2.cvtColor(face_aligned, cv2.COLOR_BGR2RGB)
        blob = (face_rgb.astype(np.float32) - 127.5) / 128.0
        blob = blob.transpose(2, 0, 1)[np.newaxis, ...]
        out = self.session.run(None, {self.input_name: blob})[0]
        return l2_normalize(out.flatten())

class FrameGrabber:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        # Try DirectShow (Windows) or V4L2 (Linux) for faster init
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0, cv2.CAP_MSMF) 
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.lock = threading.Lock()
        self.frame = None
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.running:
            ok, f = self.cap.read()
            if ok:
                with self.lock:
                    self.frame = f
    
    def read(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def release(self):
        self.running = False
        self.cap.release()

def load_users():
    users = {}
    if not USERS_DIR.exists(): return users
    for d in USERS_DIR.iterdir():
        if d.is_dir() and (d / "embeddings.npy").exists():
            users[d.name] = np.load(d / "embeddings.npy").astype(np.float32)
            print(f"  Loaded {d.name}: {users[d.name].shape[0]} vectors")
    return users

def find_best_model():
    for p in PREFERRED_MODELS:
        if Path(p).exists(): return p
    return None

def main():
    print(f"=== Laptop Face Auth (Ryzen 5600U) ===")
    
    embed_path = find_best_model()
    if not embed_path:
        print("ERROR: No model found.")
        print("Please download 'buffalo_l' (ResNet50) for best accuracy:")
        print("1. Download: https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip")
        print("2. Unzip to: models/buffalo_l/")
        return
    print(f"Model: {embed_path}")
    
    users = load_users()
    if not users:
        print("No users found! Run enroll.py first.")
        return

    detector = cv2.FaceDetectorYN.create(
        YUNET_MODEL, "", (DETECT_W, DETECT_H),
        score_threshold=0.7, nms_threshold=0.3, top_k=1
    )
    
    if USE_ORT:
        embedder = ONNXEmbedder(embed_path)
    else:
        # Fallback to CV2 (not recommended for ResNet50 on CPU)
        print("Using OpenCV DNN (Legacy)")
        net = cv2.dnn.readNetFromONNX(embed_path)
        embedder = lambda img: l2_normalize(net.forward(net.setInput(cv2.dnn.blobFromImage(img, 1/128.0, (112,112), (127.5,127.5,127.5), swapRB=True))).flatten())

    grab = FrameGrabber()
    votes = deque(maxlen=VOTE_WINDOW)
    label, score = "UNKNOWN", 0.0

    print("Running... Press Q to quit")
    
    while True:
        frame = grab.read()
        if frame is None:
            time.sleep(0.01)
            continue
            
        # Detect
        detector.setInputSize((frame.shape[1], frame.shape[0]))
        _, faces = detector.detect(frame)
        
        if faces is not None and len(faces) > 0:
            f = faces[0]
            bbox = [int(x) for x in f[0:4]]
            lm = f[4:14].reshape(5, 2)
            
            # Align & Embed
            aligned = align_face_112(frame, lm)
            if aligned is not None:
                if hasattr(embedder, 'embed'): vec = embedder.embed(aligned)
                else: vec = embedder(aligned)
                
                # Match
                best_user, best_sim = "UNKNOWN", 0.0
                for name, templates in users.items():
                    sim = (templates @ vec).max()
                    if sim > best_sim:
                        best_sim = sim
                        best_user = name
                
                # Vote
                votes.append(1 if best_sim > THRESH else 0)
                if sum(votes) >= VOTE_PASS:
                    label = best_user
                    score = best_sim
                else:
                    label = "UNKNOWN"
                
                # Draw
                color = (0, 255, 0) if label != "UNKNOWN" else (0, 0, 255)
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), color, 2)
                cv2.putText(frame, f"{label} {score:.2f}", (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        else:
            votes.clear()
            label = "UNKNOWN"

        cv2.imshow("Laptop Auth", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    grab.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

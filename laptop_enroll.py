#!/usr/bin/env python3
"""
Laptop-Optimized Face Enrollment (Ryzen 5 5600U + Vega 7)
=========================================================
1. Matches resolution of recognize.py (640x480) for consistent accuracy.
2. Uses DirectML (GPU) for faster processing.
3. CRITICAL: Uses the exact same model (ResNet-50) as recognize.py.
"""
import time
import threading
from pathlib import Path
import cv2
import numpy as np

try:
    import onnxruntime as ort
    USE_ORT = True
except ImportError:
    USE_ORT = False

# ================= CONFIG =================
BASE_DIR = Path("data/users")
TOTAL_SAMPLES = 40
STARE_DURATION = 4.0        
STARE_SAMPLES = 15          
BLUR_THRESHOLD = 50.0       # Increased for Webcam (usually sharper than Pi Cam)

# MATCH THE RECOGNIZE SCRIPT RESOLUTION
CAM_W, CAM_H = 640, 480
DETECT_W, DETECT_H = 640, 480 

YUNET_MODEL = "models/face_detection_yunet_2023mar.onnx"

# EXACT SAME PREFERENCE ORDER AS RECOGNIZE.PY
PREFERRED_MODELS = [
    "models/buffalo_l/w600k_r50.onnx",   # Best Accuracy (ResNet-50)
    "models/w600k_r50.onnx",             
    "models/buffalo_sc/w600k_mbf.onnx",  
    "models/w600k_mbf.onnx"              
]
# ==========================================

ARCFACE_REF = np.array([
    [38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
    [41.5493, 92.3655], [70.7299, 92.2041]
], dtype=np.float32)

def l2_normalize(v, eps=1e-10):
    return v / (np.linalg.norm(v) + eps)

def is_blurry(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(g, cv2.CV_64F).var() < BLUR_THRESHOLD

def align_face_112(frame, lm5):
    M, _ = cv2.estimateAffinePartial2D(lm5, ARCFACE_REF, method=cv2.LMEDS)
    if M is None: return None
    return cv2.warpAffine(frame, M, (112, 112))

class Embedder:
    def __init__(self, model_path):
        if USE_ORT:
            # GPU Support (DirectML)
            providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
            try:
                opts = ort.SessionOptions()
                opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                self.session = ort.InferenceSession(model_path, sess_options=opts, providers=providers)
            except Exception as e:
                print(f"GPU failed, using CPU: {e}")
                self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
                
            self.input_name = self.session.get_inputs()[0].name
            self.use_ort = True
            print(f"  Embedder Engine: {self.session.get_providers()[0]}")
        else:
            self.net = cv2.dnn.readNetFromONNX(model_path)
            self.use_ort = False
    
    def embed(self, face):
        if self.use_ort:
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            blob = (face_rgb.astype(np.float32) - 127.5) / 128.0
            blob = blob.transpose(2, 0, 1)[np.newaxis, ...]
            out = self.session.run(None, {self.input_name: blob})[0]
        else:
            blob = cv2.dnn.blobFromImage(face, 1/128.0, (112,112), (127.5,127.5,127.5), swapRB=True)
            self.net.setInput(blob)
            out = self.net.forward()
        return l2_normalize(out.flatten())

class FrameGrabber:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        # Try DirectShow (Windows) for faster startup
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

def find_embedding_model():
    for p in PREFERRED_MODELS:
        if Path(p).exists(): return p
    return None

def main():
    print("=== Laptop Face Enrollment ===")
    
    embed_path = find_embedding_model()
    if not embed_path:
        print("ERROR: No embedding model found!")
        print("Please ensure you downloaded 'buffalo_l' as per previous instructions.")
        return
    print(f"Using Model: {embed_path}")
    
    name = input("Enter username: ").strip()
    if not name: return
    
    user_dir = BASE_DIR / name
    user_dir.mkdir(parents=True, exist_ok=True)
    
    # High Res Detector
    detector = cv2.FaceDetectorYN.create(
        YUNET_MODEL, "", (DETECT_W, DETECT_H),
        score_threshold=0.6, nms_threshold=0.3, top_k=1
    )
    embedder = Embedder(embed_path)
    
    grab = FrameGrabber()
    embeddings = []
    
    # State
    last_bbox = None
    status = "Get ready..."
    status_color = (255, 255, 255)
    
    # Phases
    phase = "countdown" 
    countdown_start = None
    stare_start = None
    
    print(f"\nEnrollment: {TOTAL_SAMPLES} samples")
    print(f"  Phase 1: Stare at camera ({STARE_DURATION}s)")
    print(f"  Phase 2: Rotate head slowly")
    print("\nPress ESC to cancel\n")
    
    time.sleep(0.5)
    countdown_start = time.time()
    
    while len(embeddings) < TOTAL_SAMPLES:
        frame = grab.read()
        if frame is None:
            time.sleep(0.01)
            continue
        
        now = time.time()
        
        # Phase Logic
        if phase == "countdown":
            elapsed = now - countdown_start
            remaining = 3 - int(elapsed)
            if remaining > 0:
                status = f"Starting in {remaining}..."
                status_color = (0, 255, 255)
            else:
                phase = "stare"
                stare_start = now
                status = "Look straight at camera"
                status_color = (0, 255, 0)
        
        elif phase == "stare":
            elapsed = now - stare_start
            remaining = STARE_DURATION - elapsed
            if remaining > 0:
                status = f"Hold still... {remaining:.1f}s"
                status_color = (0, 255, 0)
            else:
                phase = "rotate"
                status = "Now rotate head slowly"
                status_color = (255, 200, 0)
        
        elif phase == "rotate":
            remaining = TOTAL_SAMPLES - len(embeddings)
            status = f"Keep rotating... {remaining} left"
            status_color = (255, 200, 0)
        
        # Detection
        if phase in ("stare", "rotate"):
            small = cv2.resize(frame, (DETECT_W, DETECT_H))
            detector.setInputSize((DETECT_W, DETECT_H))
            _, faces = detector.detect(small)
            
            if faces is not None and len(faces) > 0:
                f = faces[0]
                
                # Scale landmarks
                lm = f[4:14].reshape(5, 2)
                lm[:, 0] *= CAM_W / DETECT_W
                lm[:, 1] *= CAM_H / DETECT_H
                
                # Bbox for display
                sx, sy = CAM_W / DETECT_W, CAM_H / DETECT_H
                last_bbox = (
                    int(f[0] * sx), int(f[1] * sy),
                    int((f[0] + f[2]) * sx), int((f[1] + f[3]) * sy)
                )
                
                aligned = align_face_112(frame, lm)
                if aligned is not None and not is_blurry(aligned):
                    vec = embedder.embed(aligned)
                    
                    if phase == "stare":
                        if len(embeddings) < STARE_SAMPLES:
                            embeddings.append(vec)
                    
                    elif phase == "rotate":
                        # Check if this angle is new
                        is_diverse = True
                        if embeddings:
                            sims = np.array(embeddings) @ vec
                            # If too similar to any existing frame, skip it
                            if sims.max() > 0.90:  # Stricter diversity for ResNet
                                is_diverse = False
                        
                        if is_diverse:
                            embeddings.append(vec)
            else:
                last_bbox = None
        
        # Draw GUI
        if last_bbox is not None:
            x1, y1, x2, y2 = last_bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), status_color, 2)
        
        cv2.putText(frame, f"Enrolling: {name}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Samples: {len(embeddings)}/{TOTAL_SAMPLES}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, status, (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        progress = int((len(embeddings) / TOTAL_SAMPLES) * 300)
        cv2.rectangle(frame, (20, CAM_H - 40), (320, CAM_H - 20), (50, 50, 50), -1)
        cv2.rectangle(frame, (20, CAM_H - 40), (20 + progress, CAM_H - 20), (0, 255, 0), -1)
        
        cv2.imshow("Enroll", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            print("\nCancelled.")
            break
    
    grab.release()
    cv2.destroyAllWindows()
    
    if len(embeddings) >= TOTAL_SAMPLES:
        np.save(user_dir / "embeddings.npy", np.stack(embeddings).astype(np.float32))
        print(f"\nSUCCESS: Enrolled {name}")
        print(f"Saved to: {user_dir / 'embeddings.npy'}")
    else:
        print("\nEnrollment incomplete.")

if __name__ == "__main__":
    main()

# app.py
import streamlit as st
import cv2
import time
import numpy as np
import os
import torch
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image

# -------------------------
# Page / CSS
# -------------------------
st.set_page_config(page_title="Sign Language - 4s Hold", layout="centered")
st.markdown("""
<style>
.stApp { background: linear-gradient(120deg,#071025 0%, #0f172a 50%, #061226 100%); color: #e6eef8; font-family: "Inter", "Segoe UI", sans-serif; }
.header { font-size: 2.0rem; font-weight:700; color:#7dd3fc; text-align:center; margin-bottom:4px; }
.sub { color:#9fb4c9; text-align:center; margin-bottom:10px; }
.instructions { color:#dbeafe; background: rgba(255,255,255,0.02); padding:10px; border-radius:8px; }
.box-wrap { display:flex; justify-content:center; margin-top:10px; }
.cam-box { border:3px solid #38bdf8; border-radius:12px; padding:8px; box-shadow: 0 8px 20px rgba(2,6,23,0.6); }
.word { text-align:center; font-size:1.4rem; color:#facc15; margin-top:12px; font-weight:700; }
.current { text-align:center; font-size:1.6rem; color:#4ade80; margin-top:8px; font-weight:700; }
.clear-btn { margin-top: 8px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="header">Sign Language Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">Show each sign for <b>4 seconds</b> inside the box to record a letter</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="instructions"><b>Instructions:</b><br>'
    '1) Place your hand entirely inside the central box.<br>'
    '2) Hold a sign steady for <b>4 seconds</b> — it will be recorded automatically.<br>'
    '3) After recording, remove your hand, then show the next sign.<br>'
    "4) Use 'Clear Word' to reset whenever you want.</div>", unsafe_allow_html=True
)

# -------------------------
# Load model (cached)
# -------------------------
@st.cache_resource
def load_model(path="best_model_resnet18.pth"):
    if not os.path.exists(path):
        return None, None, None
    ckpt = torch.load(path, map_location="cpu")
    model_state = ckpt["model_state_dict"]
    class_to_idx = ckpt["class_to_idx"]
    img_size = ckpt.get("img_size", 224)
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    model = resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(class_to_idx))
    model.load_state_dict(model_state)
    model.eval()
    return model, idx_to_class, img_size

model, idx_to_class, IMG_SIZE = load_model()
if model is None:
    st.error("Model file 'best_model_resnet18.pth' not found in this folder. Place it here and reload.")
    st.stop()

# Preprocess (must match training)
preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# -------------------------
# Session state defaults
# -------------------------
if "camera" not in st.session_state:
    st.session_state.camera = None
if "running" not in st.session_state:
    st.session_state.running = False
if "word" not in st.session_state:
    st.session_state.word = ""
if "last_pred" not in st.session_state:
    st.session_state.last_pred = ""
if "pred_start_time" not in st.session_state:
    st.session_state.pred_start_time = None
if "await_removal" not in st.session_state:
    st.session_state.await_removal = False  # True after commit; wait until hand removed

# Controls
col1, col2, col3 = st.columns([1,1,1])
with col1:
    if st.button("Start Camera"):
        st.session_state.running = True
        # initialize camera
        if st.session_state.camera is None:
            st.session_state.camera = cv2.VideoCapture(0)
            time.sleep(0.3)
with col2:
    if st.button("Stop Camera"):
        st.session_state.running = False
        try:
            if st.session_state.camera is not None:
                st.session_state.camera.release()
        except:
            pass
        st.session_state.camera = None
with col3:
    if st.button("Clear Word"):
        st.session_state.word = ""

# Placeholders for UI updates
frame_placeholder = st.empty()
current_placeholder = st.empty()
word_placeholder = st.empty()

# Parameters
BOX_SIZE = 240              # size of center box in pixels
SKIN_MIN = np.array([0, 30, 60], dtype=np.uint8)
SKIN_MAX = np.array([20, 150, 255], dtype=np.uint8)
MIN_HAND_AREA = 1200       # minimal contour area inside box to count as hand
HOLD_SECONDS = 4.0         # 4-second hold to commit

# Main camera loop (non-blocking check each iteration)
if st.session_state.running:
    cam = st.session_state.camera
    if cam is None:
        st.error("Unable to open camera.")
        st.session_state.running = False
    else:
        try:
            while st.session_state.running:
                ret, frame = cam.read()
                if not ret:
                    st.error("Failed to read from camera.")
                    break

                frame = cv2.flip(frame, 1)  # mirror
                h, w, _ = frame.shape

                # fixed center box
                bx1 = w//2 - BOX_SIZE//2
                by1 = h//2 - BOX_SIZE//2
                bx2 = bx1 + BOX_SIZE
                by2 = by1 + BOX_SIZE
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), (56,182,255), 3, cv2.LINE_AA)

                # crop roi
                roi = frame[by1:by2, bx1:bx2]
                # ensure roi has content (camera smaller than box edge case)
                if roi is None or roi.size == 0 or roi.shape[0] == 0 or roi.shape[1] == 0:
                    frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    current_placeholder.markdown("", unsafe_allow_html=True)
                    word_placeholder.markdown(f"<div class='word'>{st.session_state.word}</div>", unsafe_allow_html=True)
                    time.sleep(0.03)
                    continue

                hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv, SKIN_MIN, SKIN_MAX)
                # clean mask
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
                mask = cv2.GaussianBlur(mask, (5,5), 0)

                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                detected_hand = False
                pred_label = None

                if contours:
                    cnt = max(contours, key=cv2.contourArea)
                    area = cv2.contourArea(cnt)
                    if area > MIN_HAND_AREA:
                        detected_hand = True
                        # Predict from cropped region (RGB)
                        crop_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                        pil = Image.fromarray(crop_rgb)
                        tensor = preprocess(pil).unsqueeze(0)
                        with torch.no_grad():
                            out = model(tensor)
                            probs = torch.softmax(out, dim=1)
                            idx = int(probs.argmax(1).item())
                            pred_label = idx_to_class[idx]
                            conf = float(probs[0][idx].item())
                        # draw a small label on the box (optional)
                        cv2.putText(frame, f"{pred_label} {int(conf*100)}%", (bx1+6, by1-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

                # --- Logic for 4-second hold + awaiting removal ---
                if detected_hand and pred_label is not None and not st.session_state.await_removal:
                    # if same as last predicted, check timer
                    if st.session_state.last_pred == pred_label and st.session_state.pred_start_time is not None:
                        elapsed = time.time() - st.session_state.pred_start_time
                        # show countdown in UI
                        remaining = max(0.0, HOLD_SECONDS - elapsed)
                        current_placeholder.markdown(f"<div class='current'>Hold: {pred_label} &nbsp; ({remaining:.1f}s)</div>", unsafe_allow_html=True)
                        if elapsed >= HOLD_SECONDS:
                            # commit letter
                            st.session_state.word += pred_label
                            # set await_removal until hand leaves the box
                            st.session_state.await_removal = True
                            st.session_state.last_pred = ""
                            st.session_state.pred_start_time = None
                            # show committed briefly
                            current_placeholder.markdown(f"<div class='current'>Committed: {pred_label}</div>", unsafe_allow_html=True)
                    else:
                        # new candidate
                        st.session_state.last_pred = pred_label
                        st.session_state.pred_start_time = time.time()
                        # show starting timer
                        current_placeholder.markdown(f"<div class='current'>Hold: {pred_label} &nbsp; (4.0s)</div>", unsafe_allow_html=True)
                else:
                    # No hand detected OR we are awaiting removal
                    if st.session_state.await_removal:
                        # if awaiting removal, check if hand has left (no contour)
                        if not detected_hand:
                            st.session_state.await_removal = False
                            # reset placeholders
                            current_placeholder.markdown("", unsafe_allow_html=True)
                    else:
                        # box empty -> show nothing (per request)
                        current_placeholder.markdown("", unsafe_allow_html=True)
                        # reset temporary timer state to avoid carryover
                        st.session_state.last_pred = ""
                        st.session_state.pred_start_time = None

                # update word display and show frame
                word_placeholder.markdown(f"<div class='word'>{st.session_state.word}</div>", unsafe_allow_html=True)
                frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

                # control CPU usage
                time.sleep(0.03)

        except Exception as e:
            st.error(f"Camera loop error: {e}")
        finally:
            try:
                if st.session_state.camera is not None:
                    st.session_state.camera.release()
            except:
                pass
            st.session_state.camera = None
            st.session_state.running = False
else:
    # not running: show empty placeholders
    frame_placeholder.image(np.zeros((480, 640, 3), dtype=np.uint8))
    current_placeholder.markdown("", unsafe_allow_html=True)
    word_placeholder.markdown(f"<div class='word'>{st.session_state.word}</div>", unsafe_allow_html=True)

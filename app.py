import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile


#This gives the Live metrics for only 4 options showing everytime where problem is present or not



# --- 1. CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Smart Physio AI")

# Load MediaPipe
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils # Re-enabled for drawing skeleton
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=0, # Lite model for speed
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Diagnostic Thresholds
HUNCHBACK_THRESHOLD = 0.08  
NECK_TILT_LIMIT = 10      
SHOULDER_DIFF_LIMIT = 0.03 
KNEE_VALGUS_LIMIT = 170    

# --- 2. MATH HELPERS ---
def calculate_angle(a, b, c):
    a = np.array(a); b = np.array(b); c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0: angle = 360-angle
    return angle

def calculate_y_diff(a, b):
    return abs(a.y - b.y)

# --- 3. DIAGNOSTIC FUNCTIONS ---
def analyze_neck(landmarks):
    nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x, landmarks[mp_pose.PoseLandmark.NOSE.value].y]
    l_shldr = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    r_shldr = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    shldr_mid = [(l_shldr.x + r_shldr.x)/2, (l_shldr.y + r_shldr.y)/2]
    vertical_ref = [shldr_mid[0], shldr_mid[1] - 0.5]
    neck_angle = calculate_angle(vertical_ref, shldr_mid, nose)
    
    if neck_angle > NECK_TILT_LIMIT: return "Neck Tilt", f"{int(neck_angle)}°", "off", "Tip: Align your head vertically. Stretching the neck muscles may help."
    return "Neck Status", "Aligned", "normal", ""

def analyze_shoulders(landmarks):
    l = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    r = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    if calculate_y_diff(l, r) > SHOULDER_DIFF_LIMIT: return "Shoulders", "Uneven", "off", "Tip: One shoulder is higher. Check if you are leaning or carrying tension."
    return "Shoulders", "Balanced", "normal", ""

def analyze_legs(landmarks):
    l_hip = [landmarks[23].x, landmarks[23].y]
    l_knee = [landmarks[25].x, landmarks[25].y]
    l_ankle = [landmarks[27].x, landmarks[27].y]
    angle = calculate_angle(l_hip, l_knee, l_ankle)
    if angle < KNEE_VALGUS_LIMIT: return "Knee Valgus", f"{int(angle)}°", "off", "Tip: Your knees are collapsing inward. Try pushing your knees out to align with toes."
    return "Leg Align", "Good", "normal", ""

def check_hunchback(landmarks):
    l_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR.value]
    l_shldr = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    if abs(l_ear.x - l_shldr.x) > HUNCHBACK_THRESHOLD: return "Posture", "Hunchback", "off", "Tip: Forward head detected. Tuck your chin in and pull shoulders back."
    return "Posture", "Good", "normal", ""

# --- 4. STREAMLIT UI ---
st.title("Smart Physio")

# Metric Dashboard (Top Row)
st.markdown("### 📊 Live Health Metrics")
m1, m2, m3, m4 = st.columns(4)
metric_posture = m1.empty()
metric_neck = m2.empty()
metric_shoulder = m3.empty()
metric_legs = m4.empty()

st.divider()

# Layout: Video Centered
c1, c2, c3 = st.columns([1, 6, 1]) 
with c2:
    st.subheader("Live Analysis")
    video_placeholder = st.empty()
    # Placeholder for suggestions BELOW the video
    suggestion_box = st.empty()

# Sidebar Control
st.sidebar.header("Configuration")
source = st.sidebar.radio("Video Source", ["Webcam", "Upload Video"])

# --- 5. MAIN LOOP ---
cap = None
if source == "Webcam":
    cap = cv2.VideoCapture(0)
elif source == "Upload Video":
    uploaded_file = st.sidebar.file_uploader("Upload MP4", type=["mp4"])
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)

if cap:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # Resize for performance (640px width)
        h, w, c = frame.shape
        new_w = 640
        new_h = int(h * (new_w / w))
        image = cv2.resize(frame, (new_w, new_h))

        # Process Image
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = pose.process(image_rgb)
        image_rgb.flags.writeable = True
        
        # Initialize list to hold active suggestions
        active_suggestions = []

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # 1. DRAW SKELETON (Restored)
            mp_draw.draw_landmarks(
                image_rgb, 
                results.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS,
                mp_draw.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), # Joint Color
                mp_draw.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)  # Bone Color
            )
            
            # 2. RUN DIAGNOSTICS & COLLECT TIPS
            lbl_n, val_n, col_n, tip_n = analyze_neck(landmarks)
            lbl_s, val_s, col_s, tip_s = analyze_shoulders(landmarks)
            lbl_l, val_l, col_l, tip_l = analyze_legs(landmarks)
            lbl_p, val_p, col_p, tip_p = check_hunchback(landmarks)
            
            # Update Dashboard
            metric_neck.metric(label=lbl_n, value=val_n, delta=val_n if col_n=="off" else None, delta_color=col_n)
            metric_shoulder.metric(label=lbl_s, value=val_s, delta=val_s if col_s=="off" else None, delta_color=col_s)
            metric_legs.metric(label=lbl_l, value=val_l, delta=val_l if col_l=="off" else None, delta_color=col_l)
            metric_posture.metric(label=lbl_p, value=val_p, delta=val_p if col_p=="off" else None, delta_color=col_p)

            # Collect Suggestions if there is a problem
            if tip_n: active_suggestions.append(f"**Neck:** {tip_n}")
            if tip_s: active_suggestions.append(f"**Shoulders:** {tip_s}")
            if tip_l: active_suggestions.append(f"**Legs:** {tip_l}")
            if tip_p: active_suggestions.append(f"**Posture:** {tip_p}")

        # Display Video
        video_placeholder.image(image_rgb, channels="RGB", use_container_width=True)
        
        # Display Suggestions Below Video
        if active_suggestions:
            # Show a warning box with all active tips
            suggestion_box.warning("\n\n".join(active_suggestions), icon="⚠️")
        else:
            # Show success box if posture is perfect
            suggestion_box.success("✅ Perfect Form! Keep maintaining this posture.", icon="🌟")

    cap.release()
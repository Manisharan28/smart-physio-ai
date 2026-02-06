import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import time
import ollama

# --- 1. CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Smart Physio AI (Ollama Edition)")

# Load MediaPipe
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1, 
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Initialize Session State
if "last_analysis_time" not in st.session_state:
    st.session_state.last_analysis_time = 0
if "last_message" not in st.session_state:
    st.session_state.last_message = "Waiting for AI analysis..."

# --- 2. MATH ENGINE ---
def calculate_angle(a, b, c):
    """Calculates angle between three points (a-b-c)."""
    a = np.array(a); b = np.array(b); c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0: angle = 360-angle
    return int(angle)

def get_full_body_metrics(landmarks):
    """Extracts data for the LLM."""
    def loc(idx): return [landmarks[idx].x, landmarks[idx].y]
    metrics = {}
    
    # Upper Body
    metrics["Neck_Tilt"] = calculate_angle(
        [landmarks[11].x, landmarks[11].y - 0.5], 
        [(landmarks[11].x+landmarks[12].x)/2, (landmarks[11].y+landmarks[12].y)/2], 
        loc(0)
    )
    metrics["Shoulder_Balance"] = round(abs(landmarks[11].y - landmarks[12].y), 3)

    # Lower Body
    metrics["Right_Knee_Angle"] = calculate_angle(loc(24), loc(26), loc(28))
    metrics["Left_Knee_Angle"]  = calculate_angle(loc(23), loc(25), loc(27))
    
    return metrics

def calculate_model_accuracy(landmarks):
    """Calculates visibility confidence."""
    key_indices = [11, 12, 23, 24, 25, 26, 27, 28]
    confidences = [landmarks[i].visibility for i in key_indices]
    return round((sum(confidences) / len(confidences)) * 100, 2)

# --- 3. OLLAMA AI ENGINE ---
def ask_ollama(metrics):
    """Sends data to local Llama 3."""
    prompt = f"""
    You are an expert Physiotherapist AI. Analyze these metrics:
    {metrics}
    
    Rules:
    - Normal Neck Tilt: < 10 degrees.
    - Normal Shoulder Diff: < 0.04.
    - Normal Standing Knee Angle: > 165 degrees (if less, it's Valgus).

    Task:
    Identify the ONE most critical issue and give a 1-sentence correction tip.
    If healthy, say "Form is excellent."
    Output ONLY the diagnosis and tip.
    """
    
    try:
        # Ensure you have run 'ollama pull llama3.2' in terminal
        response = ollama.chat(model='llama3.2', messages=[
            {'role': 'user', 'content': prompt},
        ])
        return response['message']['content']
    except Exception as e:
        return "⚠️ Error: Ollama not running. Please open the Ollama app."

# --- 4. STREAMLIT UI ---
st.title("🤖 Smart Physio: Local Llama 3 Diagnostics")

col_video, col_chat = st.columns([1.5, 1])

# Sidebar
st.sidebar.header("Settings")
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

# UI Containers (Created ONCE outside the loop)
with col_video:
    st.subheader("Live Feed")
    video_placeholder = st.empty()
    # This single placeholder prevents the stacking numbers
    accuracy_container = st.empty()

with col_chat:
    st.subheader("🧠 Dr. Llama Analysis")
    chat_placeholder = st.empty()

if cap:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # Resize for speed
        h, w, c = frame.shape
        new_w = 640
        new_h = int(h * (new_w / w))
        image = cv2.resize(frame, (new_w, new_h))

        # Process
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = pose.process(image_rgb)
        image_rgb.flags.writeable = True

        accuracy_score = 0

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Draw Skeleton
            mp_draw.draw_landmarks(image_rgb, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            accuracy_score = calculate_model_accuracy(landmarks)
            
            # --- AI CHECK (Every 4 Seconds) ---
            current_time = time.time()
            if (current_time - st.session_state.last_analysis_time > 4.0):
                
                if accuracy_score > 60:
                    metrics = get_full_body_metrics(landmarks)
                    
                    # Indicate thinking
                    chat_placeholder.markdown("🔄 *Thinking...*")
                    
                    # Call Ollama
                    ai_response = ask_ollama(metrics)
                    
                    st.session_state.last_message = ai_response
                    st.session_state.last_analysis_time = current_time
                else:
                    st.session_state.last_message = "⚠️ Patient not fully visible. Step back to show hips/knees."

        # --- UPDATE UI ---
        video_placeholder.image(image_rgb, channels="RGB", use_container_width=True)

        # Update Chat
        chat_placeholder.info(st.session_state.last_message)

        # Update Accuracy (Using container to overwrite previous frame)
        with accuracy_container.container():
            col_bar, col_text = st.columns([0.85, 0.15])
            with col_bar:
                st.progress(int(accuracy_score))
            with col_text:
                st.markdown(f"**{int(accuracy_score)}%**")

    cap.release()
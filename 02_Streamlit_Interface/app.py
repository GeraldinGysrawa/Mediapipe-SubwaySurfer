import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import joblib
import pyautogui
import time
import math
from PIL import Image
import os
import importlib
import sys

# Add current directory to path so we can import other modules
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import other pages
import gesture_capture
import train_model_app

# Set page config
st.set_page_config(
    page_title="Subway Surfers Head Gesture Control",
    page_icon="🎮",
    layout="wide"
)

# Create session state for navigation if it doesn't exist
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Play Game"

# Sidebar navigation
st.sidebar.title("Navigation")
selected_page = st.sidebar.radio(
    "Go to",
    ["Play Game", "Capture Gestures", "Train Model"],
    key="navigation",
)

# Update current page in session state
st.session_state.current_page = selected_page

# Display the selected page
if selected_page == "Play Game":
    st.title("🎮 Subway Surfers Head Gesture Control")
    st.markdown("""
        Control Subway Surfers using head gestures! 
        - **Tilt Left**: Move Left
        - **Tilt Right**: Move Right
        - **Tilt Up**: Jump
        - **Tilt Down**: Duck
        - **Neutral**: No Action
    """)

    # Sidebar controls for game
    st.sidebar.header("Game Controls")
    gesture_delay = st.sidebar.slider("Gesture Delay (seconds)", 0.1, 2.0, 0.5)

    @st.cache_resource
    def load_models():
        # Get the current script's directory
        models_dir = os.path.join(current_dir, 'models')
        
        # Load models with correct paths
        model = joblib.load(os.path.join(models_dir, 'gesture_svm.pkl'))
        scaler = joblib.load(os.path.join(models_dir, 'scaler.pkl'))
        label_encoder = joblib.load(os.path.join(models_dir, 'label_encoder.pkl'))
        return model, scaler, label_encoder

    def calculate_head_pose(landmarks):
        if not landmarks:
            return None
        nose_tip = landmarks[1]
        left_eye = landmarks[33]
        right_eye = landmarks[263]
        mouth_left = landmarks[61]
        mouth_right = landmarks[291]
        
        eye_center_x = (left_eye.x + right_eye.x) / 2
        nose_x = nose_tip.x
        yaw = (nose_x - eye_center_x) * 180
        
        eye_center_y = (left_eye.y + right_eye.y) / 2
        mouth_center_y = (mouth_left.y + mouth_right.y) / 2
        nose_y = nose_tip.y
        face_height = abs(eye_center_y - mouth_center_y)
        
        if face_height > 0:
            pitch = ((nose_y - eye_center_y) / face_height) * 60
        else:
            pitch = 0
            
        eye_slope = (right_eye.y - left_eye.y) / (right_eye.x - left_eye.x) if right_eye.x != left_eye.x else 0
        roll = math.atan(eye_slope) * 180 / math.pi
        
        return {'yaw': yaw, 'pitch': pitch, 'roll': roll}

    # Load models
    try:
        model, scaler, label_encoder = load_models()
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.warning("Please train the model first using the Train Model page!")
        st.stop()

    # Setup MediaPipe Face Mesh with default optimal values
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("Could not open webcam!")
        st.stop()

    # Create placeholders for video feed and status
    video_placeholder = st.empty()
    status_placeholder = st.empty()
    metrics_placeholder = st.empty()
    
    last_gesture = None
    last_time = time.time()
    
    # Warning message
    st.warning("⚠️ Make sure your Subway Surfers browser window is active and focused before using gestures!")
    
    # Start/Stop button
    running = st.checkbox("Start Head Gesture Control")

    while running:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture frame from webcam!")
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        gesture_text = "NEUTRAL"
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                pose = calculate_head_pose(face_landmarks.landmark)
                if pose:
                    fitur = np.array([[pose['yaw'], pose['pitch'], pose['roll']]])
                    fitur_scaled = scaler.transform(fitur)
                    pred = model.predict(fitur_scaled)
                    gesture = label_encoder.inverse_transform(pred)[0]
                    gesture_text = gesture
                
                # Draw landmarks
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, 
                    face_landmarks, 
                    mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
                )

        # Control game with detected gesture
        if gesture_text != last_gesture and gesture_text in ["JUMP", "LEFT", "RIGHT", "DUCK"]:
            if time.time() - last_time > gesture_delay:
                if gesture_text == "JUMP":
                    pyautogui.press('up')
                elif gesture_text == "LEFT":
                    pyautogui.press('left')
                elif gesture_text == "RIGHT":
                    pyautogui.press('right')
                elif gesture_text == "DUCK":
                    pyautogui.press('down')
                last_gesture = gesture_text
                last_time = time.time()
        elif gesture_text == "NEUTRAL":
            last_gesture = None

        # Display gesture text on frame
        cv2.putText(
            frame, 
            f"Gesture: {gesture_text}", 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (0, 255, 0), 
            2
        )

        # Convert frame to PIL Image and display
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        video_placeholder.image(img, channels="RGB", use_column_width=True)
        
        # Update status and metrics
        status_placeholder.info(f"Current Gesture: {gesture_text}")
        
        with metrics_placeholder.container():
            col1, col2, col3 = st.columns(3)
            if results.multi_face_landmarks and pose:
                col1.metric("Yaw", f"{pose['yaw']:.2f}°")
                col2.metric("Pitch", f"{pose['pitch']:.2f}°")
                col3.metric("Roll", f"{pose['roll']:.2f}°")

    # Cleanup
    if not running and 'cap' in locals():
        cap.release()

elif selected_page == "Capture Gestures":
    # Run the gesture capture page
    gesture_capture.main()

elif selected_page == "Train Model":
    # Run the model training page
    train_model_app.main() 
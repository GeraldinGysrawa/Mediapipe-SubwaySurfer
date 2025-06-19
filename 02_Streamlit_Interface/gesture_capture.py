import streamlit as st
import cv2
import mediapipe as mp
import json
import os
import datetime
import numpy as np
import math
import time
from PIL import Image

# Get the current script's directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Directory to save gesture images and JSON
GESTURE_DIR = os.path.join(CURRENT_DIR, 'head_gestures')
GESTURE_JSON = os.path.join(GESTURE_DIR, 'head_gestures.json')

# Create directory if not exists
if not os.path.exists(GESTURE_DIR):
    os.makedirs(GESTURE_DIR, exist_ok=True)

# Load or initialize gesture database
if os.path.exists(GESTURE_JSON):
    with open(GESTURE_JSON, 'r') as f:
        gesture_db = json.load(f)
else:
    gesture_db = []

# MediaPipe setup untuk face mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Subway Surfers gestures
SUBWAY_GESTURES = {
    'NEUTRAL': 'Posisi normal/netral',
    'JUMP': 'Kepala ke atas (loncat)',
    'LEFT': 'Kepala ke kiri',
    'RIGHT': 'Kepala ke kanan',
    'DUCK': 'Kepala ke bawah (optional)'
}

# Colors for each gesture
GESTURE_COLORS = {
    'NEUTRAL': '#757575',   # Grey
    'JUMP': '#4caf50',      # Green
    'LEFT': '#2196f3',      # Blue  
    'RIGHT': '#ff9800',     # Orange
    'DUCK': '#9c27b0'       # Purple
}

def calculate_head_pose(landmarks):
    """Calculate head pose angles (yaw, pitch, roll) from face landmarks"""
    if not landmarks:
        return None
        
    # Key landmarks for head pose calculation
    nose_tip = landmarks[1]
    left_eye = landmarks[33]
    right_eye = landmarks[263]
    mouth_left = landmarks[61]
    mouth_right = landmarks[291]
    
    # Calculate yaw (left-right rotation)
    eye_center_x = (left_eye.x + right_eye.x) / 2
    nose_x = nose_tip.x
    yaw = (nose_x - eye_center_x) * 180  # Rough approximation
    
    # Calculate pitch (up-down rotation)
    eye_center_y = (left_eye.y + right_eye.y) / 2
    mouth_center_y = (mouth_left.y + mouth_right.y) / 2
    nose_y = nose_tip.y
    
    # Pitch calculation based on relative positions
    face_height = abs(eye_center_y - mouth_center_y)
    if face_height > 0:
        pitch = ((nose_y - eye_center_y) / face_height) * 60  # Normalized
    else:
        pitch = 0
        
    # Calculate roll (tilt rotation)
    eye_slope = (right_eye.y - left_eye.y) / (right_eye.x - left_eye.x) if right_eye.x != left_eye.x else 0
    roll = math.atan(eye_slope) * 180 / math.pi
    
    return {
        'yaw': yaw,
        'pitch': pitch, 
        'roll': roll
    }

def classify_gesture(head_pose, calibration_data):
    """Classify head pose into gesture categories"""
    if not head_pose:
        return 'UNKNOWN'
        
    # Adjust for calibrated neutral position
    yaw = head_pose['yaw'] - calibration_data['neutral_yaw']
    pitch = head_pose['pitch'] - calibration_data['neutral_pitch']
    
    thresholds = calibration_data['thresholds']
    
    # Classify based on thresholds
    if yaw < thresholds['yaw_left']:
        return 'LEFT'
    elif yaw > thresholds['yaw_right']:
        return 'RIGHT'
    elif pitch < thresholds['pitch_up']:
        return 'JUMP'
    elif pitch > thresholds['pitch_down']:
        return 'DUCK'
    else:
        return 'NEUTRAL'

def crop_face(frame, face_landmarks):
    """Crop face region from frame"""
    if not face_landmarks:
        return frame
        
    h, w, _ = frame.shape
    
    # Get face bounding box
    x_coords = [int(lm.x * w) for lm in face_landmarks]
    y_coords = [int(lm.y * h) for lm in face_landmarks]
    
    # Perbesar margin crop
    x_min = max(min(x_coords) - 60, 0)
    x_max = min(max(x_coords) + 60, w)
    y_min = max(min(y_coords) - 80, 0)
    y_max = min(max(y_coords) + 60, h)
    
    face_crop = frame[y_min:y_max, x_min:x_max].copy()
    return face_crop

def main():
    st.title("🎮 Head Gesture Capture - Subway Surfers")
    
    # Initialize session state
    if 'calibration_data' not in st.session_state:
        st.session_state.calibration_data = {
            'neutral_yaw': 0,
            'neutral_pitch': 0,
            'neutral_roll': 0,
            'thresholds': {
                'yaw_left': -15,
                'yaw_right': 15,
                'pitch_up': -15,
                'pitch_down': 15
            }
        }

        # Load calibration if exists
        cal_file = os.path.join(GESTURE_DIR, 'calibration.json')
        if os.path.exists(cal_file):
            with open(cal_file, 'r') as f:
                st.session_state.calibration_data = json.load(f)

    # Sidebar
    st.sidebar.title("Capture Controls")
    selected_gesture = st.sidebar.selectbox(
        "Select Gesture",
        list(SUBWAY_GESTURES.keys()),
        format_func=lambda x: f"{x} - {SUBWAY_GESTURES[x]}"
    )

    # Main content columns
    col1, col2 = st.columns([2, 1])

    with col1:
        # Video feed placeholder
        video_placeholder = st.empty()
        
        # Status indicators
        status_placeholder = st.empty()
        metrics_container = st.container()

    with col2:
        # Calibration and capture controls
        st.subheader("Actions")
        
        if st.button("📏 Calibrate Neutral Position", use_container_width=True):
            if 'head_pose' in st.session_state:
                st.session_state.calibration_data.update({
                    'neutral_yaw': st.session_state.head_pose['yaw'],
                    'neutral_pitch': st.session_state.head_pose['pitch'],
                    'neutral_roll': st.session_state.head_pose['roll']
                })
                st.success("Neutral position calibrated!")
                
                # Save calibration
                cal_file = os.path.join(GESTURE_DIR, 'calibration.json')
                with open(cal_file, 'w') as f:
                    json.dump(st.session_state.calibration_data, f, indent=2)

        if st.button("📸 Capture Single", use_container_width=True):
            if 'frame' in st.session_state and 'face_landmarks' in st.session_state:
                timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                img_filename = f'{selected_gesture}_{timestamp}.png'
                img_path = os.path.join(GESTURE_DIR, img_filename)
                
                face_img = crop_face(st.session_state.frame, st.session_state.face_landmarks)
                cv2.imwrite(img_path, face_img)
                
                gesture_entry = {
                    'name': selected_gesture,
                    'head_pose': st.session_state.head_pose,
                    'landmarks_count': len(st.session_state.face_landmarks),
                    'image_path': img_path,
                    'timestamp': timestamp
                }
                
                gesture_db.append(gesture_entry)
                with open(GESTURE_JSON, 'w') as f:
                    json.dump(gesture_db, f, indent=2)
                    
                st.success(f"Captured {selected_gesture} gesture!")
            else:
                st.error("No face detected!")

        if st.button("📸 Capture Multiple (10x)", use_container_width=True):
            if 'frame' not in st.session_state or 'face_landmarks' not in st.session_state:
                st.error("No face detected!")
                return
                
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i in range(10):
                if 'frame' in st.session_state and 'face_landmarks' in st.session_state:
                    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                    img_filename = f'{selected_gesture}_{timestamp}_{i+1}.png'
                    img_path = os.path.join(GESTURE_DIR, img_filename)
                    
                    face_img = crop_face(st.session_state.frame, st.session_state.face_landmarks)
                    cv2.imwrite(img_path, face_img)
                    
                    gesture_entry = {
                        'name': selected_gesture,
                        'head_pose': st.session_state.head_pose,
                        'landmarks_count': len(st.session_state.face_landmarks),
                        'image_path': img_path,
                        'timestamp': timestamp,
                        'batch_id': i + 1
                    }
                    
                    gesture_db.append(gesture_entry)
                    with open(GESTURE_JSON, 'w') as f:
                        json.dump(gesture_db, f, indent=2)
                    
                    progress_bar.progress((i + 1) / 10)
                    status_text.text(f"Captured {i+1}/10")
                    time.sleep(0.5)
            
            st.success(f"Captured 10 samples of {selected_gesture} gesture!")

        # Display database stats
        st.subheader("Database Stats")
        gesture_counts = {}
        for entry in gesture_db:
            gesture_counts[entry['name']] = gesture_counts.get(entry['name'], 0) + 1
            
        for gesture in SUBWAY_GESTURES:
            count = gesture_counts.get(gesture, 0)
            st.metric(gesture, count)

    # Initialize face mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # Initialize webcam
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to access webcam!")
            break

        # Flip frame horizontally
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        display_frame = rgb_frame.copy()
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Store in session state
                st.session_state.face_landmarks = face_landmarks.landmark
                st.session_state.frame = frame
                
                # Draw landmarks
                mp_drawing.draw_landmarks(
                    display_frame,
                    face_landmarks,
                    mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                )
                
                # Calculate and store head pose
                head_pose = calculate_head_pose(face_landmarks.landmark)
                if head_pose:
                    st.session_state.head_pose = head_pose
                    
                    # Classify gesture
                    detected_gesture = classify_gesture(head_pose, st.session_state.calibration_data)
                    
                    # Display metrics
                    with metrics_container:
                        cols = st.columns(4)
                        cols[0].metric("Yaw", f"{head_pose['yaw']:.1f}°")
                        cols[1].metric("Pitch", f"{head_pose['pitch']:.1f}°")
                        cols[2].metric("Roll", f"{head_pose['roll']:.1f}°")
                        cols[3].metric("Gesture", detected_gesture)
                    
                    # Add text to frame
                    cv2.putText(display_frame, f"Gesture: {detected_gesture}", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display frame
        video_placeholder.image(display_frame, channels="RGB", use_column_width=True)

        # Break if window is closed
        if not st.session_state.get('run_capture', True):
            break

    cap.release()

if __name__ == "__main__":
    st.set_page_config(page_title="Head Gesture Capture", page_icon="🎮", layout="wide")
    main() 
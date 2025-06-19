import cv2
import mediapipe as mp
import numpy as np
import joblib

# Load model, scaler, dan label encoder
model = joblib.load('models/gesture_svm.pkl')
scaler = joblib.load('models/scaler.pkl')
label_encoder = joblib.load('models/label_encoder.pkl')

# Setup MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Fungsi untuk menghitung yaw, pitch, roll dari landmark
import math
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

# Mulai webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    gesture_text = "-"
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Hitung head pose
            pose = calculate_head_pose(face_landmarks.landmark)
            if pose:
                fitur = np.array([[pose['yaw'], pose['pitch'], pose['roll']]])
                fitur_scaled = scaler.transform(fitur)
                pred = model.predict(fitur_scaled)
                gesture = label_encoder.inverse_transform(pred)[0]
                gesture_text = gesture
            # Gambar face mesh
            mp.solutions.drawing_utils.draw_landmarks(
                frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)
    # Tampilkan hasil prediksi gesture
    cv2.putText(frame, f"Gesture: {gesture_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Deteksi Gesture Kepala Real-Time', frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC untuk keluar
        break
cap.release()
cv2.destroyAllWindows() 
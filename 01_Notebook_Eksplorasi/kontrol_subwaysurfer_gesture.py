import cv2
import mediapipe as mp
import numpy as np
import joblib
import pyautogui
import time
import math

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

cap = cv2.VideoCapture(0)
last_gesture = None
last_time = time.time()

print("Pastikan jendela browser Subway Surfers sudah aktif dan fokus!")
print("Gerakkan kepala untuk mengontrol game. Tekan ESC untuk keluar.")

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
            pose = calculate_head_pose(face_landmarks.landmark)
            if pose:
                fitur = np.array([[pose['yaw'], pose['pitch'], pose['roll']]])
                fitur_scaled = scaler.transform(fitur)
                pred = model.predict(fitur_scaled)
                gesture = label_encoder.inverse_transform(pred)[0]
                gesture_text = gesture
            mp.solutions.drawing_utils.draw_landmarks(
                frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)
    # Kontrol game Subway Surfers dengan pyautogui
    if gesture_text != last_gesture and gesture_text in ["JUMP", "LEFT", "RIGHT", "DUCK"]:
        if time.time() - last_time > 0.5:  # jeda 0.5 detik antar gesture
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
    # Tampilkan hasil prediksi gesture
    cv2.putText(frame, f"Gesture: {gesture_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Kontrol Subway Surfers dengan Gesture Kepala', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows() 
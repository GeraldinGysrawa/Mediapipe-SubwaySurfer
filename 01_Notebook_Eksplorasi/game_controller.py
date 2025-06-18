import cv2
import mediapipe as mp
import pyautogui
import time

# Inisialisasi MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Buka akses ke webcam
cap = cv2.VideoCapture(0)

# Dapatkan ukuran layar untuk referensi tengah
screen_w, screen_h = pyautogui.size()

# --- Variabel untuk Logika & Kalibrasi ---

# Threshold untuk menentukan seberapa jauh gerakan dianggap sebagai perintah
# Nilai ini mungkin perlu Anda sesuaikan!
X_THRESHOLD = 0.1  # 10% dari lebar frame
Y_THRESHOLD = 0.08 # 8% dari tinggi frame

# Variabel state untuk mencegah spam tombol
# Ini memastikan tombol hanya ditekan sekali per gerakan
command = None
last_command_time = 0
COMMAND_COOLDOWN = 0.5 # Cooldown 0.4 detik antar perintah

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    image = cv2.flip(image, 1)
    frame_h, frame_w, _ = image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    if results.multi_face_landmarks:
        # Ambil landmarks dari satu wajah yang terdeteksi
        face_landmarks = results.multi_face_landmarks[0]
        
        # Gambar mesh di wajah (opsional, untuk visualisasi)
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=drawing_spec,
            connection_drawing_spec=drawing_spec)

        # --- Logika Deteksi Gerakan ---
        
        # Kita gunakan titik hidung (landmark 1) sebagai acuan utama
        nose_tip = face_landmarks.landmark[1]
        nose_x = nose_tip.x
        nose_y = nose_tip.y
        
        current_time = time.time()
        
        # Reset command jika tidak ada gerakan signifikan
        command = "TENGAH"

        # 1. Deteksi Kiri / Kanan
        if nose_x < 0.5 - X_THRESHOLD:
            command = "KIRI"
        elif nose_x > 0.5 + X_THRESHOLD:
            command = "KANAN"
            
        # 2. Deteksi Atas / Bawah
        if nose_y < 0.5 - Y_THRESHOLD:
            command = "ATAS"
        elif nose_y > 0.5 + Y_THRESHOLD:
            command = "BAWAH"

        # --- Eksekusi Perintah Keyboard dengan Cooldown ---
        if command != "TENGAH" and (current_time - last_command_time > COMMAND_COOLDOWN):
            if command == "KIRI":
                pyautogui.press('left')
                print("Perintah: KIRI")
            elif command == "KANAN":
                pyautogui.press('right')
                print("Perintah: KANAN")
            elif command == "ATAS":
                pyautogui.press('up')
                print("Perintah: LOMPAT")
            elif command == "BAWAH":
                pyautogui.press('down')
                print("Perintah: BAWAH")
            
            last_command_time = current_time

    # Tampilkan status di layar
    cv2.putText(image, f"PERINTAH: {command}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Game Controller - MediaPipe Face Mesh', image)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
face_mesh.close()
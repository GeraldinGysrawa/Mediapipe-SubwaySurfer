# Import library yang diperlukan
import cv2
import mediapipe as mp

# Inisialisasi modul MediaPipe Pose dan Drawing Utils
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Buat instance dari solusi Pose
# static_image_mode=False berarti kita memproses video/stream
# min_detection_confidence adalah ambang batas keyakinan model untuk mendeteksi seseorang
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Buka akses ke webcam (kamera default biasanya di indeks 0)
cap = cv2.VideoCapture(0)

# Loop utama untuk memproses setiap frame dari webcam
while cap.isOpened():
    # Baca frame dari webcam
    success, image = cap.read()
    if not success:
        print("Tidak dapat mengakses kamera.")
        break

    # Balik gambar secara horizontal agar seperti cermin
    image = cv2.flip(image, 1)

    # Ubah format warna dari BGR (standar OpenCV) ke RGB (standar MediaPipe)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Proses gambar dengan MediaPipe Pose untuk mendapatkan hasil deteksi
    results = pose.process(image_rgb)

    # Gambar landmarks (titik-titik tubuh) dan koneksinya di atas gambar asli
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
        )

    # Tampilkan gambar yang sudah diberi anotasi
    cv2.imshow('MediaPipe Pose Detection', image)

    # Hentikan loop jika tombol 'q' ditekan
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Lepaskan sumber daya
cap.release()
cv2.destroyAllWindows()
pose.close()
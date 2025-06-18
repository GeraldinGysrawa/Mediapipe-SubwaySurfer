import cv2
import mediapipe as mp

# Inisialisasi modul MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Buka akses ke webcam
cap = cv2.VideoCapture(0)

# --- Variabel untuk Logika Deteksi Anggukan ---

# 1. Penghitung Anggukan
nod_counter = 0

# 2. Status Kepala (untuk melacak gerakan)
#    "neutral": Posisi normal
#    "down": Kepala sedang bergerak ke bawah
head_state = "neutral"

# 3. Ambang Batas (Threshold) Gerakan Sumbu Y
#    Ini untuk memastikan hanya gerakan yang signifikan yang dihitung.
#    Anda mungkin perlu menyesuaikan nilai ini.
Y_THRESHOLD = 0.02 # Artinya, kepala harus bergerak 2% dari tinggi layar

# 4. Variabel untuk menyimpan posisi Y sebelumnya
last_y = 0

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    # Balik gambar & ubah format warna
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Deteksi pose
    results = pose.process(image_rgb)

    # Gambar landmarks
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # --- Implementasi Logika Deteksi Anggukan ---
        
        # Ambil landmark hidung (indeks 0)
        nose = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
        current_y = nose.y
        
        # Inisialisasi last_y pada frame pertama
        if last_y == 0:
            last_y = current_y
        
        # LOGIKA 1: Mendeteksi gerakan ke BAWAH
        # Jika kepala sedang dalam posisi netral dan bergerak ke bawah melebihi threshold
        if head_state == "neutral" and current_y > (last_y + Y_THRESHOLD):
            # Ubah status menjadi "down"
            head_state = "down"
            
        # LOGIKA 2: Mendeteksi gerakan ke ATAS (setelah gerakan ke bawah)
        # Jika kepala sedang dalam status "down" dan sekarang bergerak ke atas
        if head_state == "down" and current_y < (last_y - Y_THRESHOLD):
            # Gerakan anggukan selesai!
            nod_counter += 1
            # Reset status kembali ke "neutral" agar bisa mendeteksi anggukan berikutnya
            head_state = "neutral"
            
        # Perbarui posisi y terakhir untuk frame berikutnya
        last_y = current_y

    # --- Tampilkan Informasi ke Layar ---
    # Latar belakang untuk teks agar mudah dibaca
    cv2.rectangle(image, (0, 0), (400, 80), (245, 117, 66), -1)
    
    # Tampilkan jumlah anggukan
    cv2.putText(image, 'JUMLAH ANGGUKAN', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(image, str(nod_counter), (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Tampilkan status kepala saat ini (untuk debugging)
    cv2.putText(image, f'Status: {head_state}', (200, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow('Detektor Anggukan Kepala - MediaPipe', image)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pose.close()
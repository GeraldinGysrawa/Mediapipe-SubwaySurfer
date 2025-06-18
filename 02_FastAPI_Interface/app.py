import cv2
import mediapipe as mp
import pyautogui
import time
import asyncio
import threading
import json
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn

# --- Konfigurasi dan Inisialisasi ---
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Inisialisasi MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# --- Variabel Global untuk state aplikasi (dibagi antar thread) ---
lock = threading.Lock()
output_frame = None
# Ubah state menjadi dictionary untuk menampung lebih banyak data
global_state = {
    "command": "TENGAH",
    "coords": {"x": 0.5, "y": 0.5},
    "fps": 0
}

# --- Logika Inti Computer Vision (dijalankan di thread terpisah) ---
def run_cv_logic():
    global output_frame, lock, global_state
    
    cap = cv2.VideoCapture(0)
    # Variabel Kalibrasi
    X_THRESHOLD = 0.1
    Y_THRESHOLD = 0.08
    COMMAND_COOLDOWN = 0.5
    last_command_time = 0
    
    # Variabel untuk hitung FPS
    start_time = time.time()
    frame_counter = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        frame_h, frame_w, _ = frame.shape
        
        # Hitung FPS
        frame_counter += 1
        elapsed_time = time.time() - start_time
        if elapsed_time > 1:
            fps = frame_counter / elapsed_time
            with lock:
                global_state["fps"] = int(fps)
            frame_counter = 0
            start_time = time.time()

        # --- VISUALISASI THRESHOLD ---
        # Buat overlay untuk menggambar bentuk transparan
        overlay = frame.copy()
        # Tentukan titik sudut dari kotak zona mati
        top_left = (int(frame_w * (0.5 - X_THRESHOLD)), int(frame_h * (0.5 - Y_THRESHOLD)))
        bottom_right = (int(frame_w * (0.5 + X_THRESHOLD)), int(frame_h * (0.5 + Y_THRESHOLD)))
        # Gambar kotak transparan
        cv2.rectangle(overlay, top_left, bottom_right, (0, 255, 0), -1) # Warna hijau, terisi
        alpha = 0.15 # Tingkat transparansi
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        # Gambar garis tepi kotak
        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2) # Garis tepi hijau


        # Proses dengan MediaPipe
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)
        
        local_command = "TENGAH"
        nose_coords = {"x": 0.5, "y": 0.5}

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            nose_tip = face_landmarks.landmark[1]
            nose_x, nose_y = nose_tip.x, nose_tip.y
            nose_coords = {"x": nose_x, "y": nose_y}

            if nose_x < 0.5 - X_THRESHOLD: local_command = "KIRI"
            elif nose_x > 0.5 + X_THRESHOLD: local_command = "KANAN"
            if nose_y < 0.5 - Y_THRESHOLD: local_command = "ATAS"
            elif nose_y > 0.5 + Y_THRESHOLD: local_command = "BAWAH"
            
            # Eksekusi perintah keyboard
            current_time = time.time()
            if local_command != "TENGAH" and (current_time - last_command_time > COMMAND_COOLDOWN):
                pyautogui.press(local_command.lower())
                print(f"Perintah dieksekusi: {local_command}")
                last_command_time = current_time
        
        # Update state global
        with lock:
            global_state["command"] = local_command
            global_state["coords"] = nose_coords
            
        # Tampilkan informasi detail di frame video
        cv2.putText(frame, f"CMD: {local_command}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Coords: ({nose_coords['x']:.2f}, {nose_coords['y']:.2f})", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"FPS: {global_state['fps']}", (frame_w - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Simpan frame untuk di-stream
        with lock:
            _, encoded_image = cv2.imencode('.jpg', frame)
            output_frame = encoded_image.tobytes()

    cap.release()

# --- Definisi Endpoint FastAPI (Tidak berubah, hanya payload WebSocket) ---

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

def generate_video_stream():
    global output_frame, lock
    while True:
        with lock:
            if output_frame is None:
                continue
            frame = output_frame
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.03)

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_video_stream(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.websocket("/ws_status")
async def websocket_status(websocket: WebSocket):
    global global_state, lock
    await websocket.accept()
    print("WebSocket client terhubung untuk status")
    try:
        while True:
            with lock:
                # Kirim seluruh state sebagai JSON
                state_to_send = json.dumps(global_state)
            await websocket.send_text(state_to_send)
            await asyncio.sleep(0.1)
    except Exception:
        print("WebSocket client status terputus")

# --- Main execution block ---
if __name__ == "__main__":
    cv_thread = threading.Thread(target=run_cv_logic, daemon=True)
    cv_thread.start()
    uvicorn.run(app, host="0.0.0.0", port=8000)
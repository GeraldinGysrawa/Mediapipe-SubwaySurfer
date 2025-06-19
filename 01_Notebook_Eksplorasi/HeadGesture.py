import cv2
import mediapipe as mp
import tkinter as tk
from tkinter import simpledialog, messagebox
from PIL import Image, ImageTk
import json
import os
import datetime
import numpy as np
import math

# Directory to save gesture images and JSON
GESTURE_DIR = 'head_gestures'
GESTURE_JSON = os.path.join(GESTURE_DIR, 'head_gestures.json')

if not os.path.exists(GESTURE_DIR):
    os.makedirs(GESTURE_DIR)

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

# Face mesh dengan parameter yang dioptimalkan untuk head pose
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

class HeadGestureCaptureApp:
    def __init__(self, master):
        self.master = master
        self.master.title('Head Gesture Capture - Subway Surfers Control')
        self.master.geometry('800x700')
        
        self.cap = cv2.VideoCapture(0)
        self.frame = None
        self.face_landmarks = None
        self.head_pose = None
        self.imgtk = None
        
        # Subway Surfers gestures
        self.subway_gestures = {
            'NEUTRAL': 'Posisi normal/netral',
            'JUMP': 'Kepala ke atas (loncat)',
            'LEFT': 'Kepala ke kiri',
            'RIGHT': 'Kepala ke kanan',
            'DUCK': 'Kepala ke bawah (optional)'
        }
        
        # Colors for each gesture
        self.gesture_colors = {
            'NEUTRAL': '#757575',   # Grey
            'JUMP': '#4caf50',      # Green
            'LEFT': '#2196f3',      # Blue  
            'RIGHT': '#ff9800',     # Orange
            'DUCK': '#9c27b0'       # Purple
        }
        
        # Calibration data
        self.calibration_data = {
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
        
        # Setup GUI
        self.setup_gui()
        
        # Start frame update
        self.update_frame()
        
    def setup_gui(self):
        # Main frame
        main_frame = tk.Frame(self.master)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Video panel
        self.panel = tk.Label(main_frame)
        self.panel.pack(pady=10)
        
        # Gesture selection
        gesture_frame = tk.Frame(main_frame)
        gesture_frame.pack(pady=10)
        
        tk.Label(gesture_frame, text="Pilih Gesture:", font=("Arial", 12, "bold")).pack()
        
        self.gesture_var = tk.StringVar(self.master)
        gesture_options = [f"{gesture} - {desc}" for gesture, desc in self.subway_gestures.items()]
        self.gesture_var.set(gesture_options[0])
        
        self.gesture_dropdown = tk.OptionMenu(gesture_frame, self.gesture_var, *gesture_options, 
                                            command=self.update_status_color)
        self.gesture_dropdown.config(font=("Arial", 10))
        self.gesture_dropdown.pack(pady=5)
        
        # Buttons frame
        buttons_frame = tk.Frame(main_frame)
        buttons_frame.pack(pady=10)
        
        # Calibration button
        self.calibrate_btn = tk.Button(buttons_frame, text='Kalibrasi Posisi Netral', 
                                     command=self.calibrate_neutral,
                                     bg='#607d8b', fg='white', font=("Arial", 11, "bold"))
        self.calibrate_btn.pack(side=tk.LEFT, padx=5)
        
        # Single capture button
        self.capture_btn = tk.Button(buttons_frame, text='Capture Gesture', 
                                   command=self.capture_gesture,
                                   bg='#4caf50', fg='white', font=("Arial", 11, "bold"))
        self.capture_btn.pack(side=tk.LEFT, padx=5)
        
        # Multiple capture button
        self.multi_capture_btn = tk.Button(buttons_frame, text='Capture 10x', 
                                         command=self.capture_multiple_gestures,
                                         bg='#2196f3', fg='white', font=("Arial", 11, "bold"))
        self.multi_capture_btn.pack(side=tk.LEFT, padx=5)
        
        # Info frame
        info_frame = tk.Frame(main_frame)
        info_frame.pack(pady=10, fill=tk.X)
        
        # Status label
        self.status = tk.Label(info_frame, text='Siap untuk capture gesture', 
                             font=("Arial", 12, "bold"), relief=tk.RAISED, padx=10, pady=5)
        self.status.pack(fill=tk.X, pady=5)
        
        # Head pose info
        self.pose_info = tk.Label(info_frame, text='Head Pose: -', 
                                font=("Arial", 10), relief=tk.SUNKEN, padx=10, pady=5)
        self.pose_info.pack(fill=tk.X, pady=2)
        
        # Gesture detection info
        self.gesture_info = tk.Label(info_frame, text='Detected Gesture: -', 
                                   font=("Arial", 10), relief=tk.SUNKEN, padx=10, pady=5)
        self.gesture_info.pack(fill=tk.X, pady=2)
        
        self.update_status_color(self.gesture_var.get())

    def calculate_head_pose(self, landmarks):
        """Calculate head pose angles (yaw, pitch, roll) from face landmarks"""
        if not landmarks:
            return None
            
        # Key landmarks for head pose calculation
        # Nose tip
        nose_tip = landmarks[1]
        
        # Left and right eye corners
        left_eye = landmarks[33]
        right_eye = landmarks[263]
        
        # Mouth corners
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

    def classify_gesture(self, head_pose):
        """Classify head pose into gesture categories"""
        if not head_pose:
            return 'UNKNOWN'
            
        # Adjust for calibrated neutral position
        yaw = head_pose['yaw'] - self.calibration_data['neutral_yaw']
        pitch = head_pose['pitch'] - self.calibration_data['neutral_pitch']
        
        thresholds = self.calibration_data['thresholds']
        
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

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.status.config(text='Camera error!', bg='red')
            self.master.after(10, self.update_frame)
            return
            
        self.frame = frame.copy()
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Convert BGR to RGB untuk proses face mesh dan display
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process face mesh pakai RGB
        results = face_mesh.process(rgb_frame)
        
        self.face_landmarks = None
        self.head_pose = None
        
        display_frame = rgb_frame.copy()  # Frame yang akan ditampilkan (RGB)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Draw face mesh (harus di BGR, jadi convert dulu)
                bgr_frame_for_draw = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
                mp_drawing.draw_landmarks(
                    bgr_frame_for_draw,
                    face_landmarks,
                    mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                )
                # Setelah digambar, convert lagi ke RGB untuk display
                display_frame = cv2.cvtColor(bgr_frame_for_draw, cv2.COLOR_BGR2RGB)
                
                # Store landmarks
                self.face_landmarks = face_landmarks.landmark
                
                # Calculate head pose
                self.head_pose = self.calculate_head_pose(self.face_landmarks)
                
                # Classify gesture
                if self.head_pose:
                    detected_gesture = self.classify_gesture(self.head_pose)
                    
                    # Update info displays
                    pose_text = f"Yaw: {self.head_pose['yaw']:.1f}°, Pitch: {self.head_pose['pitch']:.1f}°, Roll: {self.head_pose['roll']:.1f}°"
                    self.pose_info.config(text=f"Head Pose: {pose_text}")
                    
                    self.gesture_info.config(text=f"Detected Gesture: {detected_gesture}")
                    
                    # Add gesture text on frame
                    cv2.putText(display_frame, f"Gesture: {detected_gesture}", (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    # Add pose info on frame
                    cv2.putText(display_frame, f"Yaw: {self.head_pose['yaw']:.1f}", (10, 70), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    cv2.putText(display_frame, f"Pitch: {self.head_pose['pitch']:.1f}", (10, 100), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Convert frame ke PhotoImage (harus RGB)
        img = Image.fromarray(display_frame)
        self.imgtk = ImageTk.PhotoImage(image=img)
        self.panel.imgtk = self.imgtk
        self.panel.config(image=self.imgtk)
        
        # Schedule next update
        self.master.after(10, self.update_frame)

    def calibrate_neutral(self):
        """Calibrate neutral head position"""
        if not self.head_pose:
            messagebox.showerror('Error', 'Tidak ada wajah terdeteksi! Posisikan wajah di depan kamera.')
            return
            
        # Store current pose as neutral
        self.calibration_data['neutral_yaw'] = self.head_pose['yaw']
        self.calibration_data['neutral_pitch'] = self.head_pose['pitch']
        self.calibration_data['neutral_roll'] = self.head_pose['roll']
        
        # Save calibration data
        cal_file = os.path.join(GESTURE_DIR, 'calibration.json')
        with open(cal_file, 'w') as f:
            json.dump(self.calibration_data, f, indent=2)
            
        self.status.config(text='Kalibrasi berhasil! Posisi netral tersimpan.', bg='green', fg='white')

    def capture_gesture(self):
        if not self.face_landmarks or not self.head_pose:
            messagebox.showerror('Error', 'Tidak ada wajah terdeteksi!')
            return
            
        # Get selected gesture
        gesture_full = self.gesture_var.get().strip()
        gesture_name = gesture_full.split(' - ')[0]
        
        if not gesture_name:
            messagebox.showerror('Error', 'Pilih gesture terlebih dahulu!')
            return
            
        # Generate filename
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        img_filename = f'{gesture_name}_{timestamp}.png'
        img_path = os.path.join(GESTURE_DIR, img_filename)
        
        # Save face crop
        face_img = self.crop_face(self.frame)
        cv2.imwrite(img_path, face_img)
        
        # Save gesture data
        gesture_entry = {
            'name': gesture_name,
            'head_pose': self.head_pose,
            'landmarks_count': len(self.face_landmarks),
            'image_path': img_path,
            'timestamp': timestamp
        }
        
        gesture_db.append(gesture_entry)
        
        # Save to JSON
        with open(GESTURE_JSON, 'w') as f:
            json.dump(gesture_db, f, indent=2)
            
        self.status.config(text=f'Gesture "{gesture_name}" berhasil disimpan!', 
                          bg=self.gesture_colors.get(gesture_name, '#757575'), fg='white')

    def crop_face(self, frame):
        """Crop face region from frame"""
        if not self.face_landmarks:
            return frame
            
        h, w, _ = frame.shape
        
        # Get face bounding box
        x_coords = [int(lm.x * w) for lm in self.face_landmarks]
        y_coords = [int(lm.y * h) for lm in self.face_landmarks]
        
        # Perbesar margin crop
        x_min = max(min(x_coords) - 60, 0)
        x_max = min(max(x_coords) + 60, w)
        y_min = max(min(y_coords) - 80, 0)
        y_max = min(max(y_coords) + 60, h)
        
        face_crop = frame[y_min:y_max, x_min:x_max].copy()
        return face_crop

    def capture_multiple_gestures(self):
        """Capture multiple gestures automatically"""
        import time
        
        if not self.face_landmarks:
            messagebox.showerror('Error', 'Tidak ada wajah terdeteksi!')
            return
            
        gesture_full = self.gesture_var.get().strip()
        gesture_name = gesture_full.split(' - ')[0]
        
        count = 0
        target = 10
        
        messagebox.showinfo('Info', f'Akan mengcapture {target} sampel untuk gesture "{gesture_name}".\nPertahankan posisi gesture yang diinginkan!')
        
        while count < target:
            if not self.face_landmarks or not self.head_pose:
                self.status.config(text=f'Menunggu deteksi wajah... ({count}/{target})', bg='orange')
                self.master.update_idletasks()
                time.sleep(0.5)
                continue
                
            # Save gesture
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            img_filename = f'{gesture_name}_{timestamp}_{count+1}.png'
            img_path = os.path.join(GESTURE_DIR, img_filename)
            
            face_img = self.crop_face(self.frame)
            cv2.imwrite(img_path, face_img)
            
            gesture_entry = {
                'name': gesture_name,
                'head_pose': self.head_pose,
                'landmarks_count': len(self.face_landmarks),
                'image_path': img_path,
                'timestamp': timestamp,
                'batch_id': count + 1
            }
            
            gesture_db.append(gesture_entry)
            
            with open(GESTURE_JSON, 'w') as f:
                json.dump(gesture_db, f, indent=2)
                
            count += 1
            self.status.config(text=f'Captured {count}/{target} - {gesture_name}', 
                             bg=self.gesture_colors.get(gesture_name, '#757575'), fg='white')
            self.master.update_idletasks()
            time.sleep(0.5)
            
        self.status.config(text=f'Selesai! {target} sampel "{gesture_name}" tersimpan.', bg='green', fg='white')

    def update_status_color(self, gesture_text):
        """Update status color based on selected gesture"""
        gesture_name = gesture_text.split(' - ')[0]
        color = self.gesture_colors.get(gesture_name, '#757575')
        self.status.config(bg=color, fg='white')

    def __del__(self):
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()

if __name__ == '__main__':
    root = tk.Tk()
    app = HeadGestureCaptureApp(root)
    root.mainloop()
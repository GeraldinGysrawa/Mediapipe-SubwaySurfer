import pyautogui
import time

print("Tes kontrol keyboard akan dimulai dalam 5 detik...")
print("!!! SEGERA BUKA NOTEPAD ATAU TEXT EDITOR DAN KLIK DI DALAMNYA !!!")

# Memberi Anda waktu 5 detik untuk pindah jendela
time.sleep(5)

# PyAutoGUI akan mulai mengetik
pyautogui.write('Halo, ini adalah tes dari PyAutoGUI. ', interval=0.1)
pyautogui.write('Jika Anda bisa membaca ini, berarti library berfungsi.', interval=0.1)

# Menekan tombol Enter
pyautogui.press('enter')
time.sleep(1)

# Menekan tombol panah
pyautogui.press(['down', 'down', 'right', 'right', 'up', 'left'])

print("Tes selesai.")


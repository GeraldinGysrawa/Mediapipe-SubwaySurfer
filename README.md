# MediaPipe Subway Surfer Controller

Proyek ini mengimplementasikan kontrol game Subway Surfers menggunakan gerakan kepala dengan MediaPipe Face Mesh.

## Struktur Proyek

```
├── 01_Notebook_Eksplorasi/     # Jupyter notebook untuk eksplorasi dan pengembangan
├── 02_FastAPI_Interface/       # API untuk kontrol game
```

## Instalasi

1. Clone repositori ini
2. Buat virtual environment Python:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```
3. Install dependensi:
   ```bash
   pip install -r requirements.txt
   ```

## Penggunaan

1. Buka game Subway Surfers
2. Jalankan FastAPI server:
   ```bash
   cd 02_FastAPI_Interface
   uvicorn main:app --reload
   ```
3. Buka browser dan akses `http://localhost:8000`
4. Ikuti instruksi di interface untuk memulai kontrol game

## Kontrol Game

- Gerakan kepala ke kiri: Karakter bergerak ke kiri
- Gerakan kepala ke kanan: Karakter bergerak ke kanan
- Gerakan kepala ke atas: Karakter melompat
- Gerakan kepala ke bawah: Karakter bergerak ke bawah

## Kontributor

[Ketik nama anggota tim di sini] 
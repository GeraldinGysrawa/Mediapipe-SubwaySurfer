# Kontrol Game Subway Surfers Menggunakan Gerakan Kepala dengan MediaPipe

## Deskripsi Proyek

[cite_start]Proyek ini adalah sebuah sistem inovatif yang memungkinkan pengguna untuk memainkan game **Subway Surfers** menggunakan gerakan kepala sebagai input kontrol, memberikan pengalaman bermain yang *hands-free* dan interaktif. Sistem ini memanfaatkan computer vision secara *real-time* untuk mendeteksi gestur kepala dan menerjemahkannya menjadi perintah dalam game.

[cite_start]Proyek ini dikembangkan sebagai bagian dari mata kuliah Pengolahan Citra Digital (PCD) di Politeknik Negeri Bandung.

## Fitur Utama

* [cite_start]**Deteksi Wajah Real-Time**: Menggunakan MediaPipe Face Mesh untuk mendeteksi 478 *landmark* wajah dari input webcam secara *real-time*.
* [cite_start]**Ekstraksi Pose Kepala**: Mengekstraksi orientasi kepala dalam tiga sumbu (Yaw, Pitch, Roll) sebagai fitur utama.
* [cite_start]**Klasifikasi Gestur Berbasis Machine Learning**: Menggunakan model Support Vector Machine (SVM) untuk mengklasifikasikan lima gestur kepala: **Kiri, Kanan, Lompat, Menunduk, dan Netral**.
* [cite_start]**Kontrol Game Hands-Free**: Menerjemahkan prediksi gestur menjadi perintah penekanan tombol keyboard virtual (`up`, `down`, `left`, `right`) untuk mengontrol karakter game.
* [cite_start]**Antarmuka Interaktif**: Dibangun dengan Streamlit untuk memudahkan pengumpulan data, pelatihan model, dan menjalankan demo aplikasi.

## Teknologi yang Digunakan

[cite_start]Berikut adalah daftar library utama yang digunakan dalam proyek ini:

* **OpenCV**: Untuk akuisisi dan pemrosesan gambar dari webcam.
* **MediaPipe**: Untuk deteksi *landmark* wajah.
* **Scikit-learn**: Untuk melatih dan mengevaluasi model klasifikasi SVM.
* **NumPy**: Untuk operasi numerik.
* **PyAutoGUI**: Untuk simulasi penekanan tombol keyboard.
* **Streamlit**: Untuk membangun antarmuka pengguna (UI) interaktif.
* **Pandas**: Untuk manipulasi data.

## Instalasi

1.  **Clone Repositori**
    ```bash
    git clone [https://github.com/geraldingysrawa/mediapipe-subwaysurfer.git](https://github.com/geraldingysrawa/mediapipe-subwaysurfer.git)
    cd mediapipe-subwaysurfer
    ```

2.  **Buat Virtual Environment (Direkomendasikan)**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Di Windows gunakan: venv\Scripts\activate
    ```

3.  **Install Dependensi**
    Proyek ini menyediakan file `requirements.txt` untuk kemudahan instalasi. Jalankan perintah berikut:
    ```bash
    pip install -r requirements.txt
    ```

## Cara Penggunaan

Aplikasi utama dijalankan melalui antarmuka Streamlit.

### Langkah 1: Jalankan Aplikasi Streamlit

Buka terminal, arahkan ke direktori `02_Streamlit_Interface`, dan jalankan perintah berikut:

```bash
streamlit run app.py
```

Aplikasi akan terbuka secara otomatis di browser Anda.

### Langkah 2: Kumpulkan Data Gestur (Opsional, jika ingin melatih dengan data sendiri)

Jika Anda ingin membuat model yang terkalibrasi dengan gerakan kepala Anda sendiri:

1.  Buka tab **"Gesture Data Capture"** di aplikasi Streamlit.
2.  Pilih gestur yang ingin Anda rekam dari daftar (misal: `JUMP`).
3.  Klik tombol **"Capture Gesture"**. Lakukan gerakan sesuai gestur yang dipilih.
4.  Ulangi untuk semua gestur (`LEFT`, `RIGHT`, `JUMP`, `DUCK`, `NEUTRAL`) hingga Anda memiliki sampel data yang cukup.
5.  Data akan disimpan di file `head_gestures.json`.

### Langkah 3: Latih Model (Opsional, jika data baru telah dikumpulkan)

1.  Buka tab **"Train Model"** di aplikasi.
2.  Klik tombol **"Train New Model"**.
3.  Skrip akan memuat data dari `head_gestures.json`, melatih model SVM baru, dan menyimpannya sebagai `gesture_model.pkl`.

### Langkah 4: Mainkan Game!

1.  Buka game **Subway Surfers** di browser atau desktop Anda dan posisikan berdampingan dengan jendela aplikasi Streamlit.
2.  Di aplikasi Streamlit, buka tab **"Subway Surfers Controller"**.
3.  Klik tombol **"Start Webcam"** untuk memulai deteksi.
4.  Posisikan wajah Anda di depan kamera. Sistem akan mulai mendeteksi gerakan kepala Anda dan mengontrol karakter dalam game.
5.  Untuk berhenti, klik tombol **"Stop Webcam"**.

## Struktur Proyek

```
.
├── 01_Notebook_Eksplorasi/ # Berisi notebook dan skrip untuk eksplorasi awal
│   ├── subwaysurfer_controller.ipynb # Notebook Utama
│   ├── HeadGesture.py
│   └── ...
├── 02_Streamlit_Interface/      # Berisi aplikasi utama Streamlit
│   ├── app.py                   # File utama untuk menjalankan aplikasi
│   ├── gesture_capture.py       # Modul untuk menangkap data gestur
│   └── train_model_app.py       # Modul untuk melatih model
├── Dokumen Proses dan Analisis  # File laporan praktikum
├── File Presentasi.pdf          # File presentasi hasil praktikum
├── Link Youtube.txt             # Link video presentasi
├── requirements.txt             # Daftar dependensi Python
└── README.md                    # Anda sedang membacanya
```

## Penulis

Proyek ini disusun oleh mahasiswa Jurusan Teknik Komputer dan Informatika, Politeknik Negeri Bandung:

* [cite_start]**Geraldin Gysrawa** - 231511011 
* [cite_start]**Ikhsan Zuhri Al Ghifary** - 231511015 
* [cite_start]**Muhammad Harish Al-Rasyidi** - 231511020
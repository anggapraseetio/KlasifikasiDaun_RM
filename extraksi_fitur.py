import numpy as np
import os
import cv2
import pandas as pd
from skimage.feature import graycomatrix, graycoprops

# =======================================
# KONFIGURASI DATASET
# =======================================
folder_latih = "./dataset/test"
kategori = ["Sehat", "Tidak_Sehat"]
UKURAN = 128

# =======================================
# FUNGSI EKSTRAKSI FITUR (TANPA HOG & TANPA RASIO TERANG)
# =======================================
def ekstrak_fitur(gambar_bgr):

    # Kompres ulang agar konsisten
    _, buffer = cv2.imencode('.jpg', gambar_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    gambar_bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR)

    # Preprocessing
    gambar_ubah = cv2.resize(gambar_bgr, (UKURAN, UKURAN))
    gambar_gray = cv2.cvtColor(gambar_ubah, cv2.COLOR_BGR2GRAY)
    gambar_gray = cv2.GaussianBlur(gambar_gray, (3, 3), 0)
    hsv = cv2.cvtColor(gambar_ubah, cv2.COLOR_BGR2HSV)

    # ================= GLCM ==================
    glcm = graycomatrix(
        gambar_gray,
        distances=[1],
        angles=[0, np.pi/4, np.pi/2],
        symmetric=True,
        normed=True
    )

    kontras      = graycoprops(glcm, 'contrast').mean()
    homogenitas  = graycoprops(glcm, 'homogeneity').mean()
    energi       = graycoprops(glcm, 'energy').mean()
    korelasi     = graycoprops(glcm, 'correlation').mean()

    # ================= Fitur Tambahan ==================
    masker_coklat = cv2.inRange(hsv, (15, 50, 50), (35, 255, 200))
    rasio_coklat  = np.sum(masker_coklat > 0) / (UKURAN * UKURAN)

    _, mask_thr = cv2.threshold(gambar_gray, 70, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_thr = cv2.morphologyEx(mask_thr, cv2.MORPH_OPEN, kernel)

    kontur, _ = cv2.findContours(mask_thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rasio_lubang  = sum(cv2.contourArea(c) for c in kontur) / (UKURAN * UKURAN)
    jumlah_lubang = len(kontur)

    return np.array([
        kontras, homogenitas, energi, korelasi,
        rasio_coklat, rasio_lubang, jumlah_lubang
    ])

# =======================================
# PROSES EKSTRAKSI FITUR SEMUA GAMBAR
# =======================================
data_fitur = []
nama_kolom = [
    "Nama_File",
    "GLCM_Kontras",
    "GLCM_Homogenitas",
    "GLCM_Energi",
    "GLCM_Korelasi",
    "Rasio_Coklat",
    "Rasio_Lubang",
    "Jumlah_Lubang",
    "Label"
]

print("🔄 Memulai ekstraksi fitur dataset...")

for idx, nama_kategori in enumerate(kategori):

    folder = os.path.join(folder_latih, nama_kategori)
    file_gambar = [f for f in os.listdir(folder)
                   if f.lower().endswith((".jpg", ".png", ".jpeg"))]

    for file in file_gambar:

        path_gambar = os.path.join(folder, file)
        gambar = cv2.imread(path_gambar)

        fitur = ekstrak_fitur(gambar)

        # Label mengikuti kategori
        label = 1 if nama_kategori == "Sehat" else 0

        # Nama file sesuai aslinya
        fitur_final = np.concatenate(([file], fitur, [label]))

        data_fitur.append(fitur_final)

print("✅ Ekstraksi fitur selesai!")

# =======================================
# SIMPAN KE CSV
# =======================================
df = pd.DataFrame(data_fitur, columns=nama_kolom)

output_csv = "fitur_dataset_daun_test.csv"
df.to_csv(output_csv, index=False)

print(f"📁 File CSV berhasil dibuat: {output_csv}")
print("📌 Nama file 100% mengikuti nama file asli.")
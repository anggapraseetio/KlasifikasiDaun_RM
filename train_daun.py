import numpy as np
import os
import cv2
from tqdm import tqdm
from skimage.feature import hog, graycomatrix, graycoprops
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings("ignore")

folder_latih = "./dataset/train"
folder_uji = "./dataset/test"
kategori = ["Sehat", "Tidak_Sehat"]
UKURAN = 128

# FUNGSI EKSTRAKSI FITUR
def ekstrak_fitur(gambar_bgr):

    # Kompres ulang untuk konsistensi
    _, buffer = cv2.imencode('.jpg', gambar_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    gambar_bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR)

    # Preprocessing
    gambar_ubah = cv2.resize(gambar_bgr, (UKURAN, UKURAN))
    gambar_gray = cv2.cvtColor(gambar_ubah, cv2.COLOR_BGR2GRAY)
    gambar_gray = cv2.GaussianBlur(gambar_gray, (3, 3), 0)
    hsv = cv2.cvtColor(gambar_ubah, cv2.COLOR_BGR2HSV)

    # GLCM
    glcm = graycomatrix(
        gambar_gray,
        distances=[1],
        angles=[0, np.pi/4, np.pi/2],
        symmetric=True,
        normed=True
    )

    kontras = graycoprops(glcm, 'contrast').mean()
    homogenitas = graycoprops(glcm, 'homogeneity').mean()
    energi = graycoprops(glcm, 'energy').mean()
    korelasi = graycoprops(glcm, 'correlation').mean()

    # Rasio daun coklat
    masker_coklat = cv2.inRange(hsv, (15, 50, 50), (35, 255, 200))
    rasio_coklat = np.sum(masker_coklat > 0) / (UKURAN * UKURAN)

    # Deteksi lubang (area hitam)
    _, mask_thr = cv2.threshold(gambar_gray, 70, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_thr = cv2.morphologyEx(mask_thr, cv2.MORPH_OPEN, kernel)
    kontur, _ = cv2.findContours(mask_thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rasio_lubang = sum(cv2.contourArea(c) for c in kontur) / (UKURAN * UKURAN)
    jumlah_lubang = len(kontur)

    # =================
    fitur_glcm = [kontras, homogenitas, energi, korelasi]
    fitur_tambahan = [rasio_coklat, rasio_lubang, jumlah_lubang]

    # Total fitur = 7
    return np.hstack((fitur_glcm, fitur_tambahan))


# LOAD DATASET
def muat_dataset(folder_path):
    data, label = [], []
    for idx, nama_kategori in enumerate(kategori):
        path = os.path.join(folder_path, nama_kategori)
        print(f"Memuat {nama_kategori} ...")
        for file in tqdm(os.listdir(path)):
            img_path = os.path.join(path, file)
            img = cv2.imread(img_path)
            if img is None: 
                continue
            fitur = ekstrak_fitur(img)
            data.append(fitur)
            label.append(idx)
    return np.array(data), np.array(label)

print("Memuat Data Training...")
fitur_latih, label_latih = muat_dataset(folder_latih)

print("Memuat Data Testing...")
fitur_uji, label_uji = muat_dataset(folder_uji)

print(f"\nJumlah Data Train : {len(fitur_latih)} | Test : {len(fitur_uji)}")

# SCALING
skala = StandardScaler()
fitur_latih_scaled = skala.fit_transform(fitur_latih)
fitur_uji_scaled = skala.transform(fitur_uji)

# TRAINING MODEL RANDOM FOREST
model = RandomForestClassifier(
    n_estimators=800,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
model.fit(fitur_latih_scaled, label_latih)

# PREDIKSI
pred_latih = model.predict(fitur_latih_scaled)
pred_uji = model.predict(fitur_uji_scaled)

akurasi_latih = accuracy_score(label_latih, pred_latih)
akurasi_uji = accuracy_score(label_uji, pred_uji)

print("\n" + "="*60)
print("               HASIL AKURASI MODEL RANDOM FOREST")
print("="*60)
print(f"AKURASI TRAINING : {akurasi_latih*100:.2f}%")
print(f"AKURASI TESTING  : {akurasi_uji*100:.2f}%")
print(f"SELISIH          : {abs(akurasi_latih - akurasi_uji)*100:.2f}%")
print("="*60)

# Diagnosa Overfitting
print("\nDIAGNOSA MODEL:")
if akurasi_latih > 0.995 and (akurasi_latih - akurasi_uji) > 0.12:
    print("OVERFITTING BERAT")
elif (akurasi_latih - akurasi_uji) > 0.10:
    print("OVERFITTING RINGAN")
elif abs(akurasi_latih - akurasi_uji) <= 0.05:
    print("MODEL BAGUS & STABIL")
elif akurasi_uji > akurasi_latih:
    print("UNDERFITTING")
else:
    print("MODEL BAGUS")

# LAPORAN KLASIFIKASI
print("\n" + "—"*50)
print("LAPORAN KLASIFIKASI (TEST SET)")
print("—"*50)
print(classification_report(label_uji, pred_uji, target_names=kategori))

# CONFUSION MATRIX
cm = confusion_matrix(label_uji, pred_uji)
plt.figure(figsize=(7,5.5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=kategori, yticklabels=kategori,
            annot_kws={"size": 16})
plt.ylabel("Aktual")
plt.xlabel("Prediksi")
plt.title(f"Confusion Matrix (Akurasi: {akurasi_uji*100:.2f}%)")
plt.show()

# (1) HISTOGRAM SEBARAN FITUR
plt.figure(figsize=(10,5))
plt.title("Histogram Sebaran Nilai Fitur (Setelah Scaling)")
plt.hist(fitur_latih_scaled[:,0], bins=40)
plt.xlabel("Nilai Fitur (Fitur HOG pertama)")
plt.ylabel("Jumlah")
plt.show()

# (2) ROC CURVE
pred_proba = model.predict_proba(fitur_uji_scaled)[:,1]
fpr, tpr, _ = roc_curve(label_uji, pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7,5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0,1], [0,1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# (6) PRECISION–RECALL CURVE
precision, recall, _ = precision_recall_curve(label_uji, pred_proba)

plt.figure(figsize=(7,5))
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve")
plt.show()

# (7) LAPORAN METRIK RINGKAS
print("\nRINGKASAN METRIK:")
print(f"- Akurasi: {akurasi_uji*100:.2f}%")
print(f"- AUC ROC: {roc_auc:.3f}")
print(f"- Precision rata-rata: {precision.mean():.3f}")
print(f"- Recall rata-rata: {recall.mean():.3f}")

# SIMPAN MODEL
os.makedirs("./model", exist_ok=True)
joblib.dump(model, "./model/random_forest_daun_FINAL.pkl")
joblib.dump(skala, "./model/scaler_daun_FINAL.pkl")

print("\nModel & scaler berhasil disimpan!")
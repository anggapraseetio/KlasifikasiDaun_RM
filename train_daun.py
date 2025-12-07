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

def ekstrak_fitur(gambar_bgr):

    _, buffer = cv2.imencode('.jpg', gambar_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    gambar_bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR)

    gambar_ubah = cv2.resize(gambar_bgr, (UKURAN, UKURAN))
    gambar_gray = cv2.cvtColor(gambar_ubah, cv2.COLOR_BGR2GRAY)
    gambar_gray = cv2.GaussianBlur(gambar_gray, (3, 3), 0)
    hsv = cv2.cvtColor(gambar_ubah, cv2.COLOR_BGR2HSV)

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

    masker_coklat = cv2.inRange(hsv, (15, 50, 50), (35, 255, 200))
    rasio_coklat = np.sum(masker_coklat > 0) / (UKURAN * UKURAN)

    _, mask_thr = cv2.threshold(gambar_gray, 70, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_thr = cv2.morphologyEx(mask_thr, cv2.MORPH_OPEN, kernel)
    kontur, _ = cv2.findContours(mask_thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rasio_lubang = sum(cv2.contourArea(c) for c in kontur) / (UKURAN * UKURAN)
    jumlah_lubang = len(kontur)

    fitur_glcm = [kontras, homogenitas, energi, korelasi]
    fitur_tambahan = [rasio_coklat, rasio_lubang, jumlah_lubang]

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


print("\n" + "="*60)
print("               TRAINING MODEL RANDOM FOREST")
print("="*60)

model = RandomForestClassifier(
    n_estimators=300,         
    max_depth=15,           
    min_samples_split=10,      
    min_samples_leaf=4,       
    max_features='sqrt',     
    random_state=42,
    bootstrap=False,
    n_jobs=-1
)

print("Training model...")
model.fit(fitur_latih_scaled, label_latih)

# PREDIKSI
pred_latih = model.predict(fitur_latih_scaled)
pred_uji = model.predict(fitur_uji_scaled)

akurasi_latih = accuracy_score(label_latih, pred_latih)
akurasi_uji = accuracy_score(label_uji, pred_uji)

print("\n" + "="*60)
print("               HASIL AKURASI MODEL RANDOM FOREST")
print("="*60)
print(f"AKURASI TRAINING  : {akurasi_latih*100:.2f}%")
print(f"AKURASI TESTING   : {akurasi_uji*100:.2f}%")
print(f"SELISIH (Train-Test): {abs(akurasi_latih - akurasi_uji)*100:.2f}%")
print("="*60)

# Diagnosa Model
print("\nDIAGNOSA MODEL:")
selisih = akurasi_latih - akurasi_uji

if akurasi_latih > 0.995 and selisih > 0.12:
    print("⚠️  OVERFITTING BERAT - Model terlalu menghafal data training")
elif selisih > 0.10:
    print("⚠️  OVERFITTING RINGAN - Perlu tuning parameter lebih lanjut")
elif abs(selisih) <= 0.05:
    print("✅ MODEL BAGUS & STABIL - Generalisasi sangat baik!")
elif akurasi_uji > akurasi_latih:
    print("⚠️  UNDERFITTING - Model terlalu sederhana")
else:
    print("✅ MODEL BAGUS - Performa seimbang")

# FEATURE IMPORTANCE
print("\n" + "="*60)
print("               FEATURE IMPORTANCE")
print("="*60)
feature_names = ['Kontras', 'Homogenitas', 'Energi', 'Korelasi', 
                 'Rasio Coklat', 'Rasio Lubang', 'Jumlah Lubang']
importances = model.feature_importances_

for name, importance in zip(feature_names, importances):
    print(f"{name:20s}: {importance:.4f}")

# LAPORAN KLASIFIKASI
print("\n" + "—"*60)
print("LAPORAN KLASIFIKASI (TEST SET)")
print("—"*60)
print(classification_report(label_uji, pred_uji, target_names=kategori))

# CONFUSION MATRIX
cm = confusion_matrix(label_uji, pred_uji)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=kategori, yticklabels=kategori,
            annot_kws={"size": 18})
plt.ylabel("Aktual", fontsize=12)
plt.xlabel("Prediksi", fontsize=12)
plt.title(f"Confusion Matrix\nAkurasi Test: {akurasi_uji*100:.2f}%", 
          fontsize=13)
plt.tight_layout()
plt.show()

# FEATURE IMPORTANCE CHART
plt.figure(figsize=(10,6))
indices = np.argsort(importances)[::-1]
plt.bar(range(len(importances)), importances[indices])
plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
plt.xlabel("Fitur")
plt.ylabel("Importance")
plt.title("Feature Importance - Random Forest")
plt.tight_layout()
plt.show()

# ROC CURVE
pred_proba = model.predict_proba(fitur_uji_scaled)[:,1]
fpr, tpr, _ = roc_curve(label_uji, pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, linewidth=2, label=f"Random Forest (AUC = {roc_auc:.3f})")
plt.plot([0,1], [0,1], linestyle="--", color='gray', label='Random Classifier')
plt.xlabel("False Positive Rate", fontsize=11)
plt.ylabel("True Positive Rate", fontsize=11)
plt.title("ROC Curve", fontsize=13)
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# PRECISION–RECALL CURVE
precision, recall, _ = precision_recall_curve(label_uji, pred_proba)
pr_auc = auc(recall, precision)

plt.figure(figsize=(8,6))
plt.plot(recall, precision, linewidth=2, label=f'PR AUC = {pr_auc:.3f}')
plt.xlabel("Recall", fontsize=11)
plt.ylabel("Precision", fontsize=11)
plt.title("Precision–Recall Curve", fontsize=13)
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("               RINGKASAN METRIK EVALUASI")
print("="*60)
print(f"Akurasi Training   : {akurasi_latih*100:.2f}%")
print(f"Akurasi Testing    : {akurasi_uji*100:.2f}%")
print(f"SELISIH            : {abs(akurasi_latih - akurasi_uji)*100:.2f}%")
print(f"AUC ROC            : {roc_auc:.3f}")
print(f"AUC PR             : {pr_auc:.3f}")
print(f"Precision rata-rata: {precision.mean():.3f}")
print(f"Recall rata-rata   : {recall.mean():.3f}")
print("="*60)

# SIMPAN MODEL
os.makedirs("./model", exist_ok=True)
joblib.dump(model, "./model/random_forest_daun_FINAL.pkl")
joblib.dump(skala, "./model/scaler_daun_FINAL.pkl")

print("\n✅ Model & scaler berhasil disimpan!")
print("   - random_forest_daun_FINAL.pkl")
print("   - scaler_daun_FINAL.pkl")
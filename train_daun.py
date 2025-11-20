import numpy as np
import os
import cv2
from tqdm import tqdm
from skimage.feature import hog, graycomatrix, graycoprops
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings("ignore")

train_dir = "./dataset/train"
test_dir  = "./dataset/test"
categories = ["Sehat", "Tidak_Sehat"]
IMG_SIZE = 128

# ekstraksi fitur (HOG + GLCM)
def extract_features(img_bgr):
    _, buffer = cv2.imencode('.jpg', img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    img_bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    
    img_resized = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.GaussianBlur(img_gray, (3, 3), 0)
    hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)

    hog_features, _ = hog(img_gray, orientations=9, pixels_per_cell=(16,16),
                          cells_per_block=(2,2), block_norm='L2-Hys',
                          visualize=True, feature_vector=True)

    glcm = graycomatrix(img_gray, distances=[1], angles=[0, np.pi/4, np.pi/2],
                        symmetric=True, normed=True)
    contrast     = graycoprops(glcm, 'contrast').mean()
    homogeneity  = graycoprops(glcm, 'homogeneity').mean()
    energy       = graycoprops(glcm, 'energy').mean()
    correlation  = graycoprops(glcm, 'correlation').mean()

    mask_brown = cv2.inRange(hsv, (15, 50, 50), (35, 255, 200))
    brown_ratio = np.sum(mask_brown > 0) / (IMG_SIZE * IMG_SIZE)

    _, thresh = cv2.threshold(img_gray, 70, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hole_area_ratio = sum(cv2.contourArea(c) for c in contours) / (IMG_SIZE * IMG_SIZE)
    num_holes = len(contours)

    bright_ratio = np.sum(img_gray > 190) / (IMG_SIZE * IMG_SIZE)

    extra_features = [brown_ratio, hole_area_ratio, num_holes, bright_ratio]
    glcm_features  = [contrast, homogeneity, energy, correlation]

    return np.hstack((hog_features, glcm_features, extra_features))

# load dataset
def load_dataset(folder_path):
    data, labels = [], []
    for idx, category in enumerate(categories):
        path = os.path.join(folder_path, category)
        print(f"Loading {category} ...")
        for file in tqdm(os.listdir(path)):
            img_path = os.path.join(path, file)
            img = cv2.imread(img_path)
            if img is None: continue
            features = extract_features(img)
            data.append(features)
            labels.append(idx)
    return np.array(data), np.array(labels)

# load data
print("Memuat data training...")
X_train, y_train = load_dataset(train_dir)
print("Memuat data testing...")
X_test,  y_test  = load_dataset(test_dir)

print(f"\nJumlah Train : {len(X_train)} | Test: {len(X_test)}")

# scalling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# training
model = RandomForestClassifier(
    n_estimators=800,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
model.fit(X_train_scaled, y_train)

# prediksi
y_pred_train = model.predict(X_train_scaled)
y_pred_test  = model.predict(X_test_scaled)

acc_train = accuracy_score(y_train, y_pred_train)
acc_test  = accuracy_score(y_test,  y_pred_test)

# cek overfitting
print("\n" + "="*60)
print("                 HASIL AKURASI")
print("="*60)
print(f"AKURASI TRAINING : {acc_train*100:.2f}%")
print(f"AKURASI TESTING  : {acc_test*100:.2f}%")
print(f"SELISIH          : {abs(acc_train - acc_test)*100:.2f}%")
print("="*60)

print("\nDIAGNOSA MODEL:")
if acc_train > 0.995 and (acc_train - acc_test) > 0.12:
    print("OVERFITTING BERAT – Model menghafal data training!")
elif (acc_train - acc_test) > 0.10:
    print("OVERFITTING RINGAN – Masih bisa diperbaiki")
elif abs(acc_train - acc_test) <= 0.05:
    print("MODEL SANGAT BAGUS & STABIL – Generalisasi sangat baik")
elif acc_test > acc_train:
    print("UNDERFITTING – Model terlalu sederhana")
else:
    print("MODEL BAGUS – Selisih wajar (<10%), tidak overfitting")

print("\n" + "—"*50)
print("CLASSIFICATION REPORT (Test Set)")
print("—"*50)
print(classification_report(y_test, y_pred_test, target_names=categories))

# matrix
cm = confusion_matrix(y_test, y_pred_test)
plt.figure(figsize=(7,5.5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=categories, yticklabels=categories,
            annot_kws={"size": 16})
plt.ylabel('Actual', fontsize=14)
plt.xlabel('Predicted', fontsize=14)
plt.title(f'Confusion Matrix\nAccuracy: {acc_test*100:.2f}%', fontsize=16)
plt.show()

# simpan model
os.makedirs("./model", exist_ok=True)
joblib.dump(model, "./model/random_forest_daun_FINAL.pkl")
joblib.dump(scaler, "./model/scaler_daun_FINAL.pkl")

print("\nModel dan scaler berhasil disimpan!")
print("→ ./model/random_forest_daun_FINAL.pkl")
print("→ ./model/scaler_daun_FINAL.pkl")
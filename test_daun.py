import cv2
import numpy as np
import joblib
from skimage.feature import graycomatrix, graycoprops
from rembg import remove
from PIL import Image
import io
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

model  = joblib.load("./model/random_forest_daun_FINAL.pkl")
scaler = joblib.load("./model/scaler_daun_FINAL.pkl")

IMG_SIZE = 128
categories = ["Sehat", "Tidak_Sehat"]
test_image_path = "./dataset/test/test5.jpg"

# =====================================================
# EXTRACT FEATURES
# =====================================================
def extract_features(img_bgr):
    # konsistensi kompresi
    _, buffer = cv2.imencode('.jpg', img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    img_bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    
    img_resized = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.GaussianBlur(img_gray, (3, 3), 0)
    hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)

    # ============== GLCM ==============
    glcm = graycomatrix(
        img_gray,
        distances=[1],
        angles=[0, np.pi/4, np.pi/2],
        symmetric=True,
        normed=True
    )
    contrast     = graycoprops(glcm, 'contrast').mean()
    homogeneity  = graycoprops(glcm, 'homogeneity').mean()
    energy       = graycoprops(glcm, 'energy').mean()
    correlation  = graycoprops(glcm, 'correlation').mean()

    # ============ Fitur Tambahan ============
    mask_brown = cv2.inRange(hsv, (15, 50, 50), (35, 255, 200))
    brown_ratio = np.sum(mask_brown > 0) / (IMG_SIZE * IMG_SIZE)

    _, thresh = cv2.threshold(img_gray, 70, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    hole_area_ratio = sum(cv2.contourArea(c) for c in contours) / (IMG_SIZE * IMG_SIZE)
    num_holes = len(contours)

    # fitur final (7 fitur)
    glcm_features  = [contrast, homogeneity, energy, correlation]
    extra_features = [brown_ratio, hole_area_ratio, num_holes]

    return np.hstack((glcm_features, extra_features))

print("Memproses gambar (remove background dengan aman)...")

# ==================================================================================
# PROSES GAMBAR → REMOVE BACKGROUND → SIAP EXTRACT FITUR
# ==================================================================================

img_pil = Image.open(test_image_path).convert("RGB")
width, height = img_pil.size

if max(width, height) > 1200: 
    ratio = 1200 / max(width, height)
    new_size = (int(width * ratio), int(height * ratio))
    img_pil = img_pil.resize(new_size, Image.LANCZOS)

img_bytes = io.BytesIO()
img_pil.save(img_bytes, format="PNG")
img_bytes.seek(0)

output_data = remove(
    img_bytes.read(),
    alpha_matting=False,
    alpha_matting_erode_size=7,
    background_threshold=50
)

img_rgba = Image.open(io.BytesIO(output_data)).convert("RGBA")
white_bg = Image.new("RGBA", img_rgba.size, (255, 255, 255, 255))
img_clean = Image.alpha_composite(white_bg, img_rgba).convert("RGB")

buffer = io.BytesIO()
img_clean.save(buffer, format="JPEG", quality=90, optimize=True, subsampling=0)
buffer.seek(0)
img_final = Image.open(buffer).convert("RGB")
img_cv = cv2.cvtColor(np.array(img_final), cv2.COLOR_RGB2BGR)

# =====================================================
# PREDIKSI
# =====================================================
features = extract_features(img_cv)
features_scaled = scaler.transform([features])

prediction = model.predict(features_scaled)[0]
prob = model.predict_proba(features_scaled)[0]

# =====================================================
# TAMPILKAN HASIL
# =====================================================
plt.figure(figsize=(10, 8))
plt.imshow(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
plt.title(
    f"PREDIKSI: {categories[prediction].upper()}\n"
    f"Sehat: {prob[0]*100:.2f}% | Tidak Sehat: {prob[1]*100:.2f}%",
    fontsize=20, fontweight='bold',
    pad=30,
    color='darkgreen' if prediction == 0 else 'darkred'
)
plt.axis('off')
plt.tight_layout()
plt.show()

print("="*55)
print(f"Prediksi                   : {categories[prediction]}")
print(f"Probabilitas Sehat         : {prob[0]*100:.2f}%")
print(f"Probabilitas Tidak Sehat   : {prob[1]*100:.2f}%")
print("="*55)

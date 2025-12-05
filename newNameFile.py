import os

# ================================
# KONFIGURASI FOLDER DATASET
# ================================
folder_dataset = "./dataset/test"
kategori = ["Sehat", "Tidak_Sehat"]

# ================================
# PROSES RENAME OTOMATIS
# ================================
for nama_kategori in kategori:
    folder = os.path.join(folder_dataset, nama_kategori)

    # Ambil semua file gambar
    file_gambar = sorted([
        f for f in os.listdir(folder)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    # Rename satu per satu
    for i, file in enumerate(file_gambar, start=1):
        ekstensi = file.split('.')[-1]
        nama_baru = f"{nama_kategori}{i}.{ekstensi}"

        path_lama = os.path.join(folder, file)
        path_baru = os.path.join(folder, nama_baru)

        os.rename(path_lama, path_baru)
        print(f"Rename: {file} → {nama_baru}")

print("✔️ Semua nama file berhasil diganti!")
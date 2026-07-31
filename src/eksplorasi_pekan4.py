"""
Eksplorasi Pekan 4 - InsightFace (versi lengkap, jalan di lokal)
KP: Implementasi Face Recognition pada Aplikasi Absensi
PT Samudra Aplikasi Indonesia

Cakupan pekan 4: install InsightFace, jalankan detection + embedding,
pahami struktur API-nya (atribut wajah, parameter model, perbandingan
ukuran model), dan uji similarity matching.

Cara pakai:
1. pip install -r requirements.txt
2. python eksplorasi_pekan4.py
   -> Bagian 1-6 langsung bisa jalan tanpa foto sendiri (pakai foto contoh
      bawaan library InsightFace).
   -> Bagian 7 (opsional) pakai foto kamu sendiri: taruh di folder foto/,
      lalu ganti nama file di variabel NAMA_FOTO_SENDIRI di bawah.
3. Hasil visualisasi tersimpan di folder hasil/, embedding tersimpan di data/
"""

import os
import time
import json
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis

# ── Setup folder kerja ──────────────────────────────────────────────
FOLDER_FOTO = "foto"
FOLDER_DATA = "data"
FOLDER_HASIL = "hasil"
for folder in [FOLDER_FOTO, FOLDER_DATA, FOLDER_HASIL]:
    os.makedirs(folder, exist_ok=True)


def cosine_sim(a, b):
    """Cosine similarity. Karena embedding InsightFace sudah dinormalisasi,
    ini secara matematis sama dengan dot product biasa."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return float(np.dot(a, b))


def gambar_kotak_deteksi(img, faces, path_output):
    """Gambar bbox + 5 landmark di atas foto, simpan ke disk lokal
    supaya bisa dicek visual apakah deteksinya pas."""
    out = img.copy()
    for i, f in enumerate(faces):
        box = f.bbox.astype(int)
        cv2.rectangle(out, (box[0], box[1]), (box[2], box[3]), (0, 220, 130), 3)
        for (x, y) in f.kps.astype(int):
            cv2.circle(out, (x, y), 3, (40, 80, 240), -1)
        cv2.putText(out, f"#{i+1}", (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 220, 130), 2)
    cv2.imwrite(path_output, out)
    print(f"  Hasil visualisasi disimpan: {path_output}")


# =====================================================================
# BAGIAN 1 — Inisialisasi model & pahami struktur API
# =====================================================================
print("=" * 65)
print("BAGIAN 1: INISIALISASI MODEL (buffalo_l)")
print("=" * 65)
print("""
Struktur API InsightFace secara singkat:
  FaceAnalysis(name=...)  -> pilih bundel model (buffalo_l / buffalo_s / dst)
  .prepare(ctx_id, det_size)
       ctx_id  : -1 = CPU, 0 (atau angka lain) = index GPU
       det_size: resolusi input untuk detector, mempengaruhi kecepatan & akurasi
  .get(img)  -> jalankan seluruh pipeline (detection+alignment+embedding),
                mengembalikan list objek "Face"

Tiap objek Face berisi:
  .bbox        koordinat kotak wajah [x1, y1, x2, y2]
  .kps         5 titik landmark (mata kiri-kanan, hidung, sudut mulut)
  .det_score   confidence deteksi (0-1)
  .embedding   vector 512 angka (fitur wajah)
  .sex, .age   estimasi gender & umur (dari model genderage.onnx)
""")

app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=-1, det_size=(640, 640))
print("Model buffalo_l siap dipakai.\n")


# =====================================================================
# BAGIAN 2 — Deteksi pakai foto contoh bawaan library (langsung bisa jalan)
# =====================================================================
print("=" * 65)
print("BAGIAN 2: DETEKSI MULTI-WAJAH (foto contoh bawaan library)")
print("=" * 65)

path_contoh = os.path.join(
    os.path.dirname(insightface.__file__), "data", "images", "t1.jpg"
)
img_contoh = cv2.imread(path_contoh)
faces_contoh = app.get(img_contoh)

print(f"Jumlah wajah terdeteksi: {len(faces_contoh)}\n")
for i, f in enumerate(faces_contoh):
    print(f"Wajah #{i + 1}")
    print(f"  bbox            : {f.bbox.astype(int)}")
    print(f"  det_score       : {f.det_score:.4f}")
    print(f"  gender          : {'Pria' if f.sex == 'M' else 'Wanita'}")
    print(f"  estimasi umur   : {f.age}")
    print(f"  embedding shape : {f.embedding.shape}")
    print()

gambar_kotak_deteksi(img_contoh, faces_contoh,
                      os.path.join(FOLDER_HASIL, "deteksi_contoh.jpg"))
print()


# =====================================================================
# BAGIAN 3 — Uji ketahanan similarity terhadap noise (pencahayaan)
# =====================================================================
print("=" * 65)
print("BAGIAN 3: UJI SIMILARITY - TAHAN NOISE VS BEDA ORANG")
print("=" * 65)

if len(faces_contoh) >= 2:
    # versi gelap dari foto yang sama (simulasi kondisi pencahayaan beda)
    img_gelap = (img_contoh.astype(np.float32) * 0.45).astype(np.uint8)
    faces_gelap = app.get(img_gelap)

    def bbox_center(b):
        return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)

    c1 = bbox_center(faces_contoh[0].bbox.astype(int))
    wajah_gelap_sesuai = min(
        faces_gelap,
        key=lambda f: np.hypot(*np.subtract(bbox_center(f.bbox.astype(int)), c1)),
    )

    sim_sama_orang = cosine_sim(faces_contoh[0].embedding, wajah_gelap_sesuai.embedding)
    sim_beda_orang = cosine_sim(faces_contoh[0].embedding, faces_contoh[1].embedding)

    print(f"  Wajah #1 asli vs Wajah #1 (dibuat 55% lebih gelap): "
          f"similarity = {sim_sama_orang:.4f}  -> "
          f"{'MATCH (orang sama)' if sim_sama_orang > 0.5 else 'TIDAK match'}")
    print(f"  Wajah #1 vs Wajah #2 (dua orang beda)             : "
          f"similarity = {sim_beda_orang:.4f}  -> "
          f"{'MATCH' if sim_beda_orang > 0.5 else 'TIDAK match (orang beda, sesuai ekspektasi)'}")
    print()


# =====================================================================
# BAGIAN 4 — Efek det_size terhadap kecepatan vs akurasi
# =====================================================================
print("=" * 65)
print("BAGIAN 4: PERBANDINGAN det_size (KECEPATAN VS JUMLAH TERDETEKSI)")
print("=" * 65)

for size in [(320, 320), (640, 640), (1024, 1024)]:
    app_test = FaceAnalysis(name="buffalo_l")
    app_test.prepare(ctx_id=-1, det_size=size)
    start = time.time()
    faces_test = app_test.get(img_contoh)
    durasi = time.time() - start
    print(f"  det_size={size}: {len(faces_test)} wajah, {durasi:.2f} detik")
print()


# =====================================================================
# BAGIAN 5 — Perbandingan model buffalo_l (besar) vs buffalo_s (kecil)
# =====================================================================
print("=" * 65)
print("BAGIAN 5: PERBANDINGAN MODEL buffalo_l VS buffalo_s")
print("=" * 65)
print("Catatan: buffalo_s otomatis ter-download saat pertama kali dipakai (~15-30 MB).\n")

try:
    app_small = FaceAnalysis(name="buffalo_s")
    app_small.prepare(ctx_id=-1, det_size=(640, 640))

    start = time.time()
    faces_l = app.get(img_contoh)
    durasi_l = time.time() - start

    start = time.time()
    faces_s = app_small.get(img_contoh)
    durasi_s = time.time() - start

    print(f"  buffalo_l : {len(faces_l)} wajah terdeteksi, {durasi_l:.2f} detik, "
          f"embedding dim={faces_l[0].embedding.shape[0]}")
    print(f"  buffalo_s : {len(faces_s)} wajah terdeteksi, {durasi_s:.2f} detik, "
          f"embedding dim={faces_s[0].embedding.shape[0]}")
    print("  PENTING: embedding dari 2 model beda TIDAK BISA dibandingkan langsung "
          "satu sama lain (skala angkanya beda). Pilih 1 model, pakai konsisten "
          "di seluruh sistem.")
except Exception as e:
    print(f"  [!] buffalo_s gagal dimuat ({e}). Lewati bagian ini, tidak wajib.")
print()


# =====================================================================
# BAGIAN 6 — Simpan & load embedding ke disk lokal (permanen)
# =====================================================================
print("=" * 65)
print("BAGIAN 6: SIMPAN & LOAD EMBEDDING KE DISK LOKAL")
print("=" * 65)

database_path = os.path.join(FOLDER_DATA, "database.json")
database = {}
if os.path.exists(database_path):
    with open(database_path, "r") as f:
        database = json.load(f)

if len(faces_contoh) > 0:
    database["contoh_wajah_1"] = faces_contoh[0].embedding.tolist()
    with open(database_path, "w") as f:
        json.dump(database, f)
    print(f"  Embedding disimpan ke: {database_path}")

    with open(database_path, "r") as f:
        database_loaded = json.load(f)
    embedding_loaded = np.array(database_loaded["contoh_wajah_1"])
    print(f"  Berhasil di-load ulang, shape: {embedding_loaded.shape}")
    print("  (File ini permanen di disk, tidak akan hilang seperti session Colab)")
print()


# =====================================================================
# BAGIAN 7 (OPSIONAL) — Uji pakai foto kamu sendiri
# =====================================================================
print("=" * 65)
print("BAGIAN 7 (OPSIONAL): UJI PAKAI FOTO SENDIRI")
print("=" * 65)

NAMA_FOTO_SENDIRI = "vian.jpg"  # <-- GANTI sesuai nama file di folder foto/
path_sendiri = os.path.join(FOLDER_FOTO, NAMA_FOTO_SENDIRI)

if os.path.exists(path_sendiri):
    img_sendiri = cv2.imread(path_sendiri)
    faces_sendiri = app.get(img_sendiri)
    print(f"  Jumlah wajah terdeteksi: {len(faces_sendiri)}")
    if len(faces_sendiri) > 0:
        gambar_kotak_deteksi(
            img_sendiri, faces_sendiri,
            os.path.join(FOLDER_HASIL, f"deteksi_{NAMA_FOTO_SENDIRI}")
        )
        database["diri_sendiri"] = faces_sendiri[0].embedding.tolist()
        with open(database_path, "w") as f:
            json.dump(database, f)
        print(f"  Embedding kamu ikut disimpan ke {database_path}")
else:
    print(f"  [i] Lewati bagian ini — taruh foto di '{path_sendiri}' "
          f"lalu jalankan ulang script kalau mau coba dengan foto sendiri.")

print("\nSelesai. Cek folder 'hasil/' untuk gambar hasil deteksi, "
      "dan 'data/database.json' untuk embedding tersimpan.")

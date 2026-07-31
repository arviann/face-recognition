# Face Recognition untuk Aplikasi Absensi

Kerja Praktik — PT Samudra Aplikasi Indonesia
Implementasi teknologi Face Recognition (InsightFace: RetinaFace + ArcFace)
sebagai lapisan keamanan tambahan pada aplikasi absensi karyawan.

> Status: eksplorasi & prototype lokal (belum terintegrasi ke backend/mobile app).

## Struktur folder

```
face-recognition-kp/
├── src/
│   └── eksplorasi_pekan4.py   # script eksplorasi pekan 4 (lihat cakupan di bawah)
├── foto/                      # taruh foto uji sendiri di sini (gitignored)
├── data/                      # embedding tersimpan di sini (gitignored)
├── hasil/                     # gambar hasil visualisasi deteksi (gitignored)
├── requirements.txt
├── .gitignore
├── CLAUDE.md                  # aturan kerja untuk Claude Code di repo ini
└── README.md
```

## Cara pakai

1. Install dependency:
   ```bash
   pip install -r requirements.txt
   ```
2. Jalankan langsung (Bagian 1-6 otomatis pakai foto contoh bawaan
   library InsightFace, jadi bisa langsung dicoba tanpa foto sendiri):
   ```bash
   python src/eksplorasi_pekan4.py
   ```
3. (Opsional) Untuk Bagian 7: taruh foto sendiri di folder `foto/`, edit
   variabel `NAMA_FOTO_SENDIRI` di script, jalankan ulang.

Model `buffalo_l` (~275 MB) dan `buffalo_s` (~125 MB) otomatis ter-download
saat pertama kali dijalankan, disimpan permanen di `~/.insightface/models/`
(tidak perlu download ulang di run berikutnya).

## Cakupan eksplorasi pekan 4 (isi script)

1. Inisialisasi model & penjelasan struktur API (`FaceAnalysis`, `.prepare()`, `.get()`)
2. Deteksi multi-wajah + atribut lengkap (bbox, det_score, gender, umur, embedding)
3. Uji ketahanan similarity: foto sama (dibuat gelap) vs foto beda orang
4. Perbandingan kecepatan `det_size` (320 / 640 / 1024)
5. Perbandingan model `buffalo_l` vs `buffalo_s` (akurasi vs kecepatan)
6. Simpan & load embedding ke disk lokal secara permanen
7. (Opsional) Uji pakai foto sendiri + visualisasi bbox/landmark tersimpan ke `hasil/`

## Catatan penting

- Model pretrained InsightFace berlisensi **non-commercial / academic research
  only**. Untuk penggunaan produksi (bukan riset/KP), perlu mengurus lisensi
  komersial ke InsightFace.
- Foto wajah dan file embedding adalah data biometrik — **jangan pernah
  di-commit ke git**, sudah diatur lewat `.gitignore`.

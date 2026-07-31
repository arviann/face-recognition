# CLAUDE.md

Instruksi ini berlaku untuk Claude (Claude Code) setiap kali bekerja di repo ini.

## Tentang proyek ini

Kerja Praktik: implementasi teknologi Face Recognition pada aplikasi absensi
karyawan, menggunakan InsightFace (RetinaFace + ArcFace). Tahap saat ini masih
eksplorasi & prototype lokal — belum terintegrasi ke backend/mobile app
perusahaan.

## Aturan Git — WAJIB DIPATUHI

- **Claude TIDAK BOLEH menjalankan `git commit` dalam kondisi apa pun.**
- **Claude TIDAK BOLEH menjalankan `git push` dalam kondisi apa pun.**
- **Claude TIDAK BOLEH menjalankan `git add` untuk staging perubahan.**
- Claude hanya boleh **mengedit, membuat, atau menghapus file kode** sesuai
  permintaan di working directory.
- Setelah Claude selesai mengedit, **user sendiri yang akan review perubahan
  dan melakukan commit/push secara manual.**
- Kalau diminta "commit ini" atau "push ke GitHub", Claude cukup **menyiapkan
  perubahan filenya saja**, lalu jelaskan ke user command git apa yang perlu
  dijalankan sendiri — jangan dieksekusi otomatis.
- Claude boleh menjalankan `git status`, `git diff`, `git log` (read-only,
  untuk keperluan cek konteks), tapi tidak untuk operasi yang mengubah
  riwayat/state repo.

## Data sensitif — JANGAN PERNAH disertakan ke commit

- Folder `foto/` berisi foto wajah asli (data biometrik). **Tidak boleh
  dihapus dari `.gitignore`, tidak boleh disarankan untuk di-commit.**
- File `data/database.json` (dan file `.npy` embedding lain) berisi hasil
  ekstraksi wajah. **Sama sensitifnya dengan foto asli** — tidak boleh
  disarankan untuk di-commit meski formatnya "cuma angka".
- Kalau user secara eksplisit minta menghapus `foto/` atau `data/*.json` dari
  `.gitignore`, Claude boleh membantu, tapi **ingatkan dulu risikonya**
  (data biometrik karyawan/pribadi bocor ke riwayat git, apalagi kalau repo
  publik).

## Konvensi kode

- Komentar & pesan `print()` boleh campur Bahasa Indonesia/Inggris, ikuti
  gaya yang sudah ada di file terkait — jangan diseragamkan paksa.
- Model pretrained InsightFace (`buffalo_l`, dst) punya lisensi
  non-commercial/academic — jangan tambahkan kode yang mengarah ke deployment
  produksi/komersial tanpa user menyebutkan lisensi komersial sudah diurus.
- Struktur folder:
  - `src/` — script Python
  - `foto/` — foto uji (gitignored)
  - `data/` — hasil embedding/database lokal (gitignored)

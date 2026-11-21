# Validasi training ulang Tabular Q-Learning

## Reproduksi dan sumber

- Kode acuan `5c4cf33c2e045245be1cf49340a5a5e7eb27f5fa` diekstrak menggunakan `git archive`. Branch kerja, perubahan pengguna, algoritma, dan data sumber terdahulu tidak diubah.
- Run baru `20261121T125626.506320742Z`. Konfigurasi sengaja menggunakan 36 episode dan kuota eksplorasi 50% agar eksplorasi maupun uji kecepatan dapat diukur dalam satu run. Reward, sensor, aksi, state, dan pembaruan Q tetap mengikuti kode acuan.
- Perintah dari direktori snapshot:

```bash
/home/rafli/Project/RL-pathfinding/.venv/bin/python tools/train_lidar_q.py \
  --episodes 36 --exploration-fraction 0.5 --timeout 300 \
  --evaluation-runs 12 --seed 42 --no-render \
  --run-dir /home/rafli/Project/RL-pathfinding/reports/lidar_q_learning/runs/20261121T125626.506320742Z
```

## Pengujian yang dijalankan pada checkpoint baru

| Uji | Hasil |
|---|---|
| Batas diskretisasi termasuk ambang LiDAR, lima sektor, enam aksi | Lulus |
| Penyaringan tindakan, fallback, aksi maksimum yang dilarang, dan visits | Lulus |
| Update Bellman masked, terminal tanpa bootstrap, dan penanda timeout | Lulus |
| Penalti kontak, lima langkah pemulihan, serta kelanjutan episode | Lulus |
| Checkpoint versi 9, signature arena, seluruh kode state, Q finite, dan roundtrip save/load | Lulus |
| Rekonsiliasi training.csv, summary, routes, evaluasi, jadwal epsilon, serta minimum waktu di antara percobaan sukses | Lulus |
| Replay greedy tanpa menambah visits dan penghapusan loop target | Lulus |

Ketujuh pengujian terisolasi dijalankan ulang terhadap artefak terbaru. Audit sebelumnya atas tes bawaan snapshot menghasilkan 5 lulus dan 1 gagal. Tes yang gagal memakai `models/q_learning.json` lama, dengan signature arena yang tidak cocok. Temuan itu tidak diubah dan bukan kegagalan checkpoint LiDAR versi 9 yang diuji ulang di sini.

## Rekonsiliasi fase dan rekor

- Training menyelesaikan 36 episode. Episode 1–18 berada dalam fase eksplorasi. Fase ini menghasilkan lima kandidat. Episode 19–36 berada dalam fase uji kecepatan.
- Seluruh 18 percobaan uji kecepatan mencapai goal. Waktu minimum **113,60 detik simulasi** berasal dari rute 1, episode 20, 710 langkah, satu tabrakan. Angka sama pada `training.csv`, `routes.json`, dan `summary.json`.
- Waktu minimum dengan nol tabrakan adalah **119,36 detik simulasi**, rute 1 pada episode 25. Filter keselamatan dan hasil kontak tetap dibahas terpisah dari rekor waktu mutlak pada run.
- Uji greedy terpisah mencatat 9/12 sukses. Q-table menyimpan 888 state. Ringkasan setelah evaluasi mencatat 910 karena evaluator bisa menambahkan state di memori lewat lookup tanpa menambah nilai yang dipelajari pada checkpoint.
- Plot rute terbaik menampilkan urutan target lokal. Training.csv menyimpan hasil episode, bukan pose robot per langkah pada percobaan rekor.
- Kandidat diulang dengan heading sama, tetapi jumlah percobaan per rute dan kebijakan saat setiap percobaan dapat berbeda. Rekor menyatakan minimum yang teramati pada run, dan tidak mengukur optimalitas global atau konvergensi.

## Uji Webots terpisah dan pemeriksaan dokumen

Percobaan Webots penuh berakhir oleh batas eksternal 90 detik dan tidak menulis telemetry akhir. Percobaan ulang dengan batas 24 detik simulasi, seed pencarian 6, dan checkpoint baru menulis telemetry valid. Robot menempuh 3,707 m, mengamati 10 sel, dan tidak mencatat tabrakan selama jendela uji. Status timeout pada uji singkat sesuai batas yang ditetapkan. Hasil ini belum memverifikasi misi penuh di Webots.

DOCX dan PDF A4 19 halaman dirender dan diperiksa. Enam artefak run diperiksa byte per byte terhadap sumber. Sampel sepuluh state dibandingkan dengan ekspor lengkap dan nilai Q asli. Rekor pada teks PDF diperiksa terhadap data training. Tidak ada artefak keluaran kosong.

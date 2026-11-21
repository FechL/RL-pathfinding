# Validasi Implementasi Double DQN Micromouse

## Ruang lingkup dan metode

- Acuan arsitektur 5c4cf33c2e045245be1cf49340a5a5e7eb27f5fa. HEAD workspace 8679ee6. Penambahan berada pada modul DQN, adapter bersama dengan perluasan seed opsional, dan routing perintah. Trainer, reward dan aksi tabular, DFS, serta controller Webots tabular tidak diubah oleh pekerjaan ini.
- Python 3.13.13, PyTorch 2.15.0+cpu, NumPy 2.4.4, CPU Intel Core i5-12450H. CUDA tidak tersedia. Training CPU memakai satu thread PyTorch serta mode operasi deterministik.
- Perintah utama `.venv/bin/python tools/train_lidar_dqn.py --output output/drl`. Konfigurasi lengkap, skema dan hash arena ada di `data/run.json`. Patch dan hash artefak ada di `analysis/drl_implementation.patch` dan `data/provenance.json`.

## Hasil pengujian

| Pemeriksaan | Hasil |
|---|---|
| Forward, backward, output 6 dan parameter | Lulus. Jaringan memproduksi tensor 3×6 dari 3 observasi dan memiliki 18.950 parameter. Aktivasi dan gradien finite. |
| Encoder dan normalisasi | Lulus. Dtype float32, 12 fitur, bearing kontinu dan roda terukur. Input NaN ditolak. |
| LiDAR mask dan fallback | Lulus. Greedy dan eksplorasi acak hanya memakai aksi valid. Kondisi semua aksi gerak terblokir memakai Stop. |
| Replay dan sampling | Lulus. Buffer berputar sesuai kapasitas dan mempertahankan mask, truncation, dtype state dan action. |
| Double DQN target | Lulus. Online memilih aksi valid, target mengevaluasi aksi yang sama. Nilai aksi ilegal yang lebih besar tidak ikut dipilih. Terminal hanya menggunakan reward. |
| Optimizer dan target network | Lulus. Bobot online berubah. Target identik saat awal, tersinkron sesuai interval, dan tidak memiliki gradien. |
| Timeout, collision dan recovery | Lulus. Timeout dibedakan dari terminal. Tabrakan memberi −100 dan pemulihan berjalan lima keputusan tanpa reset episode. |
| Checkpoint dan resume | Lulus. Muat bobot, optimizer, replay dan RNG. Update setelah restore cocok secara deterministik pada unit test. Uji CLI terpisah melanjutkan pada episode 2 dan menyelesaikan episode 3. |
| Integrasi kondisi awal dan evaluasi beku | Lulus. Maze asli, inferensi deterministik, serta tidak ada update pada evaluasi. |
| Suite repo | 44 lulus, 2 error dari total 46. Seluruh 9 tes DQN lulus. Dua tes kebijakan lama, `test_included_continuous_actor_completes_mission` dan `test_saved_policy_reaches_goal_and_all_segments_are_safe`, gagal karena signature checkpoint dan arena tidak cocok. Kedua error juga terjadi di snapshot acuan bersih. |
| Smoke training | Awalnya gagal memuat checkpoint dengan weights_only karena reward numpy.float64 masuk metadata. Reward pada log checkpoint dikonversi ke float standar. Smoke kedua menjalankan 100 langkah dan 37 update, memuat model dan menyimpan evaluasi. |
| Run utama | 36 episode dan 52.148 langkah selesai, dengan 51.149 update gradien dan 51 sinkronisasi target. Checkpoint final dan target memiliki bobot finite. |
| Data dan ekspor inferensi | 120 observasi evaluasi diuji terhadap PyTorch dan ekspor NumPy. Keenam nilai Q cocok dalam toleransi, serta aksi yang tercatat sama dengan argmax yang lolos mask. SHA-256 model cocok dengan summary dan metadata ekspor. |
| Rekor uji kecepatan | 21/21 sukses. Rute 2 uji 2 mencatat minimum 98,72 detik dan nol kontak. Rekor direkonsiliasi dari tabel uji aktual. |
| Webots fisika | Seed 6 mencapai goal pada 116,576 detik simulasi, 18,969 m, nol kontak tercatat, dan galat akhir 0,08721 m. Sebanyak 748 keputusan yang direkam mematuhi mask. |

## Batas interpretasi

Hasil kinematik berasal dari satu maze dengan model gerak dan sensor simulasi. Evaluasi Webots hanya satu percobaan eksplorasi, sehingga waktu Webots tidak dipakai untuk rekor rute kinematik. Kontak dihitung menurut kode clearance geometri. Replay uji kecepatan bersifat frozen dan memakai heading awal 0, −0,10, dan +0,10 rad untuk tiap rute. Tidak ada kesimpulan optimum global, konvergensi, atau generalisasi lintas maze.

Log aktual unit test, smoke, training, resume, dan Webots tersedia pada folder analisis.

## Pemeriksaan akhir

Seluruh hash data, model, grafik, dan patch cocok dengan artefak tersimpan. File DOCX lulus pemeriksaan ZIP dan hasil render A4 17 halaman diperiksa. PDF berisi identitas, rekor 98,72 s, dan waktu Webots 116,576 s. Grafik berasal dari log atau telemetry aktual. Tidak ada artefak kosong.

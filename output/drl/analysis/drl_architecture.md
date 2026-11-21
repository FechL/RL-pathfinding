# Arsitektur Double DQN untuk Micromouse

## Audit dan keputusan desain

Branch kerja master berada di 8679ee6. Acuan arsitektur adalah 5c4cf33c2e045245be1cf49340a5a5e7eb27f5fa. Perubahan lokal pada model, catatan, konfigurasi GUI Webots, dan laporan pengguna dipertahankan. Artefak tabular aktual berisi 36 episode dan 12 evaluasi. Anggaran tersebut dipakai sebagai protokol eksperimen. Direktori kode dan output Double DQN terpisah dari trainer serta artefak tabular.

Perbandingan sumber menunjukkan DFS, aksi, dan mask LiDAR sama dengan acuan. Versi kerja `LidarQEnvironment.step` menambah argumen opsional velocity bagi trainer lain. Double DQN memakai jalur enam aksi, dengan argumen velocity bawaan. Python 3.13.13 dan PyTorch 2.15.0+cpu sudah tersedia di .venv. CUDA tidak tersedia. Tidak diperlukan instalasi sistem.

## Aliran data

Maze asli → integrator pose dan LiDAR 180° → DFS menyusun target lokal → encoder float32 → jaringan online memperkirakan enam Q → mask sensor dan kontrol pemulihan menyaring aksi → motor dan reward simulator → replay → pembaruan online dan sinkronisasi target.

`LidarMission` adalah adapter siklus DFS bersama. Adapter DQN mengakses `LidarQEnvironment` melalui metode aksi diskret yang telah tersedia. DFS tetap memeriksa cabang, melakukan scanning, dan backtracking. Aksi dipaksa pada scanning dan pemulihan kontak. Pada scanning, satu aksi putar atau fallback diizinkan. Saat kontak dipulihkan, Stop menjadi token pengendali dan environment menerapkan gerak mundur yang sama dengan baseline. Transisi paksa tetap direkam, dengan penanda fase dan mask singleton, sehingga biaya dan reward pemulihan masuk estimasi masa depan.

## Observasi dan aksi

Sepuluh fitur kontinu: sin dan cos bearing, jarak target dibagi diagonal arena, lima minimum sektor dibagi 4 m, serta kecepatan maju dan putar dihitung dari kecepatan roda dengan kinematika differential drive. Batas normalisasi kecepatan berasal dari batas aktuator, radius roda, dan jarak antarroda. Dua indikator tambahan membedakan scanning dan recovery yang tidak dapat dibedakan secara andal dari sensor saja. Saat target belum tersedia, tiga fitur target bernilai nol. Webots memakai diferensiasi kausal sensor posisi roda.

Aksi 0–5 adalah maju, lengkung kiri, lengkung kanan, putar kiri, putar kanan, dan berhenti. Semua perintah gerak memakai ACTION_VW dari baseline. Mask LiDAR memakai pemeriksaan yang sama dan disimpan bersama next state pada replay. Pembatas WaypointRecovery dan aktuasi paksa diterapkan di atas mask fisik.

## Model, replay, dan eksperimen

`lidar_dqn.py` memuat encoder dan mask, `lidar_dqn_network.py` memuat MLP 128–128 ReLU, `lidar_dqn_replay.py` memuat ring buffer, dan `lidar_dqn_agent.py` menangani Double DQN dan checkpoint. Online network memilih aksi berikut yang valid. Target network mengevaluasi aksi tersebut. Goal dan DFS exhausted menonaktifkan bootstrap. Timeout tetap memungkinkan bootstrap. Reward, reset, dan pemulihan tetap mengikuti environment bersama.

Trainer melakukan eksplorasi DFS dalam semua episode training dan meluruhkan epsilon berdasarkan anggaran langkah nominal. Evaluasi menjalankan protokol heading dan seed acuan tanpa pembaruan bobot. Pengujian waktu menggunakan rute aktual yang berhasil ditemukan saat training, dengan heading awal nol dan model beku. Bukti Webots terpisah dari hasil kinematik. Checkpoint resume dipakai di batas episode dan menyimpan buffer, optimizer, kedua jaringan, counter dan RNG.

## Perintah reproduksi

Gunakan dependensi proyek yang tercantum dalam requirements-training.txt. Instalasi baru dapat memakai `.venv/bin/python -m pip install -r requirements-training.txt` setelah memilih build PyTorch yang sesuai. Eksperimen ini menggunakan lingkungan yang sudah terpasang.

```bash
bash run.sh train --algorithm dqn --output output/drl-new
# Hentikan secara terencana setelah episode 5, dengan konfigurasi tetap 36.
bash run.sh train --algorithm dqn --output output/drl-resume --stop-after-episodes 5
bash run.sh train --algorithm dqn --output output/drl-resume \
  --resume output/drl-resume/models/checkpoint_latest.pt
.venv/bin/python tools/evaluate_lidar_dqn.py --checkpoint output/drl/models/last_model.pt \
  --output output/drl-eval
.venv/bin/python tools/analyze_lidar_dqn.py --output output/drl
.venv/bin/python tools/export_lidar_dqn.py --checkpoint output/drl/models/last_model.pt \
  --output output/drl/models/webots_model.json
bash run.sh sim --algorithm dqn --fast
```

Controller Webots memakai NumPy karena interpreter simulator tidak memiliki PyTorch. Ekspor mempertahankan lapisan, aktivasi, schema, dan SHA-256 checkpoint. Inferensi kinematik memakai file PT langsung. Rangkaian perangkat di Webots membaca GPS, IMU, LiDAR, dan kedua sensor posisi roda. Sampel roda saat ini dikurangi sampel sebelumnya sebelum dinormalisasi. Kontak diperkirakan melalui clearance geometri untuk mempertahankan perilaku pemulihan simulator.

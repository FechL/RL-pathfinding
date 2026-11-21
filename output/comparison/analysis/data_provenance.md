# Provenance data komparatif

Acuan arsitektur adalah commit `5c4cf33`. HEAD yang diamati adalah `8679ee6902564e859ce15fde3511c54169ede416`. Double DQN dan instrumentasi belum menjadi bagian commit HEAD. Status perubahan yang sudah ada tercatat di `worktree_diff_stat.txt` dan `output/drl/analysis/drl_implementation.patch`. Audit menggunakan file aktual di worktree dan metadata run. Analisis ini hanya menambahkan berkas pada `output/comparison`.

| Bukti | Q-Learning | Double DQN |
|---|---|---|
| Run kinematik | `q_learning-5339bee874b3` | `double_dqn-7420a5be748f` |
| Konfigurasi, training, evaluasi, rute, ringkasan | `output/q_learning/recording_run/{run.json,training.csv,evaluations.json,routes.json,summary.json}` | `output/drl/recording_run/data/` dengan nama yang sama, ditambah `evaluation_results.csv` |
| Kebijakan final | `output/q_learning/recording_run/q_table.json` | `output/drl/recording_run/models/last_model.pt` |
| Model pendukung | visits dan enam nilai Q tiap baris pada JSON | `best_model.pt`, `target_network.pt` dan `checkpoint_latest.pt` pada direktori model run rekaman |
| Keputusan dan update | `output/q_learning/telemetry/` | `output/drl/telemetry/` |
| Video pembanding asli | `output/q_learning/videos/` | `output/drl/videos/` |

Telemetry masing-masing memuat `step_telemetry.csv`, `step_telemetry.jsonl`, `episode_summary.csv`, `learning_updates.csv`, `recording_manifest.json` dan `frame_mapping.csv`. `data/source_sha256.json` mencatat sidik SHA-256 untuk sumber aktif. `q_table_export.csv` dan `q_table_sample.csv` yang ada pada direktori Q `data/` adalah pendamping run terdahulu. Semua perhitungan Q-table komparatif membaca JSON run rekaman agar sumber tunggal konsisten. Protokol evaluasi final memakai kebijakan terakhir, termasuk saat ada `best_model.pt`. SHA checkpoint evaluasi Double DQN ada di ringkasan run.

Double DQN dilanjutkan setelah episode lengkap 21. Penggabungan memakai episode lengkap dari kedua sesi. Sebanyak 1.214 baris percobaan parsial tersimpan dalam `output/drl/recording_run/interrupted_capture/telemetry` dan tidak termasuk penyebut hasil. Webots disimpan sebagai eksperimen kinematika/fisika yang terpisah dari dataset komparatif. Rekaman Webots Tabular Q-Learning memakai checkpoint arsip, sehingga tidak diposisikan sebagai pasangan model final yang dievaluasi di sini.

Perangkat yang dilaporkan laporan/provenance DQN adalah 12th Gen Intel Core i5-12450H, CPU. Metadata DQN mencatat Python 3.13.13, PyTorch 2.15.0+cpu dan CUDA tidak aktif. Pada audit, lingkungan proyek memuat NumPy 2.4.4, Gymnasium 1.4.0, OpenCV 4.14.0.94, dan Matplotlib 3.11.2. Q mencatat seed 42 dan konfigurasi simulasi, sementara versi runtime historisnya tidak dicatat di `run.json`. Keduanya direkam di workspace yang sama. Utilisasi CPU, penggunaan RAM proses, dan latensi penuh pipeline sensor historis tidak tersedia. Benchmark tambahan memakai satu CPU logis, satu thread PyTorch, input asli yang disiapkan terlebih dahulu, serta artefak final beku. Lihat `data/inference_benchmark.json` untuk batas ukur dan ulangan.

Reproduksi analisis dari root repository dengan `.venv/bin/python`, jalankan `output/comparison/analysis/build_data.py` lalu `build_graphs.py` dan `build_videos.py` di direktori yang sama. Gunakan FFmpeg dengan libx264 dan OpenCV. Set `MPLCONFIGDIR` ke direktori writable jika diperlukan. Benchmark latency menghasilkan pengukuran baru setiap kali dijalankan. Bukti hasil agregasi dan episode tersedia di `data/comparison_summary.json`, `comparison_metrics.csv` dan `episode_metrics.csv`.

# Validasi hasil penelitian komparatif

| Pemeriksaan | Hasil | Bukti |
|---|---|---|
| Provenance dan integritas | Lulus. 39 hash input sama, termasuk Q-table dan checkpoint final DQN. SHA evaluasi DQN sama dengan checkpoint yang diaudit | `data/source_sha256.json`, `comparison_validation.json` |
| Struktur Q-table | Lulus. 888 baris, enam nilai finite per state dan kunci visits sesuai. Ekspor dan sampel historis cocok dengan nilai serta visits run rekaman | JSON final dan ekspor pada direktori Q-Learning |
| Sumber run dan keterbandingan | Lulus untuk pencocokan sumber, kondisi awal dan unit. Status desain PARTIALLY MATCHED karena fase training, anggaran langkah serta adapter berbeda | `data_provenance.md` dan `comparability_audit.md` |
| Integrasi CSV dan JSONL | Lulus. 137.239 baris cocok per field dalam 117 episode. Yield training Q 42.710 dan DQN 52.148 langkah. Reward, langkah, kontak dan keberhasilan direkonsiliasi per episode | `common_telemetry.csv`, `common_telemetry.jsonl` dan `episode_metrics.csv` |
| Evaluasi | Lulus. Dua belas pasang heading dan seed DFS, epsilon nol, nol update. JSON dan CSV evaluasi DQN cocok. Tiga timeout Tabular tetap tercatat | `matched_evaluation.csv`, `matched_evaluation_summary.md` |
| Rute tercepat | Lulus. Minimum episode sukses cocok dengan catatan rute dan manifest. 113,60 s pada Q dan 98,72 s pada DQN dari fase berbeda | `routes.json` dan video rekor masing-masing |
| Rekaman sumber | Lulus. Pemeriksaan CSV/JSONL, frame, video sumber, mask, aksi, reward dan outcome dijalankan kembali. 53 video diperiksa untuk Q dan 75 untuk DQN | `q_learning_source_validation.log` dan `drl_source_validation.log` |
| Lima MP4 komparatif | Lulus. Dekode seluruh stream dengan FFmpeg, H.264, masing-masing berdurasi positif. 3.073 frame dan 6.146 panel memiliki referensi telemetry yang valid. Pad hanya digunakan setelah terminal | `data/video_manifest.json`, `data/video_frame_mapping.csv` |
| Grafik | Lulus. 24 PNG dibangun dari data per episode, model, telemetry dan benchmark. Sumber dan definisi tertera. Galat TD dan loss ditampilkan pada panel terpisah | `graphs/`, `build_graphs.py` |
| DOCX/PDF | Lulus. Arsip DOCX terbaca, render menghasilkan 17 halaman A4, seluruh halaman dicek visual setelah perbaikan halaman kosong. Tabel, gambar, cuplikan video, caption, identitas dan sumber diperiksa. Hash template asal tetap sama | `report/` dan template Tugas 7 |
| Referensi dan kesimpulan | Judul, tahun, penulis dan penerbit diperiksa melalui penerbit atau naskah primer. Kesimpulan melaporkan hasil deskriptif satu maze dan membedakan biaya skoring, waktu dinding, waktu simulasi, serta batas pencocokan eksperimen | Daftar pustaka dan Bab V–VI |

Outcome dan kontak didasarkan pada event simulator, bukan pada tanda reward. Semua episode gagal tercatat. Webots berada di luar dataset kinematik komparatif. Tidak ada retraining, perubahan model, commit, atau push dalam pekerjaan ini.

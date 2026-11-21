# Evaluasi berpasangan kebijakan final

Status keseluruhan **PARTIALLY MATCHED**. Dua belas pasangan kondisi awal telah tersedia pada run rekaman. Keduanya memakai maze, start, goal, 300 s, heading per seed, enam aksi, reward environment, dan protokol beku yang sama. Update Q dan gradient selama evaluasi sama dengan nol. Detail mask dan kondisi penggantian target adapter berpotensi berbeda, seperti dijelaskan di `comparability_audit.md`.

| Metrik | Tabular Q-Learning | Double DQN |
|---|---:|---:|
| Tercapai | 9/12 | 12/12 |
| Success rate | 75% | 100% |
| Episode dengan collision | 6 | 0 |
| Event collision | 16 | 0 |
| Sukses walau terjadi collision | 3 | 0 |
| Timeout | 3 | 0 |
| Cabang habis | 0 | 0 |
| Durasi mean seluruh episode (s) | 196,19 | 140,21 |
| SD durasi seluruh episode (s) | 73,85 | 52,08 |
| Median durasi seluruh episode (s) | 189,68 | 118,32 |
| Best success (s) | 119,52 | 113,92 |

Pada 9 pasangan dengan kedua robot sukses, perubahan durasi DQN terhadap Tabular rata-rata −16,16 s, median −7,20 s dan SD 58,97 s. DQN lebih cepat pada 7 dari 9 pasangan sukses bersama. Dalam dua pasangan lain, seed 8 dan 9, Tabular lebih cepat. Seed 7, 10 dan 11 hanya berhasil pada DQN. Delapan heading unik diulang dengan seed DFS yang berbeda. Seed 0–11 mengontrol pilihan cabang evaluasi, sedangkan seed training hanya 42. Satu maze dan satu pasang model membatasi keragaman data. Tidak dihitung confidence interval populasi.

`data/matched_evaluation.csv` dan `data/matched_evaluation.json` memuat semua hasil pasangan yang tersedia, termasuk seed, heading, kontak, timing, keberhasilan dan sumber run. Nama file menandai pencocokan kondisi awal. Status protokol lengkap ditulis per baris. Tidak ada evaluasi baru untuk laporan ini. Validasi lebih ketat di masa depan dapat memakai satu loop kontrol bersama, checkpoint beku, sekurang-kurangnya lima skenario luar sampel, dan pencocokan anggaran sebelum training lain.

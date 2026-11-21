# Analisis Tabular Q-Learning dan Double DQN

## Hasil utama

Pada evaluasi berpasangan kondisi awal, Double DQN sukses 12/12 dan Tabular 9/12. Double DQN mencatat nol event tabrakan dan Tabular 16 event. Rerata durasi seluruh episode 140,21 s terhadap 196,19 s. Rerata untuk episode yang sukses sekaligus pada kedua algoritma berbeda −16,16 s (DQN dikurangi Tabular). Hasil mendukung DQN untuk navigasi pada maze dan kebijakan yang diuji.

Pada training, Tabular sukses 29/36 dan DQN 18/36. Tabular beralih ke replay yang terus belajar dan mencapai 18/18 keberhasilan pada fase ini, dengan empat kontak. Selama 18 episode awal keduanya masih eksplorasi. Tabular sukses 11/18 dengan 560 kontak, DQN sukses 6/18 dengan 821 kontak. Epsilon, heading dan alur kontrol berbeda pada awal yang sama sekalipun. Keberhasilan training Tabular mencampurkan dua jenis tugas. Epsilon lebih tinggi pada awal training DQN.

## Reward, stabilitas dan dinamika

Mean reward training −1.309,40 (Tabular) dan −2.977,79 (DQN), dengan median 245,17 dan −4,97. SD reward masing-masing 4.458,86 dan 5.397,76. Kerugian kontak berulang menjelaskan ekor bawah dan selisih besaran mean dan median. Pada Tabular, mean reward eksplorasi −2.917,30 dan replay 298,49, sehingga penyempitan variasi juga berkaitan dengan tugas replay. Mean akhir lima episode mencapai 264,88 dan 253,30. Maksimum DQN lebih tinggi (382,24 vs 365,08), dengan lebih banyak timeout pada training (18 vs 7).

Galat TD absolut rata-rata Tabular turun dari 19,20 ke 0,78 antara episode awal dan akhir. Indikator ini dihitung untuk update yang benar-benar terjadi. Loss minibatch Double DQN mengalami fluktuasi, dengan 0,72 dan 1,52 pada ujung run. Q rata-rata dan maksimum batch tidak mengukur keberhasilan navigasi secara langsung. Perubahan distribusi replay dan target bootstrap turut memengaruhi loss. Kurva tidak menetapkan konvergensi dan metode evaluasi kurang untuk menguji generalisasi.

## Waktu, memori dan keputusan

Waktu dinding episode aktif yang tercatat adalah sekitar 176,32 s dan 302,31 s. DQN menjalani lebih banyak langkah dan memperbarui gradient lewat replay pada CPU. Rentang pengukuran mengecualikan episode parsial yang ditinggalkan, serta bukan benchmark training terisolasi. Ukuran serialisasi Q-table 86.357 byte dan last_model.pt 333.665 byte. Nilai Q mentah tabular berjumlah 21.312 byte, bobot online DQN 75.800 byte. Replay DQN dengan kapasitas 100.000 mengalokasikan 11.600.000 byte array, dengan 52.148 transisi terisi. Overhead Python, PyTorch, dan RAM proses tidak diukur.

Tabular memakai 888 state (11,42% dari 7.776), 37.766 visits, mean 42,53, median 5. Sebanyak 462 memiliki ≤5 visits, termasuk satu state yang hanya tersedia bagi lookup/bootstrap. State baru dapat memakai enam Q awal nol. DQN memetakan 12 fitur ke enam nilai dengan 18.950 parameter, target network dan online network. Ia memungkinkan interpolasi pada observasi baru secara teori. Kinerja di maze lain belum diukur.

## Bukti visual dan interpretasi

Pada seed 0, kedua agen berhasil tanpa kontak. Lintasan Q memuat lebih banyak cabang dan kembali, dengan 244,96 s. DQN menempuh 117,76 s. Pada seed 7, Q berakhir timeout setelah tiga kontak dan menyisakan sekitar 5,39 m jarak Euclidean ke goal. DQN mencapai goal dalam 118,88 s tanpa kontak. Seed 8 memperlihatkan DQN juga dapat lambat pada sebagian konfigurasi (293,28 s). Penelusuran yang dilakukan DFS dipengaruhi lintasan dan hasil sensor, sehingga seed tidak menjamin pilihan rute identik.

Rekor semua percobaan 113,60 s pada Tabular dan 98,72 s pada DQN berasal dari tugas replay dengan kondisi berbeda. Bandingkan evaluasi eksplorasi untuk pilihan deployment pada maze ini. Pilih Tabular untuk kebutuhan penjelasan per state dan biaya skoring rendah. Tidak ada pemenang untuk generalisasi atau stabilitas lintas run karena tidak ada pengulangan training independen.

Sumber: `data/comparison_summary.json`, `data/episode_metrics.csv`, `data/matched_evaluation.csv`, `data/inference_benchmark.json`, `data/video_frame_mapping.csv` dan telemetry sumber yang diidentifikasi dalam `data_provenance.md`.

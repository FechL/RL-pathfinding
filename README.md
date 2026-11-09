# Micromouse — Tugas 8: Tabular Q-Learning

Alur utama Tugas 8 memakai **tabular Q-learning**. Robot mulai dari titik awal,
mencari goal sendiri, dan hanya menyatakan berhasil ketika kamera warna kuning
mendeteksi lantai kuning. Controller Q-learning tidak membaca node `GOAL`,
koordinat target, jarak ke target, atau rute hasil pencarian.

## State, action, dan reward

- **State:** indeks sel dari GPS, arah hadap dari IMU, empat bin sensor IR
  relatif terhadap arah robot (`blocked`, `near`, `clear`), serta flag deteksi
  warna kuning dari kamera. Maksimum teoritisnya 165.888 state; Q-table hanya
  menyimpan state yang pernah dikunjungi.
- **Action:** maju satu sel ke salah satu arah mata angin. Pengendali roda
  memutar robot lalu menggerakkannya ke pusat sel berikutnya.
- **Reward:** `-1` untuk langkah aman, `-10` jika sensor/lingkungan mendeteksi
  tabrakan, dan `+100` saat sensor warna membaca kuning. Tabrakan tidak
  mengakhiri episode; robot tetap punya kesempatan memilih tindakan berikutnya.
- **Akhir episode:** sensor warna kuning mendeteksi goal, atau batas langkah
  tercapai. Tidak ada pemotongan episode khusus karena tabrakan.

Tile kuning dipakai oleh simulator sebagai masukan untuk sensor warna. Selama
training, learner hanya menerima bit `yellow_detected`; posisi tile tidak
menjadi state atau reward jarak. Kamera fisik pada robot diarahkan ke lantai di
bagian depan dan controller hanya mengklasifikasikan piksel kuning.

## Fase training

Training dibagi menjadi dua fase eksplisit. 80% episode pertama adalah fase
**eksplorasi**: epsilon turun bertahap dari 1.0 ke 0.10. Sisa 20% adalah fase
**eksploitasi**: epsilon 0.02, sehingga kebanyakan tindakan memilih Q terbesar
sambil tetap menyisakan sedikit eksplorasi. Semua episode selalu mulai dari
sel awal yang sama. Tidak ada start acak, A*, Dijkstra, policy neural, atau
reward berdasarkan jarak goal.

## Training dan monitor

Siapkan virtual environment training yang dijelaskan pada bagian instalasi,
lalu jalankan:

```bash
bash run.sh train
```

Jendela live menampilkan robot bergerak di maze, state sensor saat ini,
nilai `Q(s,a)` untuk empat aksi, fase, epsilon, reward, langkah, jumlah state
yang sudah dikunjungi, dan grafik reward/steps. Tekan `q` atau `Esc` untuk
menghentikan sesi dan menyimpan hasil yang sudah terkumpul.

```bash
bash run.sh train --episodes 6000 --max-steps 500
bash run.sh train --render-speed 1       # gambar ulang setiap keputusan
bash run.sh train --render-speed 16      # refresh lebih sering
bash run.sh train --render-speed 64      # refresh lebih jarang
bash run.sh train --no-render            # paling cepat; tetap menyimpan CSV
```

`--render-speed N` menggambar monitor setiap N keputusan (default 64); ini mengurangi waktu
untuk menggambar tampilan, bukan mengubah jumlah update Q-learning. `--no-render`
menonaktifkan tampilan dan biasanya menjadi pilihan tercepat untuk training
panjang. `--max-steps` adalah timeout episode.

Satu run membuat folder waktu UTC di `reports/tabular_q_learning/runs/`:

- `training.csv`: fase, epsilon, reward, jumlah langkah, sukses, tabrakan, dan
  jumlah update untuk tiap episode.
- `state_space.csv`: state yang dikunjungi, bin sensor, jumlah kunjungan, dan
  empat nilai Q.
- `summary.json`: parameter dan ringkasan hasil.
- `monitor.png`: tampilan monitor terakhir jika jendela visual dipakai.

Grafik tersimpan dibuat dengan:

```bash
bash run.sh graphs
bash run.sh graphs --report-dir reports/tabular_q_learning/runs/<folder-run>
```

Perintah itu menyimpan grafik reward, langkah, success/collision, epsilon,
frekuensi kunjungan state per sel, dan nilai maksimum Q per sel di folder
`figures/` pada run tersebut.

## Menjalankan policy di Webots

```bash
bash run.sh sim
bash run.sh sim --fast
```

`sim` memuat `models/tabular_q_learning.json`. `--fast` meminta Webots
menjalankan simulasi secepat yang diizinkan komputasi. Controller membuat
keputusan greedy dari Q-table, membaca GPS/IMU, lima sensor IR, dan kamera
warna. Keberhasilan hanya dicatat ketika kamera mendeteksi kuning. Hasil run
fisik tersimpan sebagai `results/latest_run.json`.

Latih ulang setelah perubahan maze/sensor. Training menggunakan simulator
kinematik berbasis geometri maze agar banyak episode bisa dijalankan dengan
cepat; Webots kemudian menjalankan aksi dengan roda diferensial. Karena model
kinematik tidak memodelkan semua slip dan kontak fisika Webots, cek hasil
policy di Webots sebelum dipresentasikan.

## Instalasi

Pasang Webots R2025a, Python/Miniconda, lalu dependency runtime. Jika `.venv`
belum ada:

```bash
~/miniconda3/bin/python -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r requirements.txt
```

Webots memakai `~/miniconda3/bin/python` dari
`controllers/micromouse/runtime.ini`. Training dan grafik memakai `.venv`.

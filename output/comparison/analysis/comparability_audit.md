# Audit keterbandingan

**Keseluruhan training dan evaluasi adapter: PARTIALLY MATCHED.** Evaluasi menggunakan 12 pasang kondisi awal yang identik pada satu maze. Jadwal training dan beberapa detail kontrol adapter berbeda. Kesimpulan berlaku untuk kebijakan dan sistem yang diuji pada konfigurasi ini.

| Parameter | Tabular Q-Learning | Double DQN | Status |
|---|---|---|---|
| Lingkungan gerak | `LidarQEnvironment` kinematik differential drive | `DQNMission → LidarMission → LidarQEnvironment` | MATCHED untuk fisika |
| Maze | 6,4 × 6,4 m, sel 0,4 m, kisi teoritis 16 × 16 | Sama. Hash maze dan susunan dinding identik | MATCHED |
| Posisi start dan goal | (−3,0; −3,0) dan (2,8; 2,8) m | Sama | MATCHED |
| Heading awal training | RNG berurutan, heading nol ketika replay | RNG tersendiri per episode | UNMATCHED |
| LiDAR | Depan 180°, 128 sinar, jarak maksimum 4 m | Sama | MATCHED |
| Sektor | Minimum jarak depan, depan-kiri, kiri, kanan, depan-kanan | Sama, kemudian diskalakan | MATCHED untuk pengukuran |
| DFS dan target lokal | `LidarMazeSearch`, dapat memindai dan kembali ke cabang | Sama. Adapter memiliki kondisi penggantian target dan penghentian yang sedikit berbeda | PARTIALLY MATCHED |
| Aksi dan aktuasi | Enam ID dengan pasangan (v, ω) yang sama | `tabular_step` menerapkan aksi diskret yang sama | MATCHED |
| Masking | `WaypointRecovery.actions` memakai sektor fisik | Mask sektor hasil normalisasi beririsan dengan `WaypointRecovery.actions` | PARTIALLY MATCHED |
| Reward | `LidarQEnvironment.step` | Fungsi yang sama | MATCHED untuk fungsi |
| Timeout | 1.875 langkah × 0,16 s = 300 s | Sama | MATCHED |
| Tabrakan dan mundur | −100 per event, pemulihan fisik 0,8 s | Sama pada mesin gerak. Adapter memiliki kondisi pergantian target berbeda | PARTIALLY MATCHED |
| Budget episode training | 36 | 36 | MATCHED |
| Realisasi langkah training | 42.710 | 52.148 | UNMATCHED |
| Fase training | 18 eksplorasi, 18 replay rute dengan update | 36 eksplorasi, replay dibekukan setelah evaluasi | UNMATCHED |
| Seed training | 42, generator heading berurutan | 42, heading dengan RNG per episode | PARTIALLY MATCHED |
| Evaluasi eksplorasi | Final, ε=0, 12 seed DFS 0–11 dan delapan heading unik | Sama | MATCHED untuk kondisi awal |
| Definisi hasil | Mencapai target akhir, kontak dari event tabrakan environment | Sama | MATCHED |
| Uji rekor rute | Heading 0, kandidat hasil eksplorasi, ε rendah, update berlangsung | Heading 0/±0,10 rad, kandidat dari training dan evaluasi, ε=0 dan model beku | UNMATCHED |
| Beban perangkat ketika training | CPU. Beban dan utilisasi tidak tercatat | CPU. Beban tidak tercatat, run dilanjutkan dari checkpoint | UNKNOWN untuk kondisi beban |

Aksi linear dan yaw (m/s dan rad/s) untuk ID 0–5 berturut-turut (0,23; 0), (0,19; +1,15), (0,19; −1,15), (0; +2,70), (0; −2,70), (0; 0). Keduanya memakai deteksi goal yang sama. Tabrakan tidak harus mengakhiri episode. Setelah 0,8 s, langkah pemulihan fisik berakhir. Panduan akibat stagnasi `WaypointRecovery` merupakan mekanisme berbeda yang dapat membatasi aksi agen.

Dalam `tools/train_lidar_q.py`, penggantian target saat scanning bergantung pada belum recovery dan belum kontak. Waypoint tercapai memasuki kondisi penggantian target yang ditangani sendiri. Dalam `tools/lidar_drl_env.py`, adapter DQN memeriksa terminal, timeout, recovery dan kontak sebelum mengganti target. Q menghitung progres panduan setelah langkah, termasuk saat scanning. DQN meminta mask dari kombinasi observasi terenkode dan panduan pada tahap pemilihan. Pada keadaan DFS kehabisan cabang, adapter DQN dapat mengembalikan terminal tanpa tambahan keputusan gerak. Persamaan utama mode kontrol dan geometri setara, tetapi sekalipun kondisi awal sama, perbedaan adapter dapat berkontribusi pada hasil.

Heading evaluasi `((seed % 8) - 3.5) × π/4` diverifikasi dari pose awal telemetry. Dua belas pasangan merupakan pengulangan pada satu maze dan satu kebijakan final per algoritma, dengan empat heading berulang dan seed DFS berbeda. Fakta ini membatasi inferensi lintas maze dan lintas training seed.

Sumber: `run.json`, `training.csv`, `recording_manifest.json`, `matched_evaluation.csv`, `lidar_q_env.py`, `lidar_qlearning.py`, `maze_search.py`, `train_lidar_q.py`, `train_lidar_dqn.py`, `lidar_dqn_env.py`, `lidar_drl_env.py` dan `evaluate_lidar_dqn.py`.

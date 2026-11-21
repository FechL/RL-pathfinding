# Micromouse dengan eksplorasi LiDAR dan Q-learning

Posisi start dan goal berasal dari world Webots. Robot memakai GPS dan IMU
untuk mengenali posisi serta arah hadapnya. Dari setiap sel yang dicapai,
LiDAR 180° memeriksa bukaan di depan. Jika cabang berada di luar bidang
pandang, robot berputar untuk memindainya. DFS memilih cabang yang belum
dicoba, membuat target lokal, dan berbalik untuk kembali saat cabang buntu.
Layout tembok lengkap hanya dipakai untuk tampilan dan pemeriksaan tabrakan.
Q-learning memilih aksi roda ke target lokal yang dibuat oleh pencarian.

```bash
bash run.sh train --render-speed 16
bash run.sh sim --fast
bash run.sh sim --best-route --fast
```

Jalankan simulasi biasa beberapa kali untuk mencoba urutan cabang berbeda.
Setiap perjalanan yang sampai goal mencatat target yang benar-benar diikuti
(termasuk langkah kembali bila ada) dan waktu simulasi. Mode
`--best-route` memakai urutan target dari perjalanan sukses dengan waktu
terendah untuk world dan Q-table yang sama. Catatan ada di `results/lidar_routes/`.
Rute terpilih hanya dibandingkan dengan perjalanan yang sudah selesai diuji.
Gunakan `MICROMOUSE_SEARCH_SEED=6 bash run.sh sim --fast` untuk mengulang
urutan pemilihan cabang yang sama.

Episode berakhir saat goal tercapai, timeout, atau semua cabang yang terdeteksi
sudah dicoba. Tabrakan memberi penalti `-100`. Robot mencoba mundur sebentar
dengan pemeriksaan jarak aman simulator, lalu melanjutkan episode dan rute
yang sama.

Monitor training menampilkan **EKSPLORASI** saat robot mencari rute dari
LiDAR. Setelah kuota eksplorasi (60% episode secara default) terpenuhi dan
minimal satu rute sampai goal, training otomatis masuk **UJI KECEPATAN**.
Robot bergantian mengulang rute yang ditemukan tanpa putaran kembali akibat
eksplorasi. Semua uji mulai dari arah hadap yang sama. Q-learning masih
memperbarui tabel dan eksplorasi aksi dikurangi. Waktu terbaik dihitung dari
uji kecepatan yang sampai goal dan dicatat di `routes.json` dan `training.csv`
dalam folder run. Jika belum ada rute yang berhasil, fase eksplorasi berlanjut.
Gunakan `--exploration-fraction` untuk mengatur kuota.

Grafik reward dan tombol **BUKA Q-TABLE** tetap tersedia. Statistik episode
tersimpan di `reports/lidar_q_learning/runs/`. Latih ulang Q-table versi lama
untuk lima sektor LiDAR depan (versi 9).
Untuk latihan tanpa jendela, jalankan `bash run.sh train --no-render`.

## DRL dengan SAC dan perbandingan yang setara

Mode `drl` memakai Soft Actor-Critic dengan dua critic dan actor jaringan
saraf. Actor menerima lima jarak LiDAR depan, jarak dan bearing kontinu ke
target, kecepatan roda, serta penanda target aktif. Keluaran mengatur
kecepatan maju dan laju putar kontinu. Pencarian LiDAR, DFS, batas gerak,
reward, dan pemulihan tabrakan memakai aturan yang sama dengan versi tabular.
Mode `sac` dan `naf` lama tetap tersedia sebagai eksperimen tersendiri.

```bash
bash run.sh train --algorithm drl --steps 100000 --render-speed 16
bash run.sh sim --algorithm drl --fast
bash run.sh compare \
  --q-policy /path/ke/run-q/q_table.json \
  --drl-policy /path/ke/run-drl/actor.json \
  --routes /path/ke/run-q/routes.json
```

Training DRL menyimpan `actor.json`, checkpoint SAC, data episode dan loss,
grafik, serta `summary.json` di `reports/lidar_drl/runs/`. Fase
uji kecepatan dimulai setelah kuota langkah eksplorasi (default 60%)
terpenuhi dan ada rute sukses. Tambahkan `--no-render` untuk training tanpa
jendela. Pengendali Webots memuat actor JSON tanpa dependensi PyTorch.

Untuk membandingkan waktu training, jalankan kedua mode dengan `--no-render`,
`--steps` dan `--seed` yang sama. `training_seconds` mengukur waktu proses
training di luar evaluasi dan ekspor. `first_success_step` dan
`first_success_training_s` mengukur kapan goal pertama tercapai. Kolom
`duration_s` di CSV mengukur waktu perjalanan dalam simulasi.

SAC secara default melakukan satu pembaruan jaringan setiap dua langkah
simulasi. Gunakan `--update-every 1` untuk konfigurasi lama atau
`--update-every 4` untuk mengurangi biaya komputasi lebih jauh. Catat pilihan ini
saat membandingkan kualitas model karena jumlah pembaruan ikut berubah.

Perintah `compare` menguji kebijakan yang dibekukan dalam simulator kinematik.
Keduanya mengikuti urutan target yang sama dari `routes.json` dengan tiga
arah hadap awal dan timeout yang sama. File `trials.csv`, `summary.json`,
dan `comparison.png` masuk ke `results/lidar_comparison/`. Median waktu
tempuh hanya dihitung dari percobaan yang mencapai goal. Hasil ini mengukur
kontrol gerak pada rute tetap. Pencarian rute dan fisika Webots perlu
dievaluasi terpisah.

## Double DQN dengan LiDAR

`dqn` memakai observasi kontinu, enam aksi diskret, dan penyaringan LiDAR
baik saat memilih aksi maupun menghitung target Q. DFS tetap menghasilkan
target lokal. Training, evaluasi greedy, dan uji waktu dengan bobot beku
menghasilkan artefak di `output/drl/`. Konfigurasi dapat diubah melalui
`configs/lidar_dqn.json` atau file JSON yang diberikan dengan `--config`.

```bash
bash run.sh train --algorithm dqn --output output/drl-new
.venv/bin/python tools/evaluate_lidar_dqn.py --checkpoint output/drl-new/models/last_model.pt --output output/drl-eval
.venv/bin/python tools/analyze_lidar_dqn.py --output output/drl-new
.venv/bin/python tools/export_lidar_dqn.py --checkpoint output/drl-new/models/last_model.pt --output output/drl-new/models/webots_model.json
bash run.sh sim --algorithm dqn --fast --policy output/drl-new/models/webots_model.json
```

Gunakan `--stop-after-episodes` untuk berhenti pada batas episode, lalu
`--resume <run>/models/checkpoint_latest.pt` dengan konfigurasi run yang sama.
Untuk protokol, metrik, dan detail checkpoint, lihat
`output/drl/analysis/drl_architecture.md`.

## Rekaman training dan telemetry

Tambahkan `--record DIREKTORI_BARU` pada trainer `q_learning` atau `dqn`
untuk menangkap satu baris per langkah. Encoding berjalan terpisah dari
training. Codec H.264 dipilih bila FFmpeg menyediakan libx264. Perekaman
tidak perlu jendela. Gunakan `--no-render` pada trainer tabular untuk
menonaktifkan preview.

```bash
bash run.sh train --algorithm q_learning --episodes 36 --exploration-fraction 0.5 --evaluation-runs 12 --no-render --record output/q_capture_baru
bash run.sh train --algorithm dqn --output output/dqn_model_baru --record output/dqn_capture_baru
.venv/bin/python tools/render_micromouse_recording.py --output output/q_capture_baru
.venv/bin/python tools/validate_micromouse_recording.py --output output/q_capture_baru --training-episodes 36 --evaluation-episodes 12
```

Video mencakup setiap episode dengan sampling frame yang tercatat pada
manifest. `frame_mapping.csv` menghubungkan frame, episode, langkah, waktu,
dan baris asli pada telemetry. Setiap langkah tetap tersimpan di CSV dan JSONL.
Reward dan pose setelah aksi dipisahkan dari observasi yang dipakai untuk
memilih aksi. Baris loss hanya ada bila terjadi update gradien. Nilai TD
individual DQN pada log update merujuk sampel pertama batch, dan agregat batch
tersedia pada kolom tersendiri. Log dibuang dari buffer Python setiap 256
langkah dan pada batas episode.

Rekaman Webots dapat dirender terpisah dari file telemetry GPS, IMU, dan LiDAR
dengan `tools/render_micromouse_webots.py`. Nilai yang tidak berasal dari
inferensi fisika, seperti reward training, ditampilkan sebagai tidak tersedia.

## Studi komparatif Tabular Q-Learning dan Double DQN

Laporan final tersedia dalam
[PDF](output/comparison/report/Laporan_Perbandingan_Q_Learning_vs_Double_DQN_Rafliansyah.pdf)
dan [DOCX](output/comparison/report/Laporan_Perbandingan_Q_Learning_vs_Double_DQN_Rafliansyah.docx).
Folder [output/comparison](output/comparison/) berisi lima video, grafik,
data evaluasi, dan audit keterbandingan. Run sumber tersedia di
[`output/q_learning`](output/q_learning/) dan [`output/drl`](output/drl/).

Telemetry besar dipublikasikan dalam format gzip lossless. Untuk menyiapkan
ulang file CSV, JSON, dan JSONL yang dibutuhkan skrip analisis, jalankan
perintah berikut dari root repository.

```bash
find output/comparison output/q_learning output/drl -name '*.gz' -exec gzip -dk -- {} +
```

Langkah analisis ulang dan sumber data dijelaskan di
[`data_provenance.md`](output/comparison/analysis/data_provenance.md).

## Optional legacy continuous SAC

Older continuous SAC modes remain available by selecting `hidden_goal`, `sac`
or `guided_rl`. Their actor policies are separate from the tabular LiDAR
Q-table.

```bash
bash run.sh train --algorithm sac --warm-start
bash run.sh train --algorithm guided_rl
bash run.sh sim --algorithm sac
```

## Arsip eksperimen controller sebelumnya

Bagian setelah ini mendokumentasikan eksperimen NAF/SAC dan baseline lama.
Untuk controller yang sekarang, gunakan alur LiDAR tabular Q-learning di
bagian atas README; itulah default untuk `bash run.sh train` dan
`bash run.sh sim`.

## Tugas 8 — Continuous Deep Q-Learning

Select **Normalized Advantage Functions (NAF)** for the Tugas 8 controller.
NAF uses the same 26 real-valued observations and two real-valued forward
velocity and yaw-rate actions as SAC. The legacy Q-table still represents
the discrete baseline. See [Gu et al. (2016)](https://proceedings.mlr.press/v48/gu16.html).

```bash
bash run.sh sim --algorithm naf
bash run.sh train --algorithm naf --render-speed 1
bash run.sh train --algorithm naf --steps 30000 --eval-every 2500 --seed 42 --no-render
bash run.sh graphs --report-dir reports/naf
```

The NAF learner represents `Q(s,a) = V(s) − ½ ‖L(s)ᵀ(a − μ(s))‖²`.
Its greedy action is `μ(s)` in the continuous action box, and its Bellman
target bootstraps with the target network's `V(s′)`. A goal or collision
ends the bootstrap. A time limit leaves the bootstrap active. Gaussian
exploration and replay supply continuous actions for the updates.

The recorded run initializes `μ` from the exported SAC controller and uses a
small teacher anchor during training. The value and precision parameters start
randomly. Use `--no-warm-start` to omit teacher initialization and anchoring.
The published comparison starts from the warm-start checkpoint before any
NAF updates. The best mean in the fixed kinematic evaluation fell from
52.651 s to 52.587 s, a difference of one control interval. Three seeded
evaluations at each of the 13 checkpoints reached the goal.

The training window shows the actual 2D kinematic rollout and reward. Each
run saves PNG plots, a training monitor and evaluation checkpoints under
`reports/naf/runs/`. With `--activate`, the best trained checkpoint that
reaches the goal in every evaluation replaces `models/naf_actor.json`.
Without that flag, use an absolute `MICROMOUSE_POLICY` to test a specific
export without changing the deployed policy.

```bash
xvfb-run -a ~/miniconda3/bin/python tools/validate_simulation.py \
  --algorithm naf --runs 3 --keep-telemetry --report-dir reports/tugas8/naf_webots
```

Recorded training evidence is in `reports/naf/`. Webots physics results
and the comparison of the same turn across controllers are in `reports/tugas8/`.
In the original report, NAF completed all three Webots resets in 53.008 s,
matching the warm-start policy's time. The matched-speed tabular controller took 98.000 s
and followed a different route in part of the maze. On the shared corner,
NAF took 1.317 s with 6.67 cm clearance and the tabular controller took
2.444 s with 7.89 cm clearance. NAF kept moving forward throughout the
samples classified as turning between the gates.

The smoother controller uses the same NAF weights and completed all three
Webots resets in 52.352 s, with 6.15 cm minimum clearance. In the same corner,
RMS yaw jerk from the IMU fell from 272.3 to 162.0 rad/s³. The updated
telemetry and before/after plot are in `reports/tugas8/smooth_naf_webots/`.

Regenerate the corner and route comparisons from their inputs:

```bash
.venv/bin/python tools/evaluate_tugas8.py webots \
  reports/tugas8/{naf_webots,tabular_webots,sac_webots}/telemetry.json
.venv/bin/python tools/evaluate_tugas8.py routes models/naf_actor.json
```

In the paired kinematic route test, the same trained NAF policy completed the
mission in a mean of 52.203 s with the alternate guide and 52.587 s with the
default guide. The comparison changes one graph edge in the Dijkstra guide
while retaining the same wall geometry and reset seeds. The guide selects the
corridor and NAF controls the continuous motion within it.

## What the agent learns

Both continuous policies make a fresh decision every 16 ms in Webots.
Training still requests actions every 64 ms and applies four 16 ms motor substeps.
The two real-valued outputs
request forward velocity (0–0.38 m/s) and yaw rate (−4–4 rad/s). Both can be
nonzero simultaneously. Shared motor conversion enforces a 17 rad/s wheel
speed limit and 45 rad/s² wheel acceleration limit. An 80 ms exponential
wheel response softens abrupt changes in the policy's requests. There
is no compulsory “stop, rotate 90°, move one cell” action sequence.

The 26 continuous observations contain a local navigation bearing, wheel
velocities, 16 geometric range rays, XY position, sine/cosine of heading, and
the goal vector in robot coordinates. The complete map is known. Range rays
are computed from that map and the current GPS/IMU pose; the four physical
IR sensors are recorded separately, not substituted for those rays.

A Dijkstra distance field over known maze connectivity supplies local guidance
and a dense progress reward. This guide biases exploration toward short
routes; it does not prescribe wheel commands or a timed trajectory. **This
is map-assisted continuous control, not autonomous mapping or proof of the
globally fastest route.** SAC can adjust its trajectory, speed and turns,
but establishing the fastest alternative corridor requires explicit route
comparisons and more training. Changing the map requires retraining.

The training reward is `12 × geodesic progress − elapsed seconds − wheel slew
penalty`, with +30 for reaching within 5 cm of the goal centre and −15 for
collision. Faster successful traversal is favoured; collisions terminate the
episode. A time limit is truncation, so SAC retains its value bootstrap.
Random continuous starting positions and headings provide a curriculum.

## Run

Install Webots R2025a, Miniconda and the lightweight runtime requirements:

```bash
~/miniconda3/bin/python -m pip install -r requirements.txt
bash run.sh
```

Webots uses `~/miniconda3/bin/python` via
`controllers/micromouse/runtime.ini`. Running the older exported actor needs only
NumPy and OpenCV, not PyTorch. Its model was `models/continuous_actor.json`;
map, controller configuration and observation version are checked at load.
No fallback controller is silently selected if the model is incompatible.

Set `MICROMOUSE_NO_GUI=1` to hide the OpenCV map. Webots still needs a display.
The map shows the actual driven trajectory; no full route is precomputed
from the continuous actor. `results/latest_run.json` records pose, actions,
measured and commanded wheel speeds, IR readings, elapsed time, trajectory,
clearance, and motion metrics. A stalled or unsafe episode stops and reports
failure.

Explicit older baselines:

```bash
MICROMOUSE_ALGORITHM=q_learning bash run.sh
MICROMOUSE_ALGORITHM=astar bash run.sh
```

## Train or continue training

Use an isolated environment for the training dependencies. For a CPU machine:

```bash
~/miniconda3/bin/python -m venv --system-site-packages .venv
.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r requirements-training.txt
bash run.sh train --steps 300000 --seed 42
```

Training uses Stable Baselines3 SAC with two critics and a squashed Gaussian
actor (64/64 hidden units). It learns in a continuous differential-drive
kinematic surrogate, with exact arc integration and swept collision checks
at 16 ms substeps. It does **not** train in Webots rigid-body physics. Real
motor slip and contact dynamics must be checked separately in Webots.

`--warm-start` explicitly uses 30,000 continuous feedback driving examples
to teach basic steering, then warms both critics for 4,000 updates. SAC
continues with its Q/entropy objective plus a behaviour-cloning regularizer
(weight 10) to retain safe basic driving while adjusting speed and turns.
The feedback reference is used only during training. The deployed actor is
a neural network, with no reference-action mixing or safety action mask.
Use `bash run.sh train --no-warm-start` to train pure SAC from scratch; it can take substantially more
experience. The included run is not evidence of learning without examples.

Each training invocation creates its own timestamped folder under
`reports/continuous/runs/`. The terminal prints its exact path. Use
`--run-dir reports/continuous/runs/my_run` to name a new run explicitly;
existing training logs cannot be overwritten. Runs and their full models
remain local and are excluded from Git.

Checkpoints are saved every 10,000 steps in that run's `models/` folder.
Every 20,000 new steps, three mission-start evaluations rank fully successful
candidates by mean time. The selected portable actor is saved as
`models/continuous_actor.json` inside the run, alongside the full SAC
checkpoints and final replay buffer. `--activate` explicitly copies a fully
successful best actor to the project's default Webots model. Without that
flag, training preserves the existing Webots actor. To try a new run without
changing the default, set `MICROMOUSE_POLICY` to its actor's absolute path
when running `bash run.sh`.

Continue the previously included full checkpoint with:

```bash
bash run.sh train --steps 200000 --resume models/sac_continuous.zip
```

For later runs, use the full checkpoint path inside that run's `models/`
folder. Training episode data and seeded evaluation trajectories are saved
inside each run. Historical evidence for the included actor remains under
`reports/continuous/`. Evaluation uses the actual mission start, deterministic
actor output, and separate seeds with start-position perturbations. A saved
checkpoint is evidence of training, not automatically evidence of success.

## See the training graphs

Regenerate the included training figures without retraining:

```bash
bash run.sh graphs
```

The newest hidden-goal run is selected by default. Its `figures/` folder has
`training_progress.png`, `episode_steps.png`, `episode_outcomes.png`,
`q_learning.png`, and `state_space.png`. These show kinematic training and
evaluation; validate policies in Webots physics separately. Historical runs
under `reports/continuous/` used known-goal guidance and have different logs.

All training runs from the same shell entry point: `bash run.sh train`.
It opens a local 2D maze window showing the robot's training movements, trail,
episode, reward, continuous state and sensor values, and collision/yellow status.
The phase label changes from exploration to exploitation. The default preview
speed is 8x simulation time for SAC and real time for `bash run.sh train --algorithm naf`.
Use `bash run.sh train --render-speed 1` for real-time SAC preview,
or `bash run.sh train --no-render` for fast training without a window.
Press Q/Esc or close the window to stop; during SAC, partial training is
saved as a run candidate and evaluated.
This is a kinematic training preview, not Webots 3D physics training.

Training prints progress to the terminal and saves local PNG graphs; no web
service or browser is needed. `bash run.sh` runs the trained robot in Webots,
and `bash run.sh graphs` regenerates the saved training figures.

PNGs refresh after each evaluation and at the end. To refresh them every five
seconds for a specific run:

```bash
bash run.sh graphs --report-dir reports/continuous/runs/my_run --watch
```

The faint lines show individual episodes; thick lines show a rolling mean over
up to 20 episodes. Hidden-goal training always starts at the mission start; a
short episode may be a collision. Check success rate and state-space plots as
well as reward and episode length. The recorded results later in this README
describe earlier known-goal SAC/NAF runs, not the new hidden-goal task.

## Verify and compare

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python tools/train_continuous.py --evaluate-only
~/miniconda3/bin/python tools/validate_simulation.py --runs 3 --algorithm sac
~/miniconda3/bin/python tools/validate_simulation.py --runs 1 --algorithm q_learning --report-dir reports/continuous_baseline_webots
```

For a baseline with the same 0.38 m/s forward and 4 rad/s yaw speed budgets:

```bash
MICROMOUSE_BASELINE_SPEED=0.38 MICROMOUSE_BASELINE_TURN_SPEED=7.7037037037 xvfb-run -a ~/miniconda3/bin/python tools/validate_simulation.py --runs 1 --algorithm q_learning --report-dir reports/continuous_matched_baseline_webots
```

On a headless Linux machine, prefix the Webots validation command with
`xvfb-run -a`. Physics validation runs isolated project copies, checks goal
error, physical wall clearance and every recorded trajectory segment, and
saves logs/screenshots under `reports/sac_webots/`.

Compare successful runs using `simulation_duration_s`, `travelled_length_m`,
`goal_error_m`, `min_physical_clearance_m`, `moving_turn_fraction`, stopped
fraction and peak wheel acceleration. Turning fraction counts yaw rates
above 0.4 rad/s and forward speed above 0.05 m/s; it does not by itself prove
optimal cornering. Three deterministic resets show repeatability, not
statistical generalization to other mazes or hardware.

## Recorded continuous-control results

The included actor was selected after reward-driven SAC fine-tuning, using
30,000 continuous driving examples, critic warmup and a BC regularizer. The
selected checkpoint has 140,000 accumulated environment steps; training
continued to 220,000 and retained the faster fully successful checkpoint.
Its seeded kinematic mean time is 52.651 s, compared with 53.184 s directly
after the demonstration warm start. See `reports/continuous/checkpoints.json`.

Three deterministic Webots runs with the exact exported actor all succeed
in **53.008 s**, with at least **0.061 m physical clearance** and goal error
at most **0.0483 m**. The previous default takes **137.888 s**. With matched
forward/yaw speed budgets the old controller takes **98.000 s**, making this
continuous configuration about **45.9% faster** on this fixed maze. The old
baseline has no acceleration governor and uses a tighter final waypoint
tolerance, so this is a controller-configuration benchmark, not a controlled
claim that one RL algorithm is universally superior.

Commanded wheel acceleration is bounded at 45 rad/s². During samples with
yaw rates above 0.4 rad/s, forward commands exceed 0.05 m/s in 99.7% of samples;
the matched legacy baseline has zero moving-turn samples. Motion summaries
use commanded wheel speeds consistently across controllers; full runtime
telemetry also records measured encoder speeds. Results and the actor's
SHA-256 fingerprint are recorded in `reports/continuous/comparison.json`.

All 26 unit/integration tests pass, including arc integration, swept
collision checks, Gymnasium semantics, exported actor equivalence and a
complete mission with the included actor. The existing Word report is
historical and has not been rewritten to represent these SAC results.

## Maze and legacy experiments

`worlds/maze_layout.json` controls geometry and seed. Regenerate and retrain:

```bash
~/miniconda3/bin/python tools/build_world.py
.venv/bin/python tools/train_continuous.py --steps 300000 --warm-start
```

The old option-based learner is retained in `q_learning.py`, with its trainer
at `tools/train_q_learning.py`, model at `models/q_learning.json`, and reports
in `reports/q_learning/` and `reports/q_learning_webots/`. It uses 256 discrete
cell states and four cardinal actions; these are not the new SAC controller.
Earlier reports and the Word document describe those historical experiments,
not the new controller's performance.

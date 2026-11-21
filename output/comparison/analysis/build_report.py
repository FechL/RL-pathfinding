"""Assemble the comparison report using the retained Tugas 7 DOCX template."""
import csv
import json
from pathlib import Path
import sys

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

out=Path('output/comparison')
template,still=map(Path,sys.argv[1:3])
doc=Document(template)
styles={s.name:s for s in doc.styles}
for e in list(doc._element.body)[20:]:
    if e.tag!=qn('w:sectPr'):
        doc._element.body.remove(e)
doc.paragraphs[0].runs[0].text='LAPORAN PENELITIAN KOMPARATIF'
title=doc.paragraphs[1]
title.clear()
run=title.add_run('ANALISIS PERBANDINGAN TABULAR Q-LEARNING DAN DOUBLE DEEP Q-NETWORK PADA NAVIGASI ROBOT MICROMOUSE MENGGUNAKAN LiDAR DALAM CONTINUOUS ENVIRONMENT')
run.font.name='Times New Roman'
run.font.size=Pt(17)
run.font.bold=True
run.font.color.rgb=RGBColor(0,0,0)
doc.paragraphs[2].runs[0].text='Evaluasi kinematik pada maze proyek Cognitive Robotics'
# The comparative title occupies more lines than the original title.
for p0 in list(doc.paragraphs)[3:6]:
    p0._element.getparent().remove(p0._element)
doc.core_properties.title='Analisis Perbandingan Tabular Q-Learning dan Double Deep Q-Network pada Navigasi Robot Micromouse'
doc.core_properties.author='Rafliansyah Rizqi Pratama Bukhori'
m=json.loads((out/'data/comparison_summary.json').read_text())
videos=json.loads((out/'data/video_manifest.json').read_text())
q,d=(m['metrics'][k] for k in ('q_learning','drl'))
mq,md=(m['memory'][k] for k in ('q_learning','drl'))
pairs=list(csv.DictReader((out/'data/matched_evaluation.csv').open()))
t_no=g_no=0
def new(h):
    doc.add_page_break()
    doc.add_paragraph(h,styles['Heading 1'])
def h(s): doc.add_paragraph(s,styles['Heading 2'])
def p(s): doc.add_paragraph(s)
def caption(t):
    a=doc.add_paragraph(t,style=styles['Caption'])
    a.alignment=WD_ALIGN_PARAGRAPH.CENTER
    return a
def n(x,k=2):
    return f'{x:,.{k}f}'.translate(str.maketrans(',.','.,'))
def table(title,heads,items,src):
    global t_no
    t_no+=1
    caption(f'Tabel {t_no}. {title}').paragraph_format.keep_with_next=True
    t=doc.add_table(rows=0,cols=len(heads))
    t.style=styles['Table Grid']
    for idx,values in enumerate([heads,*items]):
        row=t.add_row()
        row._tr.get_or_add_trPr().append(OxmlElement('w:cantSplit'))
        if idx==0: row._tr.get_or_add_trPr().append(OxmlElement('w:tblHeader'))
        for c,value in zip(row.cells,values):
            if idx==0:
                shade=OxmlElement('w:shd'); shade.set(qn('w:fill'),'E9E9E9')
                c._tc.get_or_add_tcPr().append(shade)
            text=c.paragraphs[0]
            text.paragraph_format.space_after=Pt(2)
            text.paragraph_format.space_before=Pt(2)
            text.paragraph_format.line_spacing=1
            r=text.add_run(str(value)); r.bold=(idx==0); r.font.size=Pt(9.0)
    a=doc.add_paragraph('Sumber. '+src)
    a.runs[0].font.size=Pt(9)
def graph(file, label, width=15):
    global g_no
    g_no+=1
    path=out/'graphs'/f'{file}.png' if isinstance(file,str) else file
    a=doc.add_paragraph()
    a.alignment=WD_ALIGN_PARAGRAPH.CENTER
    a.paragraph_format.keep_with_next=True
    a.paragraph_format.space_after=Pt(1)
    a.add_run().add_picture(str(path),width=Cm(width))
    caption(f'Gambar {g_no}. {label}')

doc.add_paragraph('Ringkasan',styles['Heading 1'])
p('Penelitian ini membandingkan Tabular Q-Learning dan Double DQN untuk mengendalikan robot Micromouse pada maze kinematik yang sama. Sistem memakai LiDAR depan dan Depth-First Search (DFS) untuk menemukan cabang dan menetapkan target lokal. Kedua agen memilih satu dari enam aksi roda diskret. Pose, laju roda, dan lintasan robot berkembang secara kontinu.')
p('Analisis memakai masing-masing 36 episode training dan 12 pasangan evaluasi beku. Kondisi awal evaluasi dipasangkan berdasarkan seed DFS dan heading. Jadwal training serta detail pengelolaan target dan mask di adapter sebagian berbeda. Pada evaluasi, Double DQN mencapai goal pada 12 dari 12 episode tanpa kontak. Tabular mencapai goal pada 9 dari 12 episode dengan 16 event kontak. Durasi rata-rata seluruh episode masing-masing 140,21 s dan 196,19 s. Pada sembilan pasangan yang sama-sama berhasil, durasi Double DQN rata-rata 16,16 s lebih singkat. Selama training, Tabular mencapai 29 keberhasilan dan Double DQN 18. Training Tabular mencakup 18 episode replay berhasil.')
p('Tabular memiliki artefak penyimpanan lebih kecil dan latensi skoring enam aksi lebih rendah pada benchmark CPU dengan fitur siap pakai. Double DQN menunjukkan navigasi yang lebih berhasil dan lebih sedikit tabrakan pada evaluasi ini. Data meliputi satu maze dan satu seed training per algoritma. Kesimpulan tentang generalisasi atau keunggulan keseluruhan masih memerlukan pengujian tambahan.')
h('Kata kunci')
p('Micromouse, LiDAR, DFS, continuous environment, Tabular Q-Learning, Double DQN')
h('Sumber primer eksperimen')
p('Run q_learning-5339bee874b3 dan double_dqn-7420a5be748f, masing-masing ditautkan oleh run ID, episode, langkah dan nomor baris telemetry. Acuan arsitektur repository 5c4cf33. File hasil menggunakan worktree pada HEAD 8679ee6 dengan implementasi Double DQN dan perekaman di atasnya.')

new('BAB I Pendahuluan')
h('1.1 Latar belakang')
p('Pada maze, robot perlu menghindari dinding sekaligus menemukan jalur menuju goal. Pengetahuan posisi awal dan tujuan memberi acuan navigasi. Hubungan antarsel tetap dibangun saat robot bergerak dan membaca LiDAR. Pemisahan pencarian rute dan kontrol lokal membuat kebijakan penggerak dapat dibandingkan dengan mempertahankan fisika, sensor, serta pemetaan aksi yang sama.')
h('1.2 Rumusan masalah')
p('Bagaimana kedua metode membentuk representasi pengalaman dan memperbarui nilai aksi? Bagaimana kemampuan mencapai goal, menghindari tabrakan, dan memulihkan diri terlihat pada trajectory, durasi, langkah dan reward? Apa biaya komputasi dan memori yang teramati? Sejauh mana konfigurasi dan variasi data mendukung kesimpulan mengenai kestabilan dan pemilihan algoritma?')
h('1.3 Tujuan dan manfaat')
p('Tujuan penelitian adalah mengaudit data, membandingkan kedua sistem pada kondisi awal evaluasi yang sama, serta menilai hasil dari reward, metrik navigasi, penggunaan representasi, benchmark CPU dan bukti visual. Hasilnya membantu memilih kebijakan lokal sesuai prioritas keselamatan, kebutuhan penelusuran keputusan dan anggaran komputasi.')
h('1.4 Ruang lingkup')
p('Penelitian menggunakan artefak kinematik yang sudah selesai, termasuk percobaan sukses dan gagal. Rekaman Webots memiliki sumber dan checkpoint yang berbeda dari sebagian run kinematik, sehingga hasil fisik disimpan terpisah. Analisis menggunakan kedua kebijakan final yang dibekukan. Pembanding statistik bersifat deskriptif pada satu maze.')

new('BAB II Landasan Teori')
h('2.1 Reinforcement learning dalam continuous environment')
p('Dalam reinforcement learning, agen berinteraksi dengan lingkungan, menerima reward, lalu memilih tindakan untuk meningkatkan estimasi akumulasi reward [1]. Pada Micromouse, posisi, heading, laju dan respons roda berubah kontinu. Enam perintah gerak mendiskretisasi pilihan kendali. Grid maze memudahkan penalaran atas cabang dan target DFS.')
h('2.2 Tabular Q-Learning')
p('Tabular Q-Learning menyimpan estimasi imbal hasil per pasangan state dan aksi [2]. Dengan laju pembelajaran α dan diskonto γ, estimasi bergerak menuju target TD sebanyak α kali selisih antara target dan estimasi saat ini. Target memakai reward aktual dan estimasi terbaik dari aksi yang diizinkan pada state berikutnya. Transisi terminal memakai reward aktual saja.')
h('2.3 Deep Q-Network dan Double DQN')
p('DQN mengganti lookup tabel dengan jaringan saraf, sehingga parameter yang sama dipakai di banyak observasi [3]. Experience replay menyediakan batch pengalaman untuk optimisasi. Double DQN memilih aksi berikutnya melalui jaringan online dan mengevaluasi aksi yang dipilih dengan jaringan target [4]. Pemisahan ini ditujukan untuk mengurangi bias estimasi maksimum. Sampel timeout dibedakan dari terminal. Loss Smooth L1 menggabungkan wilayah kuadrat di dekat galat nol dengan ekor linear [6].')
h('2.4 LiDAR, DFS, dan differential drive')
p('LiDAR mengukur jarak sepanjang sinar dan dapat dipetakan menjadi sektor untuk mendeteksi ruang bebas [5]. DFS mencoba cabang satu per satu dan melakukan backtracking saat tidak ada cabang baru pada jalur aktif. Pada differential drive, selisih laju roda membentuk yaw, sedangkan rata-ratanya membentuk laju linear [5]. Perintah nonnol pada keduanya menghasilkan lengkung.')
h('2.5 Epsilon-greedy, reward, dan metrik')
p('Epsilon mengatur peluang pengambilan aksi acak dari aksi yang diizinkan. Eksploitasi memilih aksi dengan estimasi Q terbesar yang lolos mask. Reward menggabungkan pendekatan pada target, keselamatan dan efisiensi. Keberhasilan diukur dari deteksi goal. Kontak, timeout, waktu simulasi, langkah, panjang lintasan dan wall-clock diperiksa terpisah.')

new('BAB III Metodologi')
h('3.1 Arsitektur, maze, dan sensor')
p('Urutan kontrol adalah pose dan LiDAR → DFS dan target lokal → representasi observasi → filter aksi → kebijakan → differential drive → reward dan event. Lingkungan kinematik menggunakan batas −3,2 sampai 3,2 m pada kedua sumbu. Sel 0,4 m membentuk kisi teoritis 16 × 16. Start berada di (−3,0; −3,0) m dan goal di (2,8; 2,8) m. Goal dikenali ketika robot mendekati target akhir dalam 0,09 m pada langkah valid. LiDAR depan 180° berisi 128 sinar dan memiliki jangkauan 4 m. Minimum lima sektor disusun depan, depan-kiri, kiri, kanan, depan-kanan.')
table('Pasangan target linear dan laju yaw yang dipakai bersama',['ID','Perintah','Linear (m/s)','Yaw (rad/s)'],
      [(0,'Maju','0,23','0'),(1,'Lengkung kiri','0,19','+1,15'),(2,'Lengkung kanan','0,19','−1,15'),
       (3,'Putar kiri','0','+2,70'),(4,'Putar kanan','0','−2,70'),(5,'Berhenti','0','0')],
      'lidar_qlearning.py, q_table.json dan skema checkpoint Double DQN.')
h('3.2 Reward dan tabrakan')
table('Komponen reward bersama',['Komponen','Nilai atau pengali'],
      [('Progres ke target lokal','4 × pengurangan jarak (m)'),
       ('Heading','0,20 × pengurangan galat sudut (rad)'),
       ('Waktu dan jarak','−0,025 per langkah dan −0,50 × jarak (m)'),
       ('Dinding','−0,45 × clamp((0,24 − clearance)/0,24, 0, 1), pengali 1,5 saat maju'),
       ('Waypoint, goal dan kontak','+4, +120, dan −100 pada event yang sesuai'),
       ('Stagnasi','−0,10 jika kemajuan jarak <0,002 m dan perbaikan heading <0,05 rad')],
      'LidarQEnvironment.step. Progres, heading dan stagnasi dinonaktifkan pada pemindaian dan mundur.')
p('Keputusan berjalan setiap 0,16 s. Timeout terjadi pada 1.875 langkah (300 s). Kontak memicu mundur berkecepatan −0,09 m/s selama 0,8 s dan episode dapat berlanjut. DFS dapat mengakhiri episode bila cabang habis. State dan target diperbarui ketika perpindahan diperlukan.')

new('BAB III Metodologi lanjutan')
h('3.3 Representasi dan protokol training')
table('Konfigurasi kedua run',['Parameter','Tabular Q-Learning','Double DQN'],
      [('Observasi','7 kategori gabungan dalam satu kode','12 fitur float32'),
       ('Parameter belajar','α = 0,16 dan γ = 0,98','Adam 0,001, γ = 0,98 dan grad clip 10'),
       ('Epsilon training','0,30 di awal, lalu turun menurut jadwal fase','1,00 → 0,05 selama 60% langkah nominal'),
       ('Replay pengalaman','Tanpa minibatch','Kapasitas 100.000, batch 64, warm-up 1.000'),
       ('Target network','Tidak digunakan','Sinkron setiap 1.000 gradient update'),
       ('Episode dan seed','36, seed 42','36, seed 42'),
       ('Fase','18 eksplorasi dan 18 replay yang masih belajar','36 eksplorasi. Replay beku setelah evaluasi')],
      'run.json, training.csv dan kode trainer masing-masing.')
h('3.4 Evaluasi dan keterbandingan')
p('Pada evaluasi, seed DFS i = 0–11 dipasangkan dengan heading ((i mod 8) − 3,5) × π/4. Telemetry awal memverifikasi ketepatan heading. Kedua agen memilih secara greedy dengan epsilon nol dan tanpa pembaruan. Terdapat delapan heading unik. Penggunaan model final dan kondisi awal yang sama mendukung perbandingan berpasangan. Perbedaan alur update target, mask, jadwal epsilon dan realisasi langkah menjadikan desain keseluruhan PARTIALLY MATCHED. Heading training sama hanya pada pasangan pertama.')
h('3.5 Rekaman dan analisis')
p('Nomor baris yang stabil menghubungkan telemetry dan frame. Reward, durasi, langkah, event dan hasil akhir direkonsiliasi dengan training.csv dan evaluations.json. Success berarti goal tercapai. Satu episode dapat mengalami kontak dan tetap sukses. Rerata bergerak memakai lima episode terakhir dengan jendela parsial di awal. Panjang lintasan memakai integrasi posisi internal dan durasi dinyatakan dalam detik simulasi. Waktu dinding aktif merupakan jumlah waktu per episode lengkap.')
p('Uji skoring tambahan mengukur lookup atau forward pass enam Q pada satu CPU logis yang sama. Observasi disiapkan sebelumnya dari keputusan evaluasi aktual. Uji dilakukan dalam batch ukuran satu, 21 putaran, dengan pemanasan sebelum tiap putaran. Biaya encoding, masking, pemilihan aksi akhir dan simulasi berada di luar ukur.')

new('BAB IV Implementasi')
h('4.1 Tabular Q-Learning')
p('Bearing dibagi delapan interval berlebar π/4. Jarak ke waypoint dikelompokkan pada batas 0,10 m, 0,25 m, dan 0,50 m. Lima minimum LiDAR masing-masing dikelompokkan pada batas 0,16 m dan 0,45 m. Kombinasi 8 × 4 × 3⁵ memberi batas teoritis 7.776 state. Tabel jarang mengalokasikan enam float32 per state yang dimasukkan. Visits meningkat pada pemilihan aksi agen selama training. Pemindaian dan mundur melewati pembaruan tabular. Lookup nilai Q untuk evaluasi bersifat baca-saja.')
h('4.2 Double DQN')
p('Encoder menghasilkan sin dan cos bearing, jarak dibagi diagonal arena, lima jarak LiDAR dibagi 4 m, dua fitur laju dari roda yang dinormalisasi, serta dua flag scanning dan recovery. Jaringan 12 → 128 → 128 → 6 dengan ReLU memiliki 18.950 parameter. Jaringan online mendapat gradient Smooth L1 dan optimizer Adam, jaringan target disinkronkan berkala tanpa gradient. Aksi berikutnya dipilih dari mask valid dengan Q online dan dinilai oleh jaringan target. Buffer menyimpan status terminated dan truncated terpisah.')
h('4.3 Integrasi sensor dan eksekusi')
p('WaypointRecovery menyaring aksi berdasarkan jarak dinding, orientasi dan stagnasi. Saat tidak ada aksi aman, stop menjadi fallback. DQN menambahkan irisan mask fisik yang dihitung kembali dari observasi terukur dan ternormalisasi. Scanning memilih pivot melalui DFS. Recovery kontak dijalankan oleh mesin gerak. Adapter DQN mengecek terminal, timeout, dan status recovery sebelum memperbarui target. Percabangan loop Tabular membedakan kondisi scanning dari waypoint tercapai.')
h('4.4 Logging dan evaluasi')
table('Arsitektur kebijakan lokal',['Aspek','Tabular Q-Learning','Double DQN'],
      [('Representasi','Kode tujuh bin','Vektor 12 fitur'),
       ('Nilai aksi','Enam nilai per state','Enam keluaran jaringan'),
       ('Penyimpanan','Tabel jarang dan visits','Bobot online dan target'),
       ('Belajar','Pembaruan TD skalar','Gradient Smooth L1 pada batch replay'),
       ('Optimizer dan replay','Pembaruan langsung','Adam dan replay buffer'),
       ('Pemilihan','Epsilon-greedy dengan mask','Epsilon-greedy dengan mask')],
      'lidar_qlearning.py, lidar_dqn_agent.py, lidar_dqn_network.py, dan run.json.')
p('Recorder menyimpan observasi sebelum aksi, pose setelah aksi, reward serta event dari simulasi. Sumber Q mencatat state, visits, galat TD, dan pembaruan. Sumber Double DQN mencatat vektor observasi, keluaran Q, dan pembaruan minibatch. Field pembelajaran bersifat opsional pada langkah tanpa update. Checkpoint Double DQN terakhir digunakan dalam evaluasi. Versi best_model.pt tersedia untuk tujuan pelacakan artefak.')

new('BAB V Hasil dan Pembahasan')
h('5.1 Keterbandingan dan keberhasilan')
def score(scope):
    return [(key,*(n(s[scope][metric],prec) for s in (q,d)))
            for key,metric,prec in [
              ('Episode','episodes',0),('Sukses','success',0),('Gagal','failed',0),
              ('Keberhasilan (%)','success_rate',2),('Episode dengan kontak','collision',0),
              ('Event kontak','collision_count_total',0),('Sukses dengan kontak','success_after_collision',0),
              ('Timeout','timeout',0),('DFS habis','search_exhausted',0)]]
table('Hasil training dengan jadwal fase berbeda',['Metrik','Tabular','Double DQN'],score('training'),'training.csv dan episode_metrics.csv.')
table('Hasil evaluasi dengan kondisi awal berpasangan',['Metrik','Tabular','Double DQN'],score('evaluation'),'matched_evaluation.csv.')
p('Tabular berhasil 11/18 pada eksplorasi dan 18/18 setelah beralih ke replay. Double DQN berhasil 6/18 dalam paruh pertama dan 12/18 dalam paruh kedua, saat tetap bereksplorasi. Episode gagal pada kedua run berakhir melalui timeout. Pada evaluasi beku, seed 7, 10, dan 11 gagal pada Tabular dan sukses pada Double DQN, dengan kondisi awal sesuai pasangan. Keberhasilan yang lebih tinggi pada training Tabular perlu dibaca bersama perbedaan tugas setelah eksplorasi.')

new('BAB V Reward dan Kinerja Eksplorasi')
h('5.2 Reward training')
table('Distribusi reward training',['Metrik','Tabular','Double DQN'],[
  (lbl,n(q['training']['reward'][key]),n(d['training']['reward'][key]))
  for key,lbl in [('mean','Mean'),('median','Median'),('sd','Simpangan baku'),('min','Minimum'),('max','Maksimum')]
  ]+[('Rerata 5 episode akhir',n(q['training']['final_5_reward']),n(d['training']['final_5_reward']))],
  'training.csv dan comparison_summary.json.')
graph('reward_comparison','Reward training per episode. Bayangan menandai episode replay Tabular.',14.8)
graph('moving_average_comparison','Rerata bergerak lima episode, dihitung dari seluruh training.',14.8)
p('Penalti kontak sebesar −100 membuat beberapa episode dengan kontak berulang menurunkan mean di bawah median. Reward lebih tenang pada replay Tabular, yang mencatat empat kontak dalam 18 episode. Pada DQN, kontak turun dari 821 menjadi 317 antara dua paruh, dengan empat kontak pada lima episode akhir. Maksimum reward DQN yang lebih tinggi muncul bersamaan dengan mean yang lebih rendah. Pemisahan fase penting untuk membaca perubahan variasi.')

new('BAB V Durasi dan Efisiensi Navigasi')
h('5.3 Langkah, durasi, dan panjang lintasan')
qa,da=q['evaluation'],d['evaluation']
vals=[
 ('Rerata langkah',qa['steps']['mean'],da['steps']['mean']),
 ('Median langkah',qa['steps']['median'],da['steps']['median']),
 ('Rerata durasi (s)',qa['duration_s']['mean'],da['duration_s']['mean']),
 ('Median durasi (s)',qa['duration_s']['median'],da['duration_s']['median']),
 ('Rerata durasi sukses (s)',qa['mean_success_s'],da['mean_success_s']),
 ('Tercepat sukses (s)',qa['best_success_s'],da['best_success_s']),
 ('Rerata panjang lintasan (m)',qa['distance_m']['mean'],da['distance_m']['mean']),
 ('Rerata kecepatan (m/s)',qa['avg_speed_m_s']['mean'],da['avg_speed_m_s']['mean']),
 ('Total langkah scanning',qa['scan_steps_total'],da['scan_steps_total']),
 ('Event waypoint termasuk goal',qa['waypoint_events_total'],da['waypoint_events_total']),
 ('Total durasi mundur (s)',qa['recovery_steps_total']*.16,da['recovery_steps_total']*.16),
]
table('Metrik navigasi pada evaluasi beku',['Metrik','Tabular','Double DQN','DQN − Q'],
      [(label,n(a,3 if 'kecepatan' in label else 2),n(b,3 if 'kecepatan' in label else 2),
        n(b-a,3 if 'kecepatan' in label else 2)) for label,a,b in vals],
      'evaluations.json, matched_evaluation.csv dan common_telemetry.jsonl.')
graph('matched_evaluation_success_rate','Keberhasilan dan durasi per pasangan. Timeout tercatat pada 300 s.',15.6)
p('Pada sembilan pasangan sukses bersama, selisih durasi DQN dikurangi Tabular bermean −16,16 s, bermedian −7,20 s, dan bersimpangan baku 58,97 s. DQN lebih cepat pada tujuh dari sembilan pasangan sukses bersama. Seed 8 dan 9 menunjukkan arah sebaliknya. Waktu terbaik pada semua fase adalah 113,60 s untuk replay dalam training Tabular dan 98,72 s untuk replay beku DQN. Kedua rekor berasal dari kandidat rute dan protokol yang berbeda.')

new('BAB V Pembelajaran dan Representasi')
h('5.4 State, visits, dan parameter')
p('Q-table final menyimpan 888 baris atau 11,42% dari batas 7.776 state. Total visits 37.766, mean 42,53, dan median 5. Satu state mempunyai nol visits, 461 mempunyai 1–5, 221 mempunyai 6–20, dan 205 di atas 20. Sebagian besar state teoretis tidak muncul di tabel. Nilai awal nol dipakai untuk state yang belum tersedia. DQN menyimpan 52.148 transisi, menjalankan 51.149 gradient update, dan menyinkronkan target 51 kali. Neural network dapat memanfaatkan pola fitur serupa, sedangkan kemampuan transfer antarmaze belum diuji.')
graph('q_value_comparison','Tabular memakai Q terpilih sebelum update. Panel DQN memakai nilai batch.',15.5)
graph('td_error_loss_comparison','Galat TD absolut Tabular dan loss Smooth L1 DQN pada panel terpisah.',15.5)
p('Rerata galat TD absolut Tabular turun dari 19,20 ke 0,78 antara episode pertama dan terakhir. Pada Double DQN, loss awal sekitar 0,72 dan akhir sekitar 1,52, dengan variasi selama training. Loss dipengaruhi perubahan distribusi replay dan target. Nilai Q maksimum dapat naik ketika rerata turun. Metrik internal dan evaluasi memberi informasi berbeda tentang perubahan kebijakan. Bukti lintas run diperlukan untuk menilai kestabilan dan konvergensi.')

new('BAB V Efisiensi Training dan Inferensi')
h('5.5 Waktu dinding, ukuran model, dan latensi')
bench=m['benchmark']['latency_us']
table('Biaya teramati pada run rekaman dan benchmark CPU',['Metrik','Tabular','Double DQN'],[
 ('Langkah training','42.710','52.148'),
 ('Waktu dinding episode aktif (s)',n(q['training']['wall_s_total']),n(d['training']['wall_s_total'])),
 ('Rerata wall-clock per episode (s)',n(q['training']['wall_s_total']/36),n(d['training']['wall_s_total']/36)),
 ('Pembaruan','37.766 pembaruan Q','51.149 gradient update'),
 ('Nilai Q atau bobot online (byte)',n(mq['weight_bytes'],0),n(md['weight_bytes'],0)),
 ('Q-table JSON atau last_model.pt (byte)',n(mq['file_bytes'],0),n(md['file_bytes'],0)),
 ('Array replay teralokasi (byte)','Tidak diperlukan',n(md['replay_allocated_bytes'],0)),
 ('Median skoring batch 1 (μs)',n(bench['q_learning']['median'],3),n(bench['drl']['median'],3)),
 ],'episode_summary.csv, comparison_summary.json dan inference_benchmark.json.')
graph('inference_latency_comparison','Skoring enam Q pada CPU i5-12450H dengan satu thread PyTorch.',14.7)
p('Training Double DQN mencatat lebih banyak langkah, pembaruan gradient dan waktu dinding aktif yang lebih panjang. Kedua jadwal dijalankan pada CPU dan beban sistem historis tidak dicatat. Waktu tersebut meliputi kerja simulasi dan pencatatan. Checkpoint Double DQN membawa metadata, optimizer serta jaringan target. Array replay dicatat terpisah. Angka array dan bobot tidak mencakup overhead alokasi proses.')
p(f'Benchmark membandingkan {m["benchmark"]["decisions_per_batch"]} observasi per putaran. Waktu skoring mendukung Tabular bila latensi fungsi nilai menjadi prioritas. Pengukuran satu komponen ini tidak mengukur throughput keseluruhan robot, termasuk sensor, pengendali, recovery dan fisika simulasi.')

new('BAB V Trajectory dan Bukti Video')
h('5.6 Analisis lintasan dan kegagalan')
graph('trajectory_comparison','Pose setelah aksi pada seed DFS 0 dan 7. Tanda silang menandai event kontak.',14.4)
p('Pada seed 0, Tabular memerlukan 244,96 s dan Double DQN 117,76 s. Keduanya sukses tanpa kontak. Lintasan Tabular mengunjungi lebih banyak bagian arena sebelum mencapai goal. Pada seed 7, Tabular timeout setelah tiga kontak dan menyisakan sekitar 5,39 m ke goal. Double DQN mencapai goal dalam 118,88 s tanpa kontak. Perbedaan pose dan heading ketika mengamati cabang dapat mengubah rute DFS walaupun posisi awal dan seed sama. Pada seed 8, Double DQN masih mencapai goal, namun durasi 293,28 s mendekati batas waktu.')

new('BAB V Pembahasan dan Keterbatasan')
table('Penilaian dengan kriteria terpisah',['Kriteria','Bukti hasil','Penilaian'],
      [('Navigasi beku','9/12 vs 12/12 serta 16 vs 0 kontak','Double DQN pada maze ini'),
       ('Reward training','Mean −1.309,40 vs −2.977,79','Q pada data run. Fase berbeda'),
       ('Durasi navigasi','Mean evaluasi 196,19 vs 140,21 s','Double DQN pada data ini'),
       ('Komputasi training','176,32 vs 302,31 s aktif','Q dalam run ini. Budget langkah berbeda'),
       ('Inferensi dan ukuran','Lookup lebih cepat dan payload lebih kecil','Q pada CPU, input siap'),
       ('Kestabilan','Epsilon dan tugas berbeda, satu seed','Data terbatas'),
       ('Generalisasi','Satu maze dan satu model final per metode','Belum diuji lintas maze'),
       ('Interpretasi','Nilai per bin dan visits tersedia langsung','Q untuk inspeksi keputusan'),
       ('Kompleksitas','DQN perlu dua jaringan, replay, optimizer','Q memiliki komponen lebih sedikit'),
       ('Reproduksibilitas','Log, artefak, hash dan frame tersedia','Kedua run dapat diaudit')],
      'comparison_summary.json, model, telemetry dan data_provenance.md.')
h('5.7 Kelebihan dan kekurangan')
p('Tabular mudah ditelusuri per state dan memberi representasi ringkas pada arena ini. Kuantisasi sudut dan jarak dapat menyatukan situasi yang memerlukan aksi berbeda. DQN menjaga resolusi fitur masukan dan berbagi bobot antarpengamatan. Ukuran replay, komputasi gradient dan pilihan hyperparameter menambah biaya dan sensitivitas. Kedua sistem bergantung pada LiDAR, DFS dan mekanisme keselamatan, sehingga keberhasilan merupakan hasil gabungan sistem.')
h('5.8 Keterbatasan')
p('Protokol training berbeda, termasuk fase replay, epsilon dan realisasi jumlah langkah. Run DQN dilanjutkan setelah episode lengkap 21 dan telemetry percobaan parsial dipisahkan. Evaluasi mengulang empat heading pada seed DFS berbeda. Evaluasi berpasangan bersifat deskriptif pada satu maze dan satu seed training. Pencocokan protokol secara penuh memerlukan loop kendali bersama agar pengaruh perbedaan adapter dapat dipisahkan.')

new('BAB VI Kesimpulan dan Saran')
h('6.1 Kesimpulan')
p('Kedua agen belajar memilih aksi diskret dalam lingkungan gerak kontinu. Kebijakan Double DQN mencapai goal lebih sering, dengan tabrakan lebih sedikit dan durasi rata-rata lebih pendek pada 12 evaluasi yang dipasangkan berdasarkan kondisi awal. Tabular mencatat lebih banyak keberhasilan pada training yang mencakup replay rute. Lookup tabel lebih mudah ditelusuri, menggunakan payload nilai Q lebih kecil dan lebih cepat pada benchmark skoring CPU.')
h('6.2 Jawaban atas rumusan masalah')
p('Tabular menyimpan nilai menurut kombinasi bin, sedangkan Double DQN menggunakan pendekatan fungsi dari 12 fitur. Perbedaan pola reward dan kontak selama training sejalan dengan perubahan epsilon dan perbedaan tugas. Pada evaluasi, laporan trajectory memverifikasi sukses, timeout dan lintasan yang berbeda. Variasi kedua run serta perbedaan skala galat membatasi penilaian kestabilan umum.')
h('6.3 Rekomendasi dan saran penelitian')
p('Untuk konfigurasi kinematik ini, Double DQN adalah pilihan jika prioritasnya keberhasilan navigasi dan minim kontak pada kondisi yang diuji. Tabular cocok jika prioritasnya menelusuri keputusan per bin atau mengurangi biaya skoring. Pengembangan berikutnya dapat menguji kedua kebijakan dengan jalur kontrol yang identik dan variasi posisi mulai, maze, dan beberapa seed training. Uji waktu dinding yang terisolasi dan pelaporan RAM proses akan melengkapi perbandingan biaya.')
h('Daftar pustaka')
for item in [
 '[1] Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction (edisi ke-2). MIT Press. https://mitpress.mit.edu/9780262039246/reinforcement-learning/',
 '[2] Watkins, C. J. C. H., & Dayan, P. (1992). Q-learning. Machine Learning, 8, 279–292. https://doi.org/10.1007/BF00992698',
 '[3] Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. Nature, 518, 529–533. https://doi.org/10.1038/nature14236',
 '[4] Van Hasselt, H., Guez, A., & Silver, D. (2016). Deep Reinforcement Learning with Double Q-Learning. Proceedings of AAAI, 30(1), 2094–2100. https://arxiv.org/abs/1509.06461',
 '[5] Thrun, S., Burgard, W., & Fox, D. (2005). Probabilistic Robotics. MIT Press. https://mitpress.mit.edu/9780262201629/probabilistic-robotics/',
 '[6] PyTorch. (2026). SmoothL1Loss, dokumentasi 2.15. https://docs.pytorch.org/docs/2.15/generated/torch.nn.SmoothL1Loss.html',
 '[7] FechL. RL-pathfinding. Commit arsitektur 5c4cf33. https://github.com/FechL/RL-pathfinding'
]: p(item)

new('Lampiran A Evaluasi Per Skenario')
table('Seed, heading dan durasi tiap pasangan evaluasi',['Seed','Heading (rad)','Tabular hasil / s / kontak','DQN hasil / s / kontak'],
      [(a['seed'],n(float(a['heading_rad']),3),
        ('Goal' if a['success']=='True' else 'Timeout')+' / '+n(float(a['duration_s']))+' / '+a['collision_count'],
        ('Goal' if b['success']=='True' else 'Timeout')+' / '+n(float(b['duration_s']))+' / '+b['collision_count'])
       for a,b in zip(pairs[::2],pairs[1::2])],
       'matched_evaluation.csv. Seluruh episode dijalankan dengan kebijakan akhir beku.')
graph(still,'Cuplikan video evaluasi komparatif, seed 0. HUD menunjukkan kondisi robot dari telemetry aktual.',16.0)
p('Gambar menampilkan maze, heading, target DFS, LiDAR, keputusan aksi dan waktu simulasi. Panel kiri menunjukkan Tabular dan panel kanan Double DQN. Saat satu panel selesai, pose serta waktu akhir panel itu dipertahankan hingga pasangan berikutnya.')

new('Lampiran B Data, Video dan Reproduksi')
descriptions={
  'training_comparison.mp4':'36 pasangan, diselaraskan ulang dari awal setiap episode.',
  'evaluation_side_by_side.mp4':'12 pasangan evaluasi pada jam simulasi bersama.',
  'best_route_comparison.mp4':'Frame MP4 asli dari setiap percobaan tercepat yang sukses.',
  'telemetry_comparison.mp4':'Inspeksi HUD evaluasi seed 1 dengan langkah yang lebih rapat.',
  'trajectory_comparison.mp4':'Lintasan progresif pada evaluasi seed 0 dan 7.',
}
table('Inventaris video komparatif',['Berkas','Isi','Durasi putar (s)'],
      [(f,descriptions[f],n(float(v['duration']))) for f,v in videos.items()],
      'data/video_manifest.json dan video_frame_mapping.csv.')
h('B.1 Cara memeriksa bukti')
p('Cari video dan frame pada data/video_frame_mapping.csv. Dua baris per frame memuat run_id, episode_key, step dan global_row. Cari identitas yang sama pada common_telemetry.csv atau common_telemetry.jsonl. Pada episode yang sudah selesai, held menandai panel yang dipertahankan. Interval sampling sama pada kedua panel untuk setiap video, dan waktu simulasi aktual tetap tercetak pada HUD.')
h('B.2 Inventaris dataset dan pemeriksaan')
p('Dataset meliputi 137.239 langkah, termasuk seluruh evaluasi dan 21 percobaan replay beku DQN. episode_metrics.csv adalah data per episode. comparison_metrics.csv berisi agregat berdasarkan tahap. matched_evaluation.csv dan JSON memuat pasangan kondisi awal dengan status protokol lengkap. source_sha256.json memuat hash input aktif. analysis/comparison_validation.md mencatat hasil pemeriksaan lintas sumber dan integritas video serta laporan.')
h('B.3 Reproduksi')
p('Dari root repository jalankan .venv/bin/python output/comparison/analysis/build_data.py, lalu skrip build_graphs.py dan build_videos.py pada lokasi yang sama. Paket memerlukan NumPy, PyTorch, OpenCV, Matplotlib, FFmpeg dan ffprobe. File audit berisi jalur sumber. Simpan report DOCX dan render sebagai PDF A4 melalui LibreOffice.')
doc.save(out/'report/Laporan_Perbandingan_Q_Learning_vs_Double_DQN_Rafliansyah.docx')

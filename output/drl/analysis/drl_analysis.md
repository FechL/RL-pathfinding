# Analisis Eksperimen Double DQN

Training menyelesaikan 36 episode, 52,148 langkah, dan 51,149 pembaruan gradien. Sebanyak 18 episode mencapai goal dan menghasilkan 7 kandidat.

Jaringan memiliki 18,950 parameter. Replay menyimpan state kontinu, reward, status terminal, truncation, serta mask aksi sensor untuk state berikutnya.

## Evaluasi beku

Model final menyelesaikan 12/12 episode. 0 episode memiliki tabrakan dengan 0 kejadian. Timeout 0, pencarian habis 0. Rerata reward 357.74, median 318.59. Rerata langkah 876.33, median 739.50. Rerata durasi 140.21 s dan jarak 23.13 m.

## Uji kecepatan

Rute 2 mencatat 98.72 detik simulasi pada percobaan 2 dari 21 pengujian. Rekor ini berlaku untuk rute yang diuji.

## Interpretasi

Rerata reward pada lima episode pertama -3766.72 dan lima episode terakhir 253.30. Reward dipengaruhi tabrakan, progres, dan durasi. Loss mengukur kesalahan terhadap target Bellman pada replay dan bergerak bersama perubahan data dan jaringan target. Nilai Q mengestimasi reward diskonto. Bukti keberhasilan navigasi berasal dari status goal pada evaluasi.

Kondisi awal, model, dan seed tercatat di data. Evaluasi memakai satu maze dan belum mengukur generalisasi.

## Webots terpisah

Status success pada 116.576 detik simulasi. Kontak 0. Hasil ini tidak digabung dengan evaluasi kinematik.

"""Generate measured training plots and an Indonesian experiment summary."""
import argparse
import csv
import json
from pathlib import Path
import statistics

from matplotlib.figure import Figure


def read_json(path):
    return json.loads(path.read_text())


def rolling(samples, window=5):
    return [statistics.mean(samples[max(0,i-window+1):i+1]) for i in range(len(samples))]


def render(path, title, x_label, y_label, draw, source='training.csv'):
    figure = Figure(figsize=(7,3.1),layout='constrained')
    ax = figure.subplots()
    draw(ax)
    ax.set(title=title,xlabel=x_label,ylabel=y_label)
    ax.grid(alpha=.18)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc='best',fontsize=8)
    figure.text(.01,.005,f'Sumber: {source}',fontsize=7,color='#505050')
    figure.savefig(path,dpi=172)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('output/drl'))
    args=parser.parse_args()
    data=args.output/'data'
    figures=args.output/'graphs'
    figures.mkdir(parents=True,exist_ok=True)
    rows=list(csv.DictReader((data/'training.csv').read_text().splitlines()))
    eps=[int(r['episode']) for r in rows]
    config=read_json(data/'run.json')
    summary=read_json(data/'summary.json')
    routes=read_json(data/'routes.json')
    if not rows:
        raise ValueError('No full training episodes to plot.')
    flag=lambda r,key:r[key]=='True'
    numeric=lambda key:[float(row[key]) for row in rows]
    reward=numeric('reward')

    def line(filename,title,label,key):
        samples=[(episode,float(row[key])) for episode,row in zip(eps,rows) if row[key]]
        if samples:
            x,y=zip(*samples)
            render(figures/f'{filename}.png',title,'Episode',label,
                   lambda ax:ax.plot(x,y,color='#2e6a85',label=label))

    line('reward_per_episode','Reward akumulasi per episode','Reward','reward')
    render(figures/'moving_average_reward.png','Reward dan rata-rata lima episode','Episode','Reward',
           lambda ax:(ax.plot(eps,reward,color='#89a7b5',label='Reward episode'),
                      ax.plot(eps,rolling(reward),color='#a84f33',label='Rata-rata 5 episode')))
    line('steps_per_episode','Langkah kontrol per episode','Langkah','steps')
    line('duration_per_episode','Durasi simulasi per episode','Detik simulasi','duration_s')
    render(figures/'success_rate.png','Keberhasilan dalam jendela lima episode','Episode','Berhasil (%)',
           lambda ax:ax.plot(eps,[100*r for r in rolling([flag(x,'success') for x in rows])],
                             color='#31724d',label='Mencapai goal'))
    render(figures/'collision_per_episode.png','Kejadian tabrakan per episode','Episode','Jumlah kejadian',
           lambda ax:ax.bar(eps,numeric('collision_count'),color='#a44b45',label='Tabrakan tercatat'))
    line('epsilon_decay','Nilai epsilon pada akhir episode','Epsilon','epsilon')
    line('td_loss','Rata-rata Smooth L1 loss per episode','Loss','td_loss')

    def q_values(ax):
        for key,label,color in [('mean_q','Rata-rata Q','#3e7d97'),('max_q','Q maksimum','#cb7434')]:
            x,y=zip(*[(episode,float(row[key])) for episode,row in zip(eps,rows) if row[key]])
            ax.plot(x,y,color=color,label=label)

    if any(row['mean_q'] for row in rows):
        render(figures/'q_value_evolution.png','Nilai Q pada batch pembaruan','Episode','Nilai Q',q_values)
    best=routes['best']
    trials=routes['trials']
    if trials:
        def speed(ax):
            for r in trials:
                ax.scatter(r['route_id'],r['duration_s'],s=18,alpha=.62,
                           color='#267258' if r['success'] else '#a54743',
                           marker='o' if r['success'] else 'x')
            ax.scatter([],[],color='#267258',label='Sukses')
            ax.scatter([],[],color='#a54743',marker='x',label='Gagal')
            if best:
                ax.scatter(best['route_id'],best['duration_s'],marker='*',s=120,
                           color='#dc9f23',label='Waktu sukses minimum',zorder=5)
        render(figures/'speed_test.png','Durasi uji rute dengan model beku','ID rute','Detik simulasi',speed,'routes.json')
    ev=summary['evaluation']
    train=summary['training']
    path=args.output/'analysis/drl_analysis.md'
    path.parent.mkdir(parents=True,exist_ok=True)
    lines=[
        '# Analisis Eksperimen Double DQN','',
        f"Training menyelesaikan {train['episodes']} episode, {train['steps']:,} langkah, "
        f"dan {train['gradient_updates']:,} pembaruan gradien. "
        f"Sebanyak {train['training_successes']} episode mencapai goal dan menghasilkan {train['routes_found']} kandidat.",
        '',
        f"Jaringan memiliki {config['parameters']:,} parameter. Replay menyimpan state kontinu, "
        'reward, status terminal, truncation, serta mask aksi sensor untuk state berikutnya.',
        '',
        '## Evaluasi beku','',
        f"Model final menyelesaikan {ev['successes']}/{ev['runs']} episode. "
        f"{ev['collision_episodes']} episode memiliki tabrakan dengan {ev['collision_events']} kejadian. "
        f"Timeout {ev['timeouts']}, pencarian habis {ev['search_exhausted']}. "
        f"Rerata reward {ev['average_reward']:.2f}, median {ev['median_reward']:.2f}. "
        f"Rerata langkah {ev['average_steps']:.2f}, median {ev['median_steps']:.2f}. "
        f"Rerata durasi {ev['average_duration_s']:.2f} s dan jarak {ev['mean_distance_m']:.2f} m.",
        '',
        '## Uji kecepatan','',
        (f"Rute {best['route_id']} mencatat {best['duration_s']:.2f} detik simulasi "
         f"pada percobaan {best['trial']} dari {len(trials)} pengujian. "
         'Rekor ini berlaku untuk rute yang diuji.' if best else
         f'BELUM ADA HASIL waktu tercepat. Jumlah percobaan yang tercatat {len(trials)}.'),
        '',
        '## Interpretasi','',
        f"Rerata reward pada lima episode pertama {statistics.mean(reward[:5]):.2f} dan "
        f"lima episode terakhir {statistics.mean(reward[-5:]):.2f}. "
        'Reward dipengaruhi tabrakan, progres, dan durasi. Loss mengukur kesalahan terhadap target Bellman pada replay dan bergerak bersama perubahan data dan jaringan target. Nilai Q mengestimasi reward diskonto. Bukti keberhasilan navigasi berasal dari status goal pada evaluasi.',
        '',
        'Kondisi awal, model, dan seed tercatat di data. Evaluasi memakai satu maze dan belum mengukur generalisasi.',
    ]
    out_webots=data/'webots/latest_run.json'
    if out_webots.is_file():
        web=read_json(out_webots)
        if web['trajectory']:
            def trajectory(ax):
                x, y = zip(*web['trajectory'])
                ax.plot(x,y,color='#247680',lw=1,label='Trajektori GPS')
                ax.scatter(*web['start'],color='#327b50',label='Start')
                ax.scatter(*web['goal'],color='#b96c2c',label='Goal')
                ax.set_aspect('equal',adjustable='datalim')
            render(figures/'webots_trajectory.png','Lintasan aktual Webots','x (m)','y (m)',
                   trajectory,'data/webots/latest_run.json')
        lines += ['', '## Webots terpisah','',
                  f"Status {web['status']} pada {web['simulation_duration_s']:.3f} detik simulasi. "
                  f"Kontak {web['collision_count']}. Hasil ini tidak digabung dengan evaluasi kinematik."]
    path.write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    main()

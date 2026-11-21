"""Graph archived training and paired frozen evaluation results."""
import csv
import json
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

out = Path('output/comparison')
with (out/'data/episode_metrics.csv').open() as f:
    data = list(csv.DictReader(f))
summary=json.loads((out/'data/comparison_summary.json').read_text())
names={'q_learning':'Tabular Q-Learning','drl':'Double DQN'}
palette=['#006e85','#af4c18']
training={a:[r for r in data if r['algorithm']==a and r['scope']=='training'] for a in names}
evaluation={a:[r for r in data if r['algorithm']==a and r['scope']=='evaluation'] for a in names}
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
def finish(fig, name, source):
    fig.text(.018,.018,source,fontsize=8,color='#555555',ha='left')
    fig.tight_layout(rect=(0,.075,1,1))
    fig.savefig(out/'graphs'/f'{name}.png',dpi=175)
    plt.close(fig)
def numeric(rows,key):
    return [float(r[key]) if r[key] not in ('True','False') else int(r[key]=='True') for r in rows]
def rolling(v,w=5):
    return [mean(v[max(0,i-w+1):i+1]) for i in range(len(v))]
def cumulative(v): return np.cumsum(v)
def identity(v): return v
def ax_label(ax,title,xlabel,ylabel):
    ax.set(title=title,xlabel=xlabel,ylabel=ylabel)
    ax.grid(alpha=.20,axis='y')
def line(ax,key,title,unit,fn=identity,groups=training):
    for (a,rs),c in zip(groups.items(),palette):
        ax.plot(numeric(rs,'episode'),fn(numeric(rs,key)),lw=1.8,label=names[a],color=c)
    ax_label(ax,title,'Episode',unit)
    ax.legend(loc='best',fontsize=8)
source='Sumber: training.csv kedua run rekaman dan episode_metrics.csv.'
caveat=' Q memakai replay saat training.'
curves = [
    ('reward_comparison','reward','Reward per episode','Reward',identity),
    ('moving_average_comparison','reward','Rata-rata bergerak reward (5 episode)','Reward',rolling),
    ('cumulative_reward_comparison','reward','Jumlah reward lintas episode','Reward',cumulative),
    ('best_reward_comparison','reward','Rekor reward hingga episode berjalan','Reward',np.maximum.accumulate),
    ('success_rate_comparison','success','Tingkat keberhasilan bergerak (5 episode)','Keberhasilan (%)',lambda v: np.array(rolling(v))*100),
    ('cumulative_success_comparison','success','Keberhasilan kumulatif','Episode sukses',cumulative),
    ('collision_comparison','collision_count','Kontak terhitung per episode','Jumlah event',identity),
    ('cumulative_collision_comparison','collision_count','Kontak terhitung kumulatif','Jumlah event',cumulative),
    ('steps_comparison','steps','Langkah per episode','Langkah',identity),
    ('duration_comparison','duration_s','Durasi per episode','Detik simulasi',identity),
    ('epsilon_comparison','epsilon_end','Epsilon pada akhir episode','Epsilon',identity),
    ('path_length_comparison','distance_m','Panjang lintasan per episode','Meter',identity),
    ('training_time_comparison','wall_s','Waktu dinding aktif kumulatif','Detik aktual',cumulative),
]
for file,k,title,y,fn in curves:
    fig,ax=plt.subplots(figsize=(7.5,3.45))
    line(ax,k,title,y,fn)
    if file!='epsilon_comparison':
        ax.axvspan(18.5,36.0,alpha=.07,color=palette[0])
    finish(fig,file,source+caveat)
def bars(ax, values, title, unit):
    p=ax.bar(list(names.values()),values,color=palette,width=.5)
    ax.bar_label(p,fmt='%.3g',padding=3)
    ax.margins(y=.2)
    ax_label(ax,title,'Algoritma',unit)
    ax.legend(p,list(names.values()),fontsize=7)
def stacked_panels(filename, labels, source):
    fig,axs=plt.subplots(1,len(labels),figsize=(8.5,3.35),squeeze=False)
    for ax,(label,unit,key,conv) in zip(axs[0],labels):
        bars(ax, [conv(summary['memory'][a][key]) for a in names],label,unit)
    finish(fig,filename,source)
fig,axes=plt.subplots(1,2,figsize=(8.5,3.65))
for ax,key,title in zip(axes,['timeout','search_exhausted'],['Timeout','Cabang DFS habis']):
    bars(ax,[summary['metrics'][a]['training'][key] for a in names],title,'Episode')
finish(fig,'failure_comparison',source+caveat)
fig,axes=plt.subplots(1,2,figsize=(8.5,3.65))
for ax,scope in zip(axes,['training','evaluation']):
    bars(ax,[summary['metrics'][a][scope]['mean_success_s'] for a in names],
         'Training' if scope=='training' else 'Evaluasi beku','Detik simulasi')
finish(fig,'successful_duration_comparison','Sumber: episode_metrics.csv. Rerata dihitung pada episode sukses masing-masing.')
for filename, specs in {
 'state_replay_growth_comparison': [('q_learning','q_states','Baris Q-table','State'),
             ('drl','replay_size','Replay buffer','Transisi'),('drl','updates','Pembaruan jaringan online','Update per episode'),
             ('drl','target_updates','Sinkronisasi jaringan target','Update kumulatif')],
 'q_value_comparison': [('q_learning','q_selected_mean','Tabular Q sebelum update pada aksi terpilih','Rerata Q terpilih'),
             ('drl','dqn_mean_q','Double DQN pada minibatch','Rerata Q batch')],
 'td_error_loss_comparison': [('q_learning','q_td_abs_mean','Tabular Q-Learning: galat TD absolut','Rerata |galat TD|'),
             ('drl','td_loss','Double DQN: Smooth L1','Loss minibatch')],
}.items():
    fig,axes=plt.subplots(len(specs)//2,2,figsize=(9,3.6 if len(specs)==2 else 6.2),squeeze=False)
    for ax,(a,k,title,unit) in zip(axes.flat,specs):
        rs=training[a]
        ax.plot(numeric(rs,'episode'),numeric(rs,k),color=palette[list(names).index(a)],label=names[a])
        if k=='dqn_mean_q':
            ax.plot(numeric(rs,'episode'),numeric(rs,'dqn_max_q'),color='#95417c',label='Q maksimum batch')
        ax_label(ax,title,'Episode',unit)
        ax.legend(fontsize=7)
    finish(fig,filename,'Sumber: training.csv, learning_updates.csv dan episode_metrics.csv. Definisi panel berbeda.')
fig,ax=plt.subplots(figsize=(6.9,3.5))
q=summary['memory']['q_learning']
p=ax.bar(['0','1–5','6–20','>20'],[q[f] for f in ('visits_0','visits_1_to_5','visits_6_to_20','visits_over_20')],color=palette[0],
         label=names['q_learning'])
ax.bar_label(p,padding=2)
ax.margins(y=.2)
ax_label(ax,'Distribusi kunjungan state tabular','Visits per state','Jumlah state')
ax.legend()
finish(fig,'visit_distribution','Sumber: q_learning/recording_run/q_table.json.')
stacked_panels('memory_comparison',[('Nilai Q / bobot online saja','KiB','weight_bytes', lambda v:v/1024),
              ('Artefak terserialisasi','KiB','file_bytes',lambda v:v/1024)],
              'Sumber: Q-table JSON dan last_model.pt (termasuk state training). Replay DQN dialokasikan 11,06 MiB.')
fig,ax=plt.subplots(figsize=(6.9,3.5))
values=[summary['benchmark']['latency_us'][a]['raw'] for a in names]
for i,(a,v) in enumerate(zip(names,values)):
    ax.scatter([i]*len(v),v,color=palette[i],alpha=.45,label=names[a])
    ax.scatter(i,np.median(v),marker='_',s=350,color='#262626',zorder=5)
ax.set_yscale('log')
ax.set_xticks([0,1],list(names.values()))
ax_label(ax,'Inferensi batch 1 pada CPU yang sama','Algoritma','Mikrodetik per skoring (log)')
ax.legend(fontsize=8)
finish(fig,'inference_latency_comparison','Sumber: inference_benchmark.json. Fitur siap pakai, skoring enam Q tanpa pipeline sensor.')
fig,axes=plt.subplots(1,2,figsize=(8.5,3.60))
bars(axes[0],[summary['metrics'][a]['evaluation']['best_success_s'] for a in names],
     'Tercepat pada evaluasi eksplorasi','Detik simulasi')
bars(axes[1],[float(summary['sources'][a]['best']['duration_simulation_s']) for a in names],
     'Rekor seluruh percobaan sukses','Detik simulasi')
finish(fig,'route_time_comparison','Sumber: evaluations.json dan recording_manifest.json. Panel kanan memiliki protokol berbeda.')
fig,axes=plt.subplots(1,2,figsize=(8.5,3.60))
bars(axes[0],[summary['metrics'][a]['evaluation']['success_rate'] for a in names],
     'Keberhasilan pada 12 pasangan awal','Keberhasilan (%)')
for (a,rs),c in zip(evaluation.items(),palette):
    axes[1].scatter(numeric(rs,'episode'),numeric(rs,'duration_s'),label=names[a],color=c)
axes[1].set_xticks(range(1,13))
axes[1].legend(fontsize=7)
ax_label(axes[1],'Durasi tiap pasangan','Pasangan (seed ditambah 1)','Detik simulasi')
finish(fig,'matched_evaluation_success_rate','Sumber: matched_evaluation.csv. Kondisi awal sama, adapter berbeda. Timeout tercatat sebagai 300 s.')

# Reuse actual positions and wall geometry for the paired trajectory plate.
import cv2
wanted={'evaluation-000001','evaluation-000008'}
fig,axes=plt.subplots(2,2,figsize=(8.2,7.7),squeeze=False)
for j,(a,root) in enumerate((('q_learning','q_learning'),('drl','drl'))):
    tele=Path('output')/root/'telemetry'
    m=json.loads((tele/'recording_manifest.json').read_text())
    tracks={k:[] for k in wanted}
    events={k:[] for k in wanted}
    with (out/'data/common_telemetry.csv').open() as stream:
        for r in csv.DictReader(stream):
            k=r['episode_key']
            if r['algorithm']!=m['algorithm'] or k not in wanted:
                continue
            point=(float(r['post_x']),float(r['post_y']))
            tracks[k].append(point)
            if r['collision_event']=='True':
                events[k].append(point)
    for i,k in enumerate(sorted(wanted)):
        ax=axes[i,j]
        xmin,xmax,ymin,ymax=m['bounds']
        ax.set(xlim=(xmin,xmax),ylim=(ymin,ymax),aspect='equal',xlabel='X (m)',ylabel='Y (m)')
        rs=next(r for r in evaluation[a] if r['episode_key']==k)
        ax.set_title(f"{names[a]} · seed {rs['seed']} · "+('goal' if rs['success']=='True' else 'timeout'))
        for w in m['walls']:
            x,y=w['center']; sx,sy=w['size']
            ax.add_patch(Rectangle((x-sx/2,y-sy/2),sx,sy,facecolor='#515761',edgecolor='none',alpha=.70))
        ax.plot(*zip(*tracks[k]),lw=1.05,color=palette[j],label='Lintasan aktual')
        ax.scatter(*m['start'],s=32,c='#2e904b',marker='o',label='Start',zorder=4)
        ax.scatter(*m['goal'],s=44,c='#dfb900',marker='*',label='Goal',zorder=4)
        if events[k]:
            ax.scatter(*zip(*events[k]),s=22,c='#d13225',marker='x',label='Kontak',zorder=5)
        ax.legend(fontsize=6,loc='upper left',ncol=2)
finish(fig,'trajectory_comparison','Sumber: common_telemetry.csv, pose sesudah aksi; dinding dari recording_manifest.json.')

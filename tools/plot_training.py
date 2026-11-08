"""Reproducible scientific plots from training monitor and checkpoint records.

Historical monitor files have only reward, length and wall-clock time. Never
infer success from return or interpret short failed episodes as faster solves.
"""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BLUE = '#2563a6'
ORANGE = '#c46a18'
GRAY = '#66717d'


class NoTrainingData(ValueError):
    """The training run has not finished an episode or evaluation yet."""


def read_monitor(path):
    path = Path(path)
    if not path.exists():
        return []
    lines = path.read_text().splitlines()
    reader = csv.DictReader(line for line in lines if not line.startswith('#'))
    rows = []
    for row in reader:
        # A live reader may see an incomplete final CSV row. Omit that row,
        # while rejecting malformed completed rows instead of inventing values.
        try:
            reward, length = float(row['r']), int(row['l'])
        except (ValueError, TypeError, KeyError) as error:
            if row is not None and row.get('t') is None:
                continue
            raise ValueError(f'Malformed training monitor row: {row}') from error
        if not math.isfinite(reward) or length < 1:
            raise ValueError('Monitor rewards must be finite and lengths positive')
        item = dict(reward=reward, steps=length)
        for key in ('is_success', 'collision'):
            value = row.get(key)
            if value is not None:
                if value not in ('True', 'False', 'true', 'false', '1', '0'):
                    raise ValueError(f'Malformed boolean {key}: {value}')
                item[key] = value.lower() in ('true', '1')
        rows.append(item)
    return rows


def read_checkpoints(path):
    path = Path(path)
    if not path.exists():
        return []
    records = json.loads(path.read_text())
    rows = []
    for record in records:
        n, successes = int(record['runs']), int(record['successes'])
        if n < 1 or not 0 <= successes <= n or not math.isfinite(record['mean_time_s']):
            raise ValueError('Invalid checkpoint evaluation counts or time')
        item = dict(record)
        # The final evaluation often repeats the last periodic checkpoint.
        # Remove only an exact duplicate, retaining genuinely different results.
        if rows and all(item[k] == rows[-1][k] for k in ('steps', 'runs', 'successes', 'mean_time_s')):
            continue
        rows.append(item)
    return rows


def rolling(values, window):
    values = np.asarray(values, dtype=float)
    sums = np.concatenate([[0.], np.cumsum(values)])
    end = np.arange(1, len(values)+1)
    begin = np.maximum(0, end-window)
    return (sums[end]-sums[begin])/(end-begin)


def _save(fig, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Replace complete images atomically so a live viewer never sees half a PNG.
    with tempfile.NamedTemporaryFile(dir=destination.parent,suffix='.png',delete=False) as file:
        temporary = Path(file.name)
    try:
        fig.savefig(temporary, dpi=160, facecolor='white')
        os.replace(temporary,destination)
    finally:
        temporary.unlink(missing_ok=True)


def generate(report_dir=ROOT/'reports/continuous', output_dir=None, window=20):
    if window < 1:
        raise ValueError('Rolling window must be positive')
    # Keep plotting optional for the lightweight inference controller.
    os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'micromouse-matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, MaxNLocator
    report = Path(report_dir)
    output = Path(output_dir) if output_dir else report/'figures'
    episodes = read_monitor(report/'training.monitor.csv')
    checkpoints = read_checkpoints(report/'checkpoints.json')
    if not episodes and not checkpoints:
        raise NoTrainingData(f'No training or evaluation data in {report}')
    manifest_path = report/'run.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    algorithm = {'naf':'NAF','rl':'SAC tanpa panduan rute'}.get(str(manifest.get('algorithm')).lower(),'SAC')
    dt = float(manifest.get('control_dt_s', .064))
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError('Control dt must be positive and finite')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,
                         'axes.titlesize':12,'axes.labelsize':10,
                         'axes.edgecolor':'#929ba5','text.color':'#253341',
                         'axes.labelcolor':'#253341','xtick.color':GRAY,'ytick.color':GRAY,
                         'grid.color':'#e1e5e9','grid.linewidth':.7})
    fig, axes = plt.subplots(2,2,figsize=(13,9))
    fig.subplots_adjust(left=.08,right=.97,bottom=.16,top=.87,wspace=.26,hspace=.5)
    fig.suptitle(f'Perkembangan latihan {algorithm}',x=.08,y=.97,ha='left',fontsize=21,fontweight='bold')
    fig.text(.08,.923,f'{len(episodes):,} episode selesai | {len(checkpoints)} evaluasi checkpoint | model latihan kinematik',fontsize=11,color=GRAY)
    for ax in axes.flat:
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y');ax.set_axisbelow(True)
    if episodes:
        x = np.arange(1,len(episodes)+1)
        reward = [row['reward'] for row in episodes]
        duration = np.array([row['steps'] for row in episodes])*dt
        for ax,values,color,title,label in (
            (axes[0,0],reward,BLUE,'Reward per episode latihan','Total reward'),
            (axes[0,1],duration,ORANGE,'Durasi episode latihan','Detik simulasi')):
            ax.plot(x,values,color=color,alpha=.25,lw=.8,label='Per episode')
            ax.plot(x,rolling(values,window),color=color,lw=2,label=f'Rata-rata hingga {window} episode terakhir')
            ax.set(title=title,xlabel='Episode pada sesi ini',ylabel=label)
            ax.set_xlim(.5,max(2,len(x)+.5));ax.xaxis.set_major_locator(MaxNLocator(integer=True,nbins=6))
            ax.legend(fontsize=8,loc='upper left',frameon=False)
        axes[0,1].set_ylim(bottom=0)
    else:
        for ax in axes[0]:
            ax.text(.5,.5,'Belum ada episode latihan selesai',ha='center',va='center',transform=ax.transAxes)
    if checkpoints:
        steps = np.array([r['steps'] for r in checkpoints])
        rates = np.array([r['successes']/r['runs'] for r in checkpoints])*100
        # Only complete success groups provide comparable completion times.
        # A short crash is never charted as an improved traversal time.
        times = np.array([r['mean_time_s'] if r['successes']==r['runs'] else np.nan for r in checkpoints])
        axes[1,0].plot(steps,times,'o-',color=BLUE,lw=1.8,ms=5)
        if np.any(np.isfinite(times)):
            index = int(np.nanargmin(times))
            axes[1,0].scatter([steps[index]],[times[index]],s=95,marker='D',facecolors='white',edgecolors=ORANGE,zorder=5)
            axes[1,0].annotate(f'Tercepat {times[index]:.3f} dtk',xy=(steps[index],times[index]),
                               xytext=(12,20),textcoords='offset points',fontsize=9,color=ORANGE)
            finite = times[np.isfinite(times)]
            margin = max(.15,float(np.ptp(finite))*.3)
            axes[1,0].set_ylim(float(finite.min())-margin,float(finite.max())+margin)
        else:
            axes[1,0].text(.5,.5,'Belum ada checkpoint yang lulus semua uji',ha='center',va='center',transform=axes[1,0].transAxes,wrap=True)
            axes[1,0].set_yticks([])
        axes[1,0].set(title='Waktu mencapai goal saat evaluasi (skala diperbesar)',
                      xlabel='Langkah lingkungan terakumulasi',ylabel='Detik; hanya checkpoint lulus semua uji')
        axes[1,1].plot(steps,rates,'s-',color=BLUE,lw=1.8,ms=5,label='Start misi penuh')
        if all('local_runs' in row for row in checkpoints):
            local_rates = [100*row['local_successes']/row['local_runs'] for row in checkpoints]
            axes[1,1].plot(steps,local_rates,'o-',color=ORANGE,lw=1.8,ms=5,label='Start dekat goal')
            axes[1,1].legend(frameon=False,fontsize=8)
        else:
            for i,row in enumerate(checkpoints):
                axes[1,1].annotate(f"{row['successes']}/{row['runs']}",xy=(steps[i],rates[i]),
                                   xytext=(0,-18 if rates[i]>=90 else 9),textcoords='offset points',ha='center',fontsize=9)
        axes[1,1].set(title='Keberhasilan evaluasi checkpoint',
                      xlabel='Langkah lingkungan terakumulasi',ylabel='Berhasil mencapai goal (%)',ylim=(0,108))
        for ax in axes[1]:
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x,_:f'{x/1000:g}k'))
            if len(set(steps))<=7:ax.set_xticks(sorted(set(steps)))
    else:
        for ax in axes[1]:
            ax.text(.5,.5,'Evaluasi checkpoint belum tersedia',ha='center',va='center',transform=ax.transAxes)
    fig.text(.08,.092,'Posisi start dan kesulitan latihan berubah. Episode pendek dapat berakhir menabrak; bukan selalu lebih cepat.',fontsize=9,color=GRAY)
    evaluation_note = ('Evaluasi kinematik memakai policy deterministik dari start misi dan posisi dekat goal.'
                       if all('local_runs' in row for row in checkpoints) and checkpoints else
                       'Evaluasi memakai start misi dan policy deterministik pada model kinematik.')
    fig.text(.08,.067,evaluation_note,fontsize=9,color=GRAY)
    fig.text(.08,.041,'Sumber: training.monitor.csv dan checkpoints.json pada folder laporan yang dipilih.',fontsize=9,color=GRAY)
    overview = output/'training_progress.png'
    _save(fig,overview)
    # Export a larger evaluation panel for reading small time differences.
    eval_fig = plt.figure(figsize=(9,5))
    eval_ax = eval_fig.add_subplot()
    if checkpoints:
        eval_ax.plot(steps,times,'o-',color=BLUE,lw=2)
        for index, (step,value) in enumerate(zip(steps,times)):
            if math.isfinite(value):
                offset = 10 if index % 2 == 0 or len(steps) < 8 else -16
                eval_ax.annotate(f'{value:.3f}',(step,value),xytext=(0,offset),textcoords='offset points',ha='center',fontsize=9)
        eval_ax.xaxis.set_major_formatter(FuncFormatter(lambda x,_:f'{x/1000:g}k'))
        eval_ax.set_xticks(sorted(set(steps)))
        if np.any(np.isfinite(times)):eval_ax.set_ylim(float(np.nanmin(times))-.18,float(np.nanmax(times))+.18)
        else:
            eval_ax.text(.5,.5,'Belum ada checkpoint yang lulus semua uji',ha='center',va='center',transform=eval_ax.transAxes)
            eval_ax.set_yticks([])
    eval_ax.spines[['top','right']].set_visible(False)
    eval_ax.grid(axis='y',color='#e1e5e9')
    eval_ax.set(title=f'Waktu evaluasi {algorithm} dari start misi',xlabel='Langkah lingkungan terakumulasi',ylabel='Detik simulasi (skala diperbesar)')
    eval_fig.text(.12,.035,'Hanya checkpoint yang berhasil pada seluruh uji; posisi awal uji sama, dengan perturbasi kecil.',fontsize=9,color=GRAY)
    eval_fig.subplots_adjust(left=.12,right=.96,top=.87,bottom=.2)
    _save(eval_fig,output/'evaluation_time.png')
    plt.close(fig);plt.close(eval_fig)
    if episodes and all('is_success' in row and 'collision' in row for row in episodes):
        outcomes, ax = plt.subplots(figsize=(9, 4.6), layout='constrained')
        x = np.arange(1, len(episodes)+1)
        for key, color, label in [('is_success', BLUE, 'Mencapai goal'), ('collision', ORANGE, 'Tabrakan')]:
            ax.plot(x, 100*rolling([row[key] for row in episodes], window), color=color, lw=2, label=label)
        ax.set(title=f'Hasil episode latihan {algorithm} (hingga {window} episode terakhir)',
               xlabel='Episode pada sesi ini', ylabel='Persentase (%)', ylim=(-2, 102))
        ax.grid(axis='y')
        ax.legend(frameon=False)
        _save(outcomes, output/'episode_outcomes.png')
        plt.close(outcomes)
    metadata = dict(completed_episodes=len(episodes),completed_episode_steps=sum(r['steps'] for r in episodes),
                    checkpoint_evaluations=len(checkpoints),rolling_window=window,control_dt_s=dt,
                    episode_success_recorded=bool(episodes) and all('is_success' in r for r in episodes),
                    sources=[str(report/'training.monitor.csv'),str(report/'checkpoints.json')],
                    notes=['Short failed episodes are not faster solves.',
                           'Duplicate final checkpoint omitted only if values match.',
                           'Unknown historic success/collision data are not inferred from rewards.',
                           'Training curriculum changes; checkpoint evaluations use mission starts.',
                           'Kinematic training/evaluation, not Webots physics.'])
    output.mkdir(parents=True,exist_ok=True)
    (output/'plot_data.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return overview


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--report-dir',type=Path,default=ROOT/'reports/continuous')
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--window',type=int,default=20)
    parser.add_argument('--watch',action='store_true',help='Regenerate PNGs while training writes new data')
    parser.add_argument('--interval',type=float,default=5.)
    args=parser.parse_args()
    if args.window<1 or args.interval<=0:parser.error('window and interval must be positive')
    try:
        while True:
            try:
                print(generate(args.report_dir,args.output_dir,args.window),flush=True)
            except NoTrainingData:
                if not args.watch:raise
                print('Menunggu episode atau evaluasi pertama...',flush=True)
            except json.JSONDecodeError:
                if not args.watch:raise
                print('Log evaluasi sedang diperbarui; mencoba lagi pada pembaruan berikutnya.',flush=True)
            if not args.watch:break
            time.sleep(args.interval)
    except KeyboardInterrupt:pass


if __name__=='__main__':main()

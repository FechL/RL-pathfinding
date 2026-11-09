"""Train SAC with continuous v/yaw commands; export a numpy inference actor."""
import argparse
import copy
import csv
from collections import deque
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'controllers/micromouse'))
sys.path.insert(0,str(ROOT/'tools'))


def navigation(*, guided=True):
    from train_q_learning import read_world
    from continuous_control import NavigationMap
    env = read_world()
    return NavigationMap(env.start,env.goal,env.walls,env.bounds,env.cell_size,guided=guided)


def export_actor(model, nav, path, training, algorithm=None):
    import torch
    layers = [m for m in model.actor.latent_pi if isinstance(m,torch.nn.Linear)]+[model.actor.mu]
    data = nav.metadata()
    data.update(algorithm=algorithm or data['algorithm'],training=training,
                layers=[dict(weight=m.weight.detach().cpu().tolist(),bias=m.bias.detach().cpu().tolist()) for m in layers])
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(data,indent=2)+'\n')


def evaluate(actor,nav,seeds=(1000,1001,1002),max_steps=None,*,local=False):
    from continuous_env import ContinuousDriveEnv
    from planner import path_length, physical_clearance
    from motion_metrics import motion_metrics
    records=[]
    for seed in seeds:
        env=ContinuousDriveEnv(nav,training=False,max_steps=max_steps)
        obs,_=env.reset(seed=seed,options={'jitter':seed!=1000,'local':local})
        path=[env.pose[:2].tolist()]; commands=[]
        for _ in range(env.max_steps):
            action=actor(obs)
            obs,_,terminated,truncated,info=env.step(action)
            path.append(env.pose[:2].tolist());commands.append(env.wheels.tolist())
            if terminated or truncated: break
        clearance=min(physical_clearance(p,nav.walls,nav.bounds) for p in path)
        records.append(dict(seed=seed,**info,travelled_length_m=path_length(path),min_physical_clearance_m=clearance,
                            trajectory=path,wheel_commands=commands,
                            motion=motion_metrics([(i+1)*nav.config.dt for i in range(len(commands))],commands)))
    return records


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--steps',type=int)
    parser.add_argument('--episodes',type=int,help='Finish this many complete training episodes')
    parser.add_argument('--eval-every-episodes',type=int,default=100,
                        help='Evaluate mission starts after this many completed episodes')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--warm-start',action=argparse.BooleanOptionalAction,default=False,help='Explicit continuous driving demonstration pretraining')
    parser.add_argument('--no-route-guide',action='store_true',help='Learn with SAC using Euclidean goal progress and no Dijkstra or demonstrations')
    parser.add_argument('--rl-mode',action='store_true',help='Train a continuous RL policy for the selected route-guidance mode')
    parser.add_argument('--mission-only',action=argparse.BooleanOptionalAction,default=False,
                        help='Reset every training episode to the exact mission start')
    parser.add_argument('--timeout',type=float,help='Episode limit in simulated seconds for training and mission evaluation')
    parser.add_argument('--learning-rate',type=float,help='Override the SAC optimizer learning rate for fine-tuning')
    parser.add_argument('--teacher-anchor-weight',type=float,default=0.,
                        help='Optional reward penalty for straying from the initial actor (changes the training objective)')
    parser.add_argument('--evaluate-only',action='store_true')
    parser.add_argument('--run-dir',type=Path,help='Empty output folder; defaults to a new timestamped run')
    parser.add_argument('--policy',type=Path,help='Actor for --evaluate-only')
    parser.add_argument('--plots',action=argparse.BooleanOptionalAction,default=True,help='Export training PNGs at checkpoints')
    parser.add_argument('--activate',action='store_true',help='Use the best successful actor as the default for Webots')
    parser.add_argument('--render',action=argparse.BooleanOptionalAction,default=False,help='Show the robot during training in a local 2D window')
    parser.add_argument('--render-speed',type=float,default=8.,help='Preview speed relative to simulation time; 1 means real time')
    args=parser.parse_args()
    if args.episodes is not None and args.steps is not None:
        parser.error('Choose either --episodes or --steps, not both.')
    if args.episodes is not None and args.episodes<1: parser.error('--episodes must be positive')
    if args.steps is None: args.steps=300000
    if args.steps<1: parser.error('steps must be positive')
    if args.eval_every_episodes<1: parser.error('--eval-every-episodes must be positive')
    if args.no_route_guide and args.warm_start:
        parser.error('The no-route-guide mode learns without demonstrations.')
    if args.mission_only and not (args.no_route_guide or args.rl_mode):
        parser.error('--mission-only is available for RL training.')
    if not math.isfinite(args.render_speed) or args.render_speed<=0:parser.error('render-speed must be positive and finite')
    if args.timeout is not None and (not math.isfinite(args.timeout) or args.timeout<=0):
        parser.error('--timeout must be positive and finite.')
    if args.learning_rate is not None and (not math.isfinite(args.learning_rate) or args.learning_rate<=0):
        parser.error('--learning-rate must be positive and finite.')
    if not math.isfinite(args.teacher_anchor_weight) or args.teacher_anchor_weight<0:
        parser.error('--teacher-anchor-weight must be finite and nonnegative.')
    import torch
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.type_aliases import TrainFreq,TrainFrequencyUnit
    from continuous_env import ContinuousDriveEnv
    from continuous_control import read_policy
    torch.set_num_threads(1)
    nav=navigation(guided=not args.no_route_guide)
    timeout_seconds = args.timeout if args.timeout is not None else nav.config.max_seconds
    evaluation_steps = math.ceil(timeout_seconds / nav.config.dt)
    training_steps = evaluation_steps
    algorithm='rl' if args.rl_mode and nav.guided else ('sac' if nav.guided else 'rl_unguided')
    actor_name='continuous_actor.json' if algorithm=='sac' else 'rl_actor.json' if nav.guided else 'rl_unguided_actor.json'
    report_root=ROOT/'reports'/('continuous' if algorithm=='sac' else 'rl_guided' if nav.guided else 'rl')
    reward_description=('12 * geodesic progress' if nav.guided else '12 * Euclidean goal progress')
    reward_description+=' - elapsed seconds - wheel slew penalty; +30 goal; -15 collision'
    if args.rl_mode and nav.guided and args.teacher_anchor_weight:
        reward_description+=' - teacher action deviation penalty'
    if args.evaluate_only:
        report=args.run_dir or report_root
        report.mkdir(parents=True,exist_ok=True)
        actor,_=read_policy(args.policy or ROOT/'models'/actor_name,nav,algorithm)
        records=evaluate(actor,nav,max_steps=evaluation_steps)
        if not nav.guided and not args.mission_only:
            local_records=evaluate(actor,nav,seeds=range(2010,2020),max_steps=200,local=True)
            (report/'local_evaluation.json').write_text(json.dumps(local_records,indent=2)+'\n')
            print(f"Near goal: {sum(r['is_success'] for r in local_records)}/{len(local_records)}",flush=True)
        (report/'evaluation.json').write_text(json.dumps(records,indent=2)+'\n')
        print(json.dumps([{k:v for k,v in r.items() if k not in ('trajectory','wheel_commands')} for r in records],indent=2))
        if not all(r['is_success'] for r in records):raise SystemExit('Continuous policy evaluation failed')
        return
    report=args.run_dir or report_root/'runs'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    if any((report/name).exists() for name in ('run.json','training.monitor.csv','training_metrics.csv','checkpoints.json','summary.json')):
        parser.error('Run folder already contains training data. Choose a new --run-dir to preserve the previous run.')
    report.mkdir(parents=True,exist_ok=True)
    model_dir=report/'models';model_dir.mkdir(exist_ok=True)
    policy=model_dir/actor_name
    print(f'Training run: {report.resolve()}',flush=True)
    def plot_progress():
        if args.plots:
            from plot_training import generate
            generate(report)
    base=ContinuousDriveEnv(nav,training=not args.mission_only,max_steps=training_steps)
    teacher_sha=None
    if args.rl_mode and nav.guided and args.teacher_anchor_weight:
        teacher_path=ROOT/'models/continuous_actor.json'
        base.teacher_actor,teacher_metadata=read_policy(teacher_path,nav,'sac')
        base.teacher_anchor_weight=args.teacher_anchor_weight
        teacher_sha=teacher_metadata['artifact_sha256']
    viewer=None
    if args.render:
        import os
        if sys.platform.startswith('linux') and not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
            parser.error('Tidak ada layar grafis. Gunakan --no-render untuk training tanpa jendela.')
        from training_view import TrainingView
        viewer=TrainingView(nav,args.render_speed)
        base.reset(seed=args.seed)
        viewer.stage=('Menyiapkan training' if args.warm_start else
                      'Training RL (SAC)' if args.rl_mode else
                      'Training SAC tanpa rute' if not nav.guided else 'Training SAC')
        viewer.update(base,force=True)
        base.step_callback=viewer.update
    env=Monitor(base,str(report/'training'),info_keywords=('is_success','collision','time_s','goal_error_m'))
    if args.rl_mode and nav.guided and args.resume is None and not args.warm_start:
        args.resume=ROOT/'models/sac_continuous.zip'
        print(f'Initializing guided RL from trained SAC checkpoint {args.resume.name}',flush=True)
    replay_loaded=False
    if args.resume:
        model=SAC.load(args.resume,env=env,device='cpu')
        if getattr(model,'route_guidance',True) != nav.guided:
            parser.error('Resume checkpoint must use the same route-guidance mode.')
        replay=args.resume.with_suffix('.replay.pkl')
        objective_changed=args.rl_mode and nav.guided and args.teacher_anchor_weight>0
        if replay.exists() and not objective_changed:
            model.load_replay_buffer(replay)
            replay_loaded=True
    else:
        model=SAC('MlpPolicy',env,learning_rate=3e-4,buffer_size=200000,learning_starts=2000,
                  batch_size=128,tau=.005,gamma=.995,train_freq=4,gradient_steps=2,
                  ent_coef='auto_0.05',target_entropy=-2.,policy_kwargs=dict(net_arch=[64,64]),seed=args.seed,device='cpu',verbose=0)
    model.route_guidance=nav.guided
    if args.learning_rate is not None or (args.rl_mode and nav.guided):
        learning_rate=args.learning_rate or 1e-7
        model.learning_rate=learning_rate
        model.lr_schedule=lambda _:learning_rate
    if args.rl_mode and nav.guided:
        # Fine-tuning uses the same reward and replay distribution, but stale
        # Adam moments from pretraining can make the first actor updates abrupt.
        model.actor.optimizer.state.clear()
        model.critic.optimizer.state.clear()
    # Keep the checkpoint's learned entropy settings when fine-tuning the same
    # reward. If users opt into a teacher penalty, the old replay is not loaded
    # because its stored rewards were generated by a different objective.
    model.tensorboard_log=None
    initial_steps=model.num_timesteps
    (report/'run.json').write_text(json.dumps(dict(
        created_at_utc=datetime.now(timezone.utc).isoformat(),seed=args.seed,
        initial_steps=initial_steps,requested_new_steps=args.steps if args.episodes is None else None,
        control_dt_s=nav.config.dt,
        warm_start=args.warm_start,route_guidance=nav.guided,algorithm=algorithm,
        mission_only=args.mission_only,
        training_timeout_s=training_steps*nav.config.dt,
        evaluation_timeout_s=evaluation_steps*nav.config.dt,
        reward=reward_description,
        teacher_policy_sha256=teacher_sha,teacher_anchor_weight=base.teacher_anchor_weight,
        replay_buffer_loaded=replay_loaded,
        learning_rate=model.learning_rate,
        target_entropy=model.target_entropy,
        initial_entropy_coefficient=float(model.ent_coef_tensor.item()) if model.ent_coef_optimizer is None else float(model.log_ent_coef.exp().item()),
        requested_episodes=args.episodes,
        resumed_from=str(args.resume.resolve()) if args.resume else None,
        training_metrics='training_metrics.csv',
        note='Separate run; runtime policy changes only with --activate.'),indent=2)+'\n')
    began=time.perf_counter()
    demonstrations=None
    if args.warm_start:
        from continuous_warm_start import warm_start, demonstration_anchor
        print('Preparing continuous driving demonstrations and pretraining actor...',flush=True)
        demonstrations=warm_start(model,nav,args.seed,preview=viewer)
        if viewer:viewer.stage='Training SAC'
        # After a warm start collect current-policy experience instead of random
        # motor commands, and allow critics to adapt conservatively.
        model.learning_starts=0
        model.learning_rate=1e-4
        model.lr_schedule=lambda _:1e-4
        demonstration_anchor(model,weight=10.)
        demonstrations['sac_actor_bc_regularizer_weight']=10.
    evaluations=[]
    best_time=float('inf')
    best_records=None
    best_local=None
    best_actor_state=None
    best_actor_optimizer_state=None
    def assess(stage):
        nonlocal best_time,best_records,best_local,best_actor_state,best_actor_optimizer_state
        candidate=report/'candidate_actor.json'
        training=dict(seed=args.seed,total_steps=model.num_timesteps,seconds=time.perf_counter()-began,
                      algorithm='Stable Baselines3 SAC',demonstrations=demonstrations,
                      resumed_from=str(args.resume) if args.resume else None,
                      stage=stage,reward=reward_description)
        export_actor(model,nav,candidate,training,algorithm)
        actor,_=read_policy(candidate,nav,algorithm)
        records=evaluate(actor,nav,max_steps=evaluation_steps)
        successes=sum(r['is_success'] for r in records)
        mean_time=sum(r['time_s'] for r in records)/len(records)
        evaluations.append(dict(stage=stage,steps=model.num_timesteps,successes=successes,runs=len(records),mean_time_s=mean_time))
        print(f'Evaluation {stage}: {successes}/{len(records)} success; mean time={mean_time:.3f}s',flush=True)
        if not nav.guided and not args.mission_only:
            local=evaluate(actor,nav,seeds=range(2010,2020),max_steps=200,local=True)
            local_successes=sum(r['is_success'] for r in local)
            local_time=sum(r['time_s'] for r in local if r['is_success'])/max(1,local_successes)
            evaluations[-1].update(local_successes=local_successes,local_runs=len(local))
            print(f'Near goal: {local_successes}/{len(local)} success',flush=True)
            score=(local_successes,-local_time)
            if model.num_timesteps>initial_steps and local_successes and (best_local is None or score>best_local):
                best_local=score
                shutil.copyfile(candidate,model_dir/'rl_local_actor.json')
                (report/'local_evaluation.json').write_text(json.dumps(local,indent=2)+'\n')
        updated_rl=(not algorithm.startswith('rl') or model.num_timesteps>initial_steps)
        if updated_rl and successes==len(records) and mean_time<best_time:
            best_time,best_records=mean_time,records
            best_actor_state=copy.deepcopy(model.actor.state_dict())
            best_actor_optimizer_state=copy.deepcopy(model.actor.optimizer.state_dict())
            policy.write_text(candidate.read_text())
            model.save(model_dir/'sac_best')
            (report/'evaluation.json').write_text(json.dumps(records,indent=2)+'\n')
        elif stage=='resume_before_rl_updates' and successes==len(records):
            # The starting SAC actor is already a known-safe fallback. Keep it
            # as the best candidate so a failed fine-tuning checkpoint can be
            # rolled back without losing the working mission policy.
            best_time,best_records=mean_time,records
            best_actor_state=copy.deepcopy(model.actor.state_dict())
            best_actor_optimizer_state=copy.deepcopy(model.actor.optimizer.state_dict())
            policy.write_text(candidate.read_text())
            model.save(model_dir/'sac_best')
            (report/'evaluation.json').write_text(json.dumps(records,indent=2)+'\n')
        elif updated_rl and best_actor_state is not None:
            model.actor.load_state_dict(best_actor_state)
            model.actor.optimizer.load_state_dict(best_actor_optimizer_state)
            model.actor.optimizer.zero_grad(set_to_none=True)
            print('Checkpoint rejected; restored the best fully successful actor.',flush=True)
        (report/'checkpoints.json').write_text(json.dumps(evaluations,indent=2)+'\n')
        plot_progress()
    if args.warm_start: assess('after_demonstrations_before_sac')
    elif args.rl_mode: assess('resume_before_rl_updates')
    elif not nav.guided: assess('before_training')
    class Progress(BaseCallback):
        def __init__(self):
            super().__init__()
            self.episode_info=deque(maxlen=100)
            self.completed_episodes=0
            self.next_episode_evaluation=args.eval_every_episodes
            self.training_metrics_file=None
            self.training_metrics_writer=None
            self.last_training_metrics={}
            self.metrics_logger=None
            self.original_logger_record=None
            self.last_metrics_timestep=None

        def _on_training_start(self):
            self.training_metrics_file=(report/'training_metrics.csv').open('w',newline='')
            self.training_metrics_writer=csv.DictWriter(
                self.training_metrics_file,
                fieldnames=('timesteps','q_value_mean','actor_loss','critic_loss','entropy_coefficient'))
            self.training_metrics_writer.writeheader()

            # Capture SAC's real training losses directly from its logger.
            # The logger is configured by learn() before this callback runs.
            self.metrics_logger=self.model.logger
            self.original_logger_record=self.metrics_logger.record
            def capture_metric(key,value,exclude=None):
                result=self.original_logger_record(key,value,exclude)
                if key in ('train/actor_loss','train/critic_loss','train/ent_coef'):
                    try: scalar=float(value)
                    except (TypeError,ValueError): return result
                    self.last_training_metrics[key]=scalar
                    if viewer: viewer.record_training_metric(key,scalar)
                return result
            self.metrics_logger.record=capture_metric

        def _sample_q_value(self):
            replay=self.model.replay_buffer
            sample_size=min(128,replay.size()) if replay is not None else 0
            if sample_size<1:return None
            sample=replay.sample(sample_size,env=self.model._vec_normalize_env)
            with torch.no_grad():
                q_values=self.model.critic(sample.observations,sample.actions)
                return float(torch.cat(q_values,dim=1).mean().item())

        def _write_training_metrics(self):
            q_value=self._sample_q_value()
            if q_value is not None:
                self.last_training_metrics['train/q_value_mean']=q_value
                self.logger.record('train/q_value_mean',q_value)
                if viewer:viewer.record_training_metric('train/q_value_mean',q_value)
            self.training_metrics_writer.writerow(dict(
                timesteps=self.model.num_timesteps,
                q_value_mean=self.last_training_metrics.get('train/q_value_mean',''),
                actor_loss=self.last_training_metrics.get('train/actor_loss',''),
                critic_loss=self.last_training_metrics.get('train/critic_loss',''),
                entropy_coefficient=self.last_training_metrics.get('train/ent_coef','')))
            self.training_metrics_file.flush()
            self.last_metrics_timestep=self.model.num_timesteps

        def _on_training_end(self):
            if (self.training_metrics_writer is not None and
                    self.last_metrics_timestep!=self.model.num_timesteps):
                self._write_training_metrics()
            if self.metrics_logger is not None and self.original_logger_record is not None:
                self.metrics_logger.record=self.original_logger_record
            if self.training_metrics_file is not None:
                self.training_metrics_file.close()

        def _on_step(self):
            if viewer:
                if viewer.closed:return False
            # Broad random starts followed by a mixture including mission start.
            if not args.mission_only:
                base.curriculum=min(1.,.25+.75*self.n_calls/(args.steps*.6))
                base.mission_probability=.5 if self.n_calls>args.steps*.8 else 0.
            for done,info in zip(self.locals['dones'],self.locals['infos']):
                if done:
                    self.episode_info.append(info)
                    self.completed_episodes+=1
                    if viewer:
                        episode=info.get('episode',{})
                        viewer.record_episode(episode.get('r',base.episode_return),
                                              info.get('is_success',False),info.get('collision',False))
            if viewer:
                viewer.set_progress(self.model.num_timesteps,self.completed_episodes)
            self.logger.record('curriculum/distance_fraction',base.curriculum)
            self.logger.record('curriculum/mission_start_probability',base.mission_probability)
            if self.episode_info:
                self.logger.record('episodes/success_rate_100',sum(r['is_success'] for r in self.episode_info)/len(self.episode_info))
                self.logger.record('episodes/collision_rate_100',sum(r['collision'] for r in self.episode_info)/len(self.episode_info))
                self.logger.record('episodes/mean_duration_s_100',sum(r['time_s'] for r in self.episode_info)/len(self.episode_info))
            if self.n_calls%1000==0:
                self._write_training_metrics()
                self.logger.dump(self.num_timesteps)
            if self.n_calls%10000==0 and args.episodes is None:
                print(f'SAC new steps={self.n_calls}/{args.steps}; elapsed={time.perf_counter()-began:.1f}s',flush=True)
                self.model.save(model_dir/'sac_checkpoint')
            if args.episodes is None and self.n_calls%20000==0:
                assess('sac_checkpoint')
            elif args.episodes is not None and self.completed_episodes>=self.next_episode_evaluation:
                print(f'Completed episodes={self.completed_episodes}/{args.episodes}; steps={self.n_calls}',flush=True)
                assess('episode_checkpoint')
                self.next_episode_evaluation+=args.eval_every_episodes
            if args.episodes is not None and self.completed_episodes>=args.episodes:
                return False
            return True
    if args.episodes is not None:
        if args.rl_mode and nav.guided:
            # Conservative guided fine-tuning: 1 update per 4 new steps,
            # half the resumed SAC update rate.
            model.train_freq=TrainFreq(4,TrainFrequencyUnit.STEP)
            model.gradient_steps=1
        else:
            model.train_freq=TrainFreq(2,TrainFrequencyUnit.STEP)
            model.gradient_steps=1
        total_timesteps=args.episodes*training_steps
    else:
        total_timesteps=args.steps
    model.learn(total_timesteps=total_timesteps,callback=Progress(),reset_num_timesteps=False)
    assess('final_sac')
    model.save(model_dir/'sac_continuous')
    model.save_replay_buffer(model_dir/'sac_continuous.replay.pkl')
    model.logger.dump(model.num_timesteps)
    summary=dict(total_steps=model.num_timesteps,seconds=time.perf_counter()-began,seed=args.seed,
                 demonstrations=demonstrations,selected_mean_time_s=best_time if best_records else None,
                 best_local_successes=best_local[0] if best_local else None,
                 checkpoint_evaluations=evaluations,
                 runs=[{k:v for k,v in r.items() if k not in ('trajectory','wheel_commands')} for r in (best_records or [])],
                 scope='Best fully successful candidate on three fixed-map mission-start kinematic seeds; Webots validation is separate. No global time-optimality claim.')
    (report/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    env.close()
    if viewer:viewer.close()
    if algorithm.startswith('rl') and not best_records:
        print('Full maze not solved on all seeds; no policy activated for the full maze.',flush=True)
        if args.activate:
            raise SystemExit(1)
        return
    if not best_records:raise SystemExit('No fully successful continuous actor; candidate retained for diagnosis, runtime policy not replaced')
    if args.activate:
        shutil.copy2(policy,ROOT/'models'/actor_name)
        print('Activated best successful actor for Webots; validate physics separately.',flush=True)
    print(f'Best actor: {policy.resolve()}',flush=True)
    if args.plots:print(f'Graphs: {(report/"figures/training_progress.png").resolve()}',flush=True)


if __name__=='__main__':main()

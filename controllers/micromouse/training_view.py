"""Local OpenCV preview of the kinematic robot and live SAC metrics."""
import math
import time
from collections import deque

import numpy as np


class TrainingView:
    def __init__(self, navigation, speed=8.):
        import cv2
        self.cv2 = cv2
        self.nav = navigation
        self.speed = speed
        self.title = 'Micromouse - Training langsung + monitor SAC'
        self.size = 760
        self.pad = 28
        self.header = 110
        self.panel_width = 540
        self.canvas_width = self.size + self.panel_width + self.pad
        self.closed = False
        self.stage = 'Training SAC / RL'
        self.key = None
        self.trail = []
        self.last_tick = None
        self.last_frame = 0.
        self.completed_episodes = 0
        self.environment_steps = 0
        self.success_rate = None
        self.collision_rate = None
        self.latest_metrics = {}
        self.reward_history = deque(maxlen=120)
        self.steps_history = deque(maxlen=120)
        self.success_history = deque(maxlen=120)
        self.collision_history = deque(maxlen=120)
        self.q_history = deque(maxlen=120)
        self.actor_loss_history = deque(maxlen=120)
        self.critic_loss_history = deque(maxlen=120)
        self.entropy_history = deque(maxlen=120)
        cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.title, self.canvas_width, self.size + self.header)
        self.background = self.frame_background()

    def pixel(self, p):
        xmin, xmax, ymin, ymax = self.nav.bounds
        return (round(self.pad + (p[0] - xmin) / (xmax - xmin) * (self.size - 2 * self.pad)),
                round(self.header + self.pad + (ymax - p[1]) / (ymax - ymin) * (self.size - 2 * self.pad)))

    def frame_background(self):
        cv2 = self.cv2
        frame = np.full((self.size + self.header, self.canvas_width, 3), 246, dtype=np.uint8)
        for wall in self.nav.walls:
            p = np.asarray(wall.center)
            half = np.asarray(wall.size) / 2
            cv2.rectangle(frame, self.pixel(p + [-half[0], half[1]]), self.pixel(p + [half[0], -half[1]]),
                          (85, 95, 105), -1)
        cv2.circle(frame, self.pixel(self.nav.start), 7, (60, 150, 65), 2)
        if self.nav.hidden_goal:
            for x, y in self.nav.goal_cells:
                center = np.asarray(self.nav.maze.center(y*self.nav.maze.width+x))
                half = self.nav.cell_size/2
                cv2.rectangle(frame, self.pixel(center+[-half, half]),
                              self.pixel(center+[half, -half]), (25, 220, 255), -1)
        else:
            cv2.circle(frame, self.pixel(self.nav.goal), 9, (35, 145, 230), -1)
        cv2.line(frame, (self.size, 0), (self.size, self.size + self.header), (205, 210, 215), 1)
        return frame

    def record_episode(self, reward, success, collision, steps=0):
        self.completed_episodes += 1
        self.reward_history.append(float(reward))
        self.steps_history.append(int(steps))
        self.success_history.append(bool(success))
        self.collision_history.append(bool(collision))
        recent_successes = list(self.success_history)[-100:]
        recent_collisions = list(self.collision_history)[-100:]
        self.success_rate = 100. * sum(recent_successes) / len(recent_successes)
        self.collision_rate = 100. * sum(recent_collisions) / len(recent_collisions)

    def set_progress(self, environment_steps, completed_episodes):
        self.environment_steps = int(environment_steps)
        self.completed_episodes = int(completed_episodes)

    def record_training_metric(self, key, value):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return
        if not math.isfinite(value):
            return
        self.latest_metrics[key] = value
        history = {
            'train/q_value_mean': self.q_history,
            'train/actor_loss': self.actor_loss_history,
            'train/critic_loss': self.critic_loss_history,
            'train/ent_coef': self.entropy_history,
        }.get(key)
        if history is not None:
            history.append(value)

    def draw_chart(self, frame, box, title, series, fixed_range=None):
        cv2 = self.cv2
        x, y, width, height = box
        cv2.rectangle(frame, (x, y), (x + width, y + height), (220, 224, 228), 1)
        cv2.putText(frame, title, (x + 8, y + 21), cv2.FONT_HERSHEY_SIMPLEX, .42, (40, 45, 50), 1, cv2.LINE_AA)
        values = [float(v) for _, _, history in series for v in history if math.isfinite(float(v))]
        if not values:
            cv2.putText(frame, 'Menunggu data training...', (x + 12, y + height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, .38, (105, 110, 115), 1, cv2.LINE_AA)
            return
        if fixed_range is None:
            low, high = min(values), max(values)
            margin = max((high - low) * .12, abs(low) * .04, abs(high) * .04, .05)
            low, high = low - margin, high + margin
        else:
            low, high = fixed_range
        left, right = x + 43, x + width - 9
        top, bottom = y + 52, y + height - 27
        cv2.rectangle(frame, (left, top), (right, bottom), (228, 231, 234), 1)
        for fraction, label in ((0., f'{high:.2g}'), (.5, ''), (1., f'{low:.2g}')):
            py = round(top + fraction * (bottom - top))
            cv2.line(frame, (left, py), (right, py), (232, 234, 237), 1)
            if label:
                cv2.putText(frame, label, (x + 4, py + 4), cv2.FONT_HERSHEY_SIMPLEX, .31,
                            (100, 105, 110), 1, cv2.LINE_AA)
        for index, (label, color, history) in enumerate(series):
            values = list(history)
            if not values:
                continue
            legend_x = x + 8 + index * 112
            cv2.line(frame, (legend_x, y + 39), (legend_x + 12, y + 39), color, 2)
            cv2.putText(frame, label, (legend_x + 16, y + 43), cv2.FONT_HERSHEY_SIMPLEX, .31,
                        (70, 75, 80), 1, cv2.LINE_AA)
            points = []
            for i, value in enumerate(values):
                px = left + (right - left) * i / max(1, len(values) - 1)
                py = bottom - (float(value) - low) / max(1e-9, high - low) * (bottom - top)
                points.append((round(px), round(py)))
            if len(points) > 1:
                cv2.polylines(frame, [np.asarray(points, dtype=np.int32).reshape(-1, 1, 2)], False, color, 2,
                              cv2.LINE_AA)
            else:
                cv2.circle(frame, points[0], 3, color, -1, cv2.LINE_AA)
        cv2.putText(frame, 'awal', (left, y + height - 7), cv2.FONT_HERSHEY_SIMPLEX, .3,
                    (110, 115, 120), 1, cv2.LINE_AA)
        cv2.putText(frame, 'terbaru', (right - 42, y + height - 7), cv2.FONT_HERSHEY_SIMPLEX, .3,
                    (110, 115, 120), 1, cv2.LINE_AA)

    def draw_monitor(self, frame, env):
        cv2 = self.cv2
        x = self.size + 15
        cv2.putText(frame, 'MONITOR SAC / RL', (x, 24), cv2.FONT_HERSHEY_SIMPLEX, .58,
                    (35, 45, 55), 2, cv2.LINE_AA)
        rate = lambda value: 'n/a' if value is None else f'{value:.1f}%'
        metric = lambda key: 'n/a' if key not in self.latest_metrics else f'{self.latest_metrics[key]:.3g}'
        lines = [
            f"Episode selesai {self.completed_episodes} | Langkah {self.environment_steps}",
            f"Sukses {rate(self.success_rate)} | Tabrakan {rate(self.collision_rate)} (100 ep terakhir)",
            f"Q mean {metric('train/q_value_mean')} | Alpha {metric('train/ent_coef')}",
            f"Actor loss {metric('train/actor_loss')} | Critic loss {metric('train/critic_loss')}",
        ]
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (x, 43 + i * 18), cv2.FONT_HERSHEY_SIMPLEX, .34,
                        (65, 70, 75), 1, cv2.LINE_AA)

        if self.nav.hidden_goal:
            ir = ' '.join(f'{label}:{value*.4:.2f}m'
                          for label, value in zip(self.nav.ir_sensor_labels, env.last_ir))
            action_v = (float(env.last_action[0])+1)*.5*env.config.max_linear
            action_w = float(env.last_action[1])*env.config.max_angular
            state_lines = [
                f"State x/y/heading: {env.pose[0]:.2f} m, {env.pose[1]:.2f} m, {env.pose[2]:.2f} rad",
                f"IR {ir} | hijau=kosong sampai 0.40m, kuning dekat, merah <0.10m",
                f"Wheel rad/s: [{env.wheels[0]:.2f},{env.wheels[1]:.2f}] | aksi v={action_v:.2f} m/s, yaw={action_w:.2f} rad/s",
                f"Sel baru +{env.novelty_reward:.2f} | ulang -{env.revisit_cell_penalty:.2f} | spin -{env.anti_spin_penalty:.2f}",
                f"Kunjungan sel {env.cell_visit_fraction:.2f} | diam {env.stagnation_fraction*.5:.1f}s | kuning={int(env.last_yellow)}",
            ]
            for i, line in enumerate(state_lines):
                cv2.putText(frame, line, (x, 128+i*17), cv2.FONT_HERSHEY_SIMPLEX, .32,
                            (40, 80, 120), 1, cv2.LINE_AA)

        gap = 12
        chart_width = (self.panel_width - 3 * gap) // 2
        chart_height = 185
        chart_x1 = x
        chart_x2 = x + chart_width + gap
        chart_y1 = self.header + 145
        chart_y2 = chart_y1 + chart_height + gap
        chart_y3 = chart_y2 + chart_height + gap
        self.draw_chart(frame, (chart_x1, chart_y1, chart_width, chart_height),
                        'Reward episode', [('reward', (190, 105, 30), self.reward_history)])
        self.draw_chart(frame, (chart_x2, chart_y1, chart_width, chart_height),
                        'Langkah per episode', [('steps', (65, 125, 185), self.steps_history)])
        self.draw_chart(frame, (chart_x1, chart_y2, chart_width, chart_height),
                        'Hasil episode (%)', [
                            ('goal', (65, 155, 75), self._rolling_success()),
                            ('tabrakan', (50, 80, 220), self._rolling_collision()),
                        ], fixed_range=(0., 100.))
        self.draw_chart(frame, (chart_x2, chart_y2, chart_width, chart_height),
                        'Estimasi Q critic', [('Q mean', (190, 105, 30), self.q_history)])
        self.draw_chart(frame, (chart_x1, chart_y3, chart_width, chart_height),
                        'Loss actor dan critic', [
                            ('actor', (65, 155, 75), self.actor_loss_history),
                            ('critic', (50, 80, 220), self.critic_loss_history),
                        ])
        self.draw_chart(frame, (chart_x2, chart_y3, chart_width, chart_height),
                        'Entropy SAC / eksplorasi', [('alpha', (121, 89, 163), self.entropy_history)])
        cv2.putText(frame, 'Q adalah estimasi critic SAC, bukan Q-table.', (x, self.size + self.header - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, .36, (90, 95, 100), 1, cv2.LINE_AA)

    def _rolling_success(self):
        return self._rolling(self.success_history)

    def _rolling_collision(self):
        return self._rolling(self.collision_history)

    @staticmethod
    def _rolling(history):
        values = list(history)
        result = []
        for end in range(len(values)):
            window = values[max(0, end - 99):end + 1]
            result.append(100. * sum(window) / len(window))
        return result

    def frame(self, env):
        cv2 = self.cv2
        frame = self.background.copy()
        label = (id(env), env.episode_index)
        if label != self.key:
            self.trail = []
            self.key = label
        point = self.pixel(env.pose[:2])
        self.trail.append(point)
        if len(self.trail) > 1:
            cv2.polylines(frame, [np.asarray(self.trail, dtype=np.int32)], False, (200, 150, 55), 2)
        if self.nav.hidden_goal:
            origins, directions, distances = self.nav.ir_ray_segments(env.pose)
            for origin, direction, distance in zip(origins, directions, distances):
                end = origin + direction * distance
                start_px, end_px = self.pixel(origin), self.pixel(end)
                color = (55, 175, 70) if distance >= .23 else (35, 180, 235) if distance >= .10 else (55, 65, 220)
                cv2.line(frame, start_px, end_px, color, 2, cv2.LINE_AA)
                cv2.circle(frame, start_px, 3, color, -1, cv2.LINE_AA)
                cv2.circle(frame, end_px, 3, color, -1, cv2.LINE_AA)
        radius = max(4, round(.075 / (self.nav.bounds[1] - self.nav.bounds[0]) * (self.size - 2 * self.pad)))
        cv2.circle(frame, point, radius, (45, 75, 220), -1)
        nose = self.pixel(env.pose[:2] + .13 * np.array([math.cos(env.pose[2]), math.sin(env.pose[2])]))
        cv2.line(frame, point, nose, (35, 35, 70), 2)
        if env.last_success:
            status = 'GOAL'
        elif env.last_collision:
            status = 'TABRAKAN'
        else:
            status = 'TIMEOUT' if env.finished else 'BERJALAN'
        phase = 'eksplorasi' if self.nav.hidden_goal and 'Eksploitasi' not in self.stage else 'eksploitasi' if self.nav.hidden_goal else ''
        lines = [f'{self.stage} | Episode {env.episode_index} | {status}',
                 f'Step {env.steps} | Reward akumulasi {env.episode_return:.1f} | Simulasi {env.steps * env.config.dt:.1f}/{env.max_steps * env.config.dt:.1f} dtk',
                 f'Fase {phase or "continuous"} | speed {self.speed:g}x | Q / Esc: akhiri training']
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (18, 26 + i * 28), cv2.FONT_HERSHEY_SIMPLEX, .48,
                        (40, 45, 50), 1, cv2.LINE_AA)
        self.draw_monitor(frame, env)
        return frame

    def update(self, env, force=False, dt=None):
        if self.closed:
            return
        now = time.monotonic()
        if self.last_tick is not None and not force:
            interval = env.config.dt if dt is None else dt
            time.sleep(max(0., interval / self.speed - (now - self.last_tick)))
        self.last_tick = time.monotonic()
        if not force and self.last_tick - self.last_frame < 1 / 32:
            return
        self.last_frame = self.last_tick
        self.cv2.imshow(self.title, self.frame(env))
        key = self.cv2.waitKey(1) & 255
        if key in (ord('q'), 27) or self.cv2.getWindowProperty(self.title, self.cv2.WND_PROP_VISIBLE) < 1:
            self.close()

    def close(self):
        if not self.closed:
            self.cv2.destroyWindow(self.title)
            self.closed = True

"""Live OpenCV monitor for sensor-based tabular Q-learning."""
from collections import Counter


class QLearningMonitor:
    def __init__(self, env, render_speed=8, show=True):
        import cv2
        import numpy as np
        self.cv2, self.np = cv2, np
        self.env, self.show = env, show
        self.render_every = max(1, int(render_speed))
        self.scale = 38
        self.map_size = max(env.width, env.height) * self.scale
        self.panel_w, self.pad, self.header = 640, 22, 58
        self.height = self.map_size + self.header + self.pad * 2 + 90
        self.width = self.map_size + self.panel_w + self.pad * 3
        self.title = 'Tabular Q-Learning | Eksplorasi dan Eksploitasi'
        self.action_counts = Counter()
        self.current_image = None
        self.total_episodes = 1
        if show:
            cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.title, min(1500, self.width), min(1000, self.height))

    def _cell_pixel(self, x, y):
        left = self.pad
        top = self.header + self.pad
        return (left + x * self.scale, top + (self.env.height - 1 - y) * self.scale)

    def update(self, env, episode, episodes, phase, epsilon, step, reward,
               q_table, visits, history, chosen_action=None):
        self.total_episodes = episodes
        if not self.show or step % self.render_every:
            return True
        cv2, np = self.cv2, self.np
        image = np.full((self.height, self.width, 3), (239, 242, 238), np.uint8)
        cv2.rectangle(image, (0, 0), (self.width, self.header), (39, 55, 61), -1)
        cv2.putText(image, 'SENSOR-BASED TABULAR Q-LEARNING', (20, 37),
                    cv2.FONT_HERSHEY_SIMPLEX, .75, (255, 255, 255), 2, cv2.LINE_AA)
        visits_by_cell = Counter()
        for key, count in visits.items():
            cell = int(key.split(',', 1)[0])
            visits_by_cell[cell] += count
        max_visits = max(visits_by_cell.values(), default=1)
        for y in range(env.height):
            for x in range(env.width):
                left, top = self._cell_pixel(x, y)
                cell_id = y * env.width + x
                if (x, y) in env.yellow_cells:
                    color = (45, 218, 255)
                elif (x, y) == env.start:
                    color = (170, 224, 194)
                else:
                    shade = int(238 - 75 * visits_by_cell[cell_id] / max_visits)
                    color = (shade, shade + 2, min(255, shade + 7))
                cv2.rectangle(image, (left, top), (left+self.scale-1, top+self.scale-1), color, -1)
                cv2.rectangle(image, (left, top), (left+self.scale-1, top+self.scale-1),
                              (201, 210, 204), 1)
        for wall in env.walls:
            x, y = wall.center
            sx, sy = wall.size
            a = self._world_pixel(x-sx/2, y+sy/2)
            b = self._world_pixel(x+sx/2, y-sy/2)
            cv2.rectangle(image, a, b, (70, 83, 87), -1)
        if len(env.path) > 1:
            points = np.array([self._world_pixel(*p) for p in env.path], dtype=np.int32)
            cv2.polylines(image, [points], False, (213, 132, 40), 2, cv2.LINE_AA)
        robot = self._world_pixel(*env.position)
        cv2.circle(image, robot, 7, (40, 72, 230), -1)
        nose = self._world_pixel(env.position[0] + .11*__import__('math').cos(env.heading*__import__('math').pi/2),
                                 env.position[1] + .11*__import__('math').sin(env.heading*__import__('math').pi/2))
        cv2.line(image, robot, nose, (20, 40, 130), 2, cv2.LINE_AA)

        panel_x = self.map_size + self.pad*2
        text = lambda value, y, scale=.52, color=(47, 58, 60), weight=1: cv2.putText(
            image, str(value), (panel_x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, weight, cv2.LINE_AA)
        color_phase = (20, 115, 75) if phase == 'eksplorasi' else (160, 90, 20)
        text(f'Fase: {phase.upper()}     Episode: {episode}/{episodes}', self.header+36, .62, color_phase, 2)
        text(f'Epsilon: {epsilon:.3f}     Step: {step}     Reward: {reward:.1f}', self.header+69)
        state = env.state()
        state_id = ','.join(map(str, state))
        values = q_table.get(state_id, [0.] * 4)
        labels = ('EAST', 'NORTH', 'WEST', 'SOUTH')
        text('STATE SPACE YANG SEDANG DIAMATI', self.header+112, .56, (30, 84, 133), 2)
        text(f'cell={state[0]} | heading={labels[state[1]]}', self.header+140)
        text(f'IR [front,left,back,right] = {state[2:6]}', self.header+166)
        text(f'Sensor warna kuning = {bool(state[6])}', self.header+192)
        text(f'Q(s,a): E {values[0]:7.2f} | N {values[1]:7.2f}', self.header+225)
        text(f'       W {values[2]:7.2f} | S {values[3]:7.2f}', self.header+251)
        text(f'Visited states: {len(q_table):,} / {env.max_state_count:,}', self.header+284)
        if chosen_action is not None:
            text(f'Tindakan terakhir: {labels[chosen_action]}', self.header+310, .48,
                 (153, 78, 19), 2)
        text('REWARD / STEP / SUCCESS', self.header+330, .56, (30, 84, 133), 2)
        self._draw_plot(image, panel_x, self.header+350, self.panel_w-40, 145, history,
                        'reward', 'reward')
        self._draw_plot(image, panel_x, self.header+525, self.panel_w-40, 112, history,
                        'steps', 'steps')
        recent = history[-100:]
        success_rate = sum(r['success'] for r in recent)/len(recent) if recent else 0.
        collision_rate = sum(r['collisions'] for r in recent)/max(1, sum(r['steps'] for r in recent))
        text(f'100 episode terakhir: success {success_rate:.0%} | collision/step {collision_rate:.1%}',
             self.header+665, .44)
        text('Legend: abu = frekuensi kunjungan | kuning = tile goal | merah = robot',
             self.height-18, .42)
        self.current_image = image
        cv2.imshow(self.title, image)
        key = cv2.waitKey(1) & 0xff
        return key not in (27, ord('q'))

    def _world_pixel(self, x, y):
        ix = (x - self.env.bounds[0]) / self.env.cell_size
        iy = (self.env.bounds[3] - y) / self.env.cell_size
        return (round(self.pad + ix*self.scale),
                round(self.header + self.pad + iy*self.scale))

    def _draw_plot(self, image, x, y, width, height, history, key, label):
        cv2 = self.cv2
        cv2.rectangle(image, (x, y), (x+width, y+height), (252, 253, 250), -1)
        cv2.rectangle(image, (x, y), (x+width, y+height), (205, 214, 207), 1)
        cv2.putText(image, label, (x+8, y+18), cv2.FONT_HERSHEY_SIMPLEX, .43,
                    (65, 76, 78), 1, cv2.LINE_AA)
        if not history:
            return
        values = [float(row[key]) for row in history[-400:]]
        smooth = [sum(values[max(0, i-19):i+1])/len(values[max(0, i-19):i+1])
                  for i in range(len(values))]
        lo, hi = min(smooth), max(smooth)
        if hi-lo < 1e-9:
            hi = lo+1.
        pts = [(x+5+round(i*(width-10)/max(1, len(smooth)-1)),
                y+height-5-round((v-lo)/(hi-lo)*(height-30))) for i, v in enumerate(smooth)]
        if len(pts) > 1:
            import numpy as np
            cv2.polylines(image, [np.array(pts, np.int32)], False, (186, 105, 28), 2, cv2.LINE_AA)
        cv2.putText(image, f'{lo:.1f} .. {hi:.1f}   (mean 20)', (x+width-165, y+18),
                    cv2.FONT_HERSHEY_SIMPLEX, .36, (78, 84, 82), 1, cv2.LINE_AA)

    def episode_finished(self, row, history, env, q_table, visits):
        if not self.show:
            return
        cv2 = self.cv2
        speed = self.render_every
        self.render_every = 1
        self.update(env, row['episode'], self.total_episodes, row['phase'],
                    row['epsilon'], row['steps'], row['reward'], q_table, visits, history)
        self.render_every = speed
        if row['success']:
            print(f"Episode {row['episode']}: goal detected by yellow sensor in {row['steps']} steps; reward={row['reward']:.1f}", flush=True)
        elif row['episode'] % 100 == 0:
            print(f"Episode {row['episode']}: reward={row['reward']:.1f}, steps={row['steps']}, collisions={row['collisions']}, states={row['unique_states']}", flush=True)
        cv2.waitKey(1)

    def save(self, path):
        if self.current_image is not None:
            self.cv2.imwrite(str(path), self.current_image)

    def close(self):
        if self.show:
            self.cv2.destroyWindow(self.title)

# 화학 2 과제

## 코드 전문
```
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
import math

box_size = 10
radius = 0.2
dt = 0.1
temperature = 300

molecule_defs = {
    'red-black': {
        'offsets': np.array([[-radius*0.8, 0], [radius*0.8, 0]]),
        'colors': ['red', 'black']
    },
    'red-red': {
        'offsets': np.array([[-radius*0.8, 0], [radius*0.8, 0]]),
        'colors': ['red', 'red']
    },
    'red-black-red': {
        'offsets': np.array([[-radius*1.2, 0], [0, 0], [radius*1.2, 0]]),
        'colors': ['red', 'black', 'red']
    },
    'red': {
        'offsets': np.array([[0, 0]]),
        'colors': ['red']
    }
}

n_type_rb = 5
n_type_rr = 5
n_type_rbr = 0
n_type_r = 0

positions = np.empty((0, 2))
velocities = np.empty((0, 2))
molecule_types = []

def init_particles(n, mol_type):
    offsets = molecule_defs[mol_type]['offsets']
    margin = np.max(np.abs(offsets)) + radius
    pos = np.random.rand(n, 2) * (box_size - 2 * margin) + margin
    vel = np.zeros((n, 2))
    for i in range(n):
        theta = np.random.uniform(0, 2 * np.pi)
        speed = math.sqrt(temperature * 2 / 40)
        vel[i] = [speed * np.cos(theta), speed * np.sin(theta)]
    types = [mol_type] * n
    return pos, vel, types

def update_particles(n_rb, n_rr, n_rbr, n_r):
    global positions, velocities, molecule_types, n_particles
    pos_rb, vel_rb, types_rb = init_particles(n_rb, 'red-black')
    pos_rr, vel_rr, types_rr = init_particles(n_rr, 'red-red')
    pos_rbr, vel_rbr, types_rbr = init_particles(n_rbr, 'red-black-red')
    pos_r, vel_r, types_r = init_particles(n_r, 'red')

    positions = np.vstack([pos_rb, pos_rr, pos_rbr, pos_r])
    velocities = np.vstack([vel_rb, vel_rr, vel_rbr, vel_r])
    molecule_types = types_rb + types_rr + types_rbr + types_r
    n_particles = len(positions)

create_circles_called = False
circles = []

fig, ax = plt.subplots(figsize=(8, 6))
fig.subplots_adjust(left=0.45, right=0.95, bottom=0.1, top=0.9)
ax.set_xlim(0, box_size)
ax.set_ylim(0, box_size)
ax.set_aspect('equal')

def create_circles():
    global circles
    for cset in circles:
        for c in cset:
            c.remove()
    circles.clear()

    for i in range(n_particles):
        mol_type = molecule_types[i]
        center = positions[i]
        offsets = molecule_defs[mol_type]['offsets']
        colors = molecule_defs[mol_type]['colors']

        c_list = []
        for off, col in zip(offsets, colors):
            c = Circle(center + off, radius, color=col)
            ax.add_patch(c)
            c_list.append(c)
        circles.append(c_list)

def handle_collision(i, j):
    global positions, velocities, molecule_types, n_particles

    pos_i, pos_j = positions[i], positions[j]
    vel_i, vel_j = velocities[i], velocities[j]

    offsets_i = molecule_defs[molecule_types[i]]['offsets']
    offsets_j = molecule_defs[molecule_types[j]]['offsets']

    collided = False
    collided_atom_i = None
    collided_atom_j = None

    # 충돌 원자 찾기
    for idx_i, off_i in enumerate(offsets_i):
        for idx_j, off_j in enumerate(offsets_j):
            dist = np.linalg.norm((pos_i + off_i) - (pos_j + off_j))
            if dist < 2 * radius:
                collided = True
                collided_atom_i = idx_i
                collided_atom_j = idx_j
                break
        if collided:
            break

    if not collided:
        return False  # 충돌 안됨

    type_i = molecule_types[i]
    type_j = molecule_types[j]

    # --- 변환 조건 및 처리 ---

    # 1) rb(black atom, idx=1) + rr 충돌 → rbr + r (기존)
    condition_rb_rr = ((type_i == 'red-black' and collided_atom_i == 1 and type_j == 'red-red') or
                       (type_j == 'red-black' and collided_atom_j == 1 and type_i == 'red-red'))

    # 2) rb(black atom) + r 충돌 → rbr (신규)
    condition_rb_r = ((type_i == 'red-black' and collided_atom_i == 1 and type_j == 'red') or
                      (type_j == 'red-black' and collided_atom_j == 1 and type_i == 'red'))

    # 3) r + r 충돌 → rr (신규)
    condition_r_r = (type_i == 'red' and type_j == 'red')

    if temperature >= 200:
        if condition_rb_rr:
            if type_i == 'red-black':
                idx_rb, idx_rr = i, j
                pos_rb, pos_rr = pos_i, pos_j
                vel_rb, vel_rr = vel_i, vel_j
            else:
                idx_rb, idx_rr = j, i
                pos_rb, pos_rr = pos_j, pos_i
                vel_rb, vel_rr = vel_j, vel_i

            mask = np.ones(n_particles, dtype=bool)
            mask[[idx_rb, idx_rr]] = False

            positions_new = positions[mask]
            velocities_new = velocities[mask]
            molecule_types_new = [molecule_types[k] for k in range(n_particles) if mask[k]]

            new_pos_rbr = (pos_rb + pos_rr) / 2
            new_pos_r = new_pos_rbr + np.array([radius * 2, 0])

            new_vel_rbr = (vel_rb + vel_rr) / 2
            new_vel_r = new_vel_rbr.copy()

            positions_new = np.vstack([positions_new, new_pos_rbr, new_pos_r])
            velocities_new = np.vstack([velocities_new, new_vel_rbr, new_vel_r])
            molecule_types_new += ['red-black-red', 'red']

            positions = positions_new
            velocities = velocities_new
            molecule_types = molecule_types_new
            n_particles = len(positions)

            create_circles()
            print("Collision transform: red-black(black) + red-red -> red-black-red + red")
            return True

        elif condition_rb_r:
            if type_i == 'red-black':
                idx_rb, idx_r = i, j
                pos_rb, pos_r = pos_i, pos_j
                vel_rb, vel_r = vel_i, vel_j
            else:
                idx_rb, idx_r = j, i
                pos_rb, pos_r = pos_j, pos_i
                vel_rb, vel_r = vel_j, vel_i

            mask = np.ones(n_particles, dtype=bool)
            mask[[idx_rb, idx_r]] = False

            positions_new = positions[mask]
            velocities_new = velocities[mask]
            molecule_types_new = [molecule_types[k] for k in range(n_particles) if mask[k]]

            new_pos_rbr = (pos_rb + pos_r) / 2
            new_vel_rbr = (vel_rb + vel_r) / 2
    
            positions_new = np.vstack([positions_new, new_pos_rbr])
            velocities_new = np.vstack([velocities_new, new_vel_rbr])
            molecule_types_new += ['red-black-red']

            positions = positions_new
            velocities = velocities_new
            molecule_types = molecule_types_new
            n_particles = len(positions)

            create_circles()
            print("Collision transform: red-black(black) + red -> red-black-red")
            return True

        elif condition_r_r:
            mask = np.ones(n_particles, dtype=bool)
            mask[[i, j]] = False

            positions_new = positions[mask]
            velocities_new = velocities[mask]
            molecule_types_new = [molecule_types[k] for k in range(n_particles) if mask[k]]

            new_pos_rr = (pos_i + pos_j) / 2
            new_vel_rr = (vel_i + vel_j) / 2

            positions_new = np.vstack([positions_new, new_pos_rr])
            velocities_new = np.vstack([velocities_new, new_vel_rr])
            molecule_types_new += ['red-red']

            positions = positions_new
            velocities = velocities_new
            molecule_types = molecule_types_new
            n_particles = len(positions)

            create_circles()
            print("Collision transform: red + red -> red-red")
            return True

    # --- 탄성 충돌 처리 (기존) ---
    delta_pos = pos_i - pos_j
    dist = np.linalg.norm(delta_pos)
    if dist == 0:
        return False

    n_vec = delta_pos / dist
    delta_vel = vel_i - vel_j
    p = np.dot(delta_vel, n_vec)

    if p >= 0:
        return False

    velocities[i] = vel_i - p * n_vec
    velocities[j] = vel_j + p * n_vec

    overlap = 2 * radius * 2 - dist  # *2 because molecules double radius apart
    if overlap > 0:
        correction = overlap / 2 * n_vec
        positions[i] += correction
        positions[j] -= correction

    def rescale_velocity(idx):
        speed = np.linalg.norm(velocities[idx])
        if speed == 0:
            return
        direction = velocities[idx] / speed
        velocities[idx] = direction * (math.sqrt(temperature*2 / 40))

    rescale_velocity(i)
    rescale_velocity(j)

    return False



def update(frame):
    global positions, velocities, n_particles

    positions += velocities * dt

    for i in range(n_particles):
        mol_type = molecule_types[i]
        offs = molecule_defs[mol_type]['offsets']
        max_offset = np.max(np.abs(offs)) + radius
        for dim in [0, 1]:
            low_limit = max_offset
            high_limit = box_size - max_offset
            if positions[i, dim] <= low_limit or positions[i, dim] >= high_limit:
                velocities[i, dim] *= -1
                positions[i, dim] = np.clip(positions[i, dim], low_limit, high_limit)

    # 충돌 처리 시, 인덱스 변경 가능하므로 while 루프와 재검사 로직 사용
    i = 0
    while i < n_particles:
        j = i + 1
        while j < n_particles:
            transformed = handle_collision(i, j)
            if transformed:
                # 변환 발생 시 인덱스 초기화해서 재검사
                i = -1
                break
            j += 1
        i += 1

    for i in range(n_particles):
        mol_type = molecule_types[i]
        center = positions[i]
        offsets = molecule_defs[mol_type]['offsets']
        for c, off in zip(circles[i], offsets):
            c.center = center + off

    return [c for cset in circles for c in cset]

def on_slider_rb(val):
    global n_type_rb
    n_type_rb = int(val)
    update_particles(n_type_rb, n_type_rr, n_type_rbr, n_type_r)
    create_circles()

def on_slider_rr(val):
    global n_type_rr
    n_type_rr = int(val)
    update_particles(n_type_rb, n_type_rr, n_type_rbr, n_type_r)
    create_circles()

def on_slider_rbr(val):
    global n_type_rbr
    n_type_rbr = int(val)
    update_particles(n_type_rb, n_type_rr, n_type_rbr, n_type_r)
    create_circles()

def on_slider_r(val):
    global n_type_r
    n_type_r = int(val)
    update_particles(n_type_rb, n_type_rr, n_type_rbr, n_type_r)
    create_circles()

def on_temp_change(val):
    global temperature, velocities
    temperature = val
    speeds = np.linalg.norm(velocities, axis=1)
    speeds_safe = np.where(speeds == 0, 1, speeds)
    directions = velocities / speeds_safe[:, np.newaxis]
    velocities = directions * (math.sqrt(temperature * 2 / 40))

update_particles(n_type_rb, n_type_rr, n_type_rbr, n_type_r)
create_circles()

ani = FuncAnimation(fig, update, frames=1000, interval=50, blit=True)

ax_temp = plt.axes([0.05, 0.80, 0.2, 0.03])
slider_temp = Slider(ax_temp, 'Temperature', 50, 600, valinit=temperature, valstep=10)
slider_temp.on_changed(on_temp_change)

ax_rb = plt.axes([0.05, 0.72, 0.2, 0.03])
slider_rb = Slider(ax_rb, 'Red+Black', 0, 20, valinit=n_type_rb, valstep=1)
slider_rb.on_changed(on_slider_rb)

ax_rr = plt.axes([0.05, 0.64, 0.2, 0.03])
slider_rr = Slider(ax_rr, 'Red+Red', 0, 20, valinit=n_type_rr, valstep=1)
slider_rr.on_changed(on_slider_rr)

ax_rbr = plt.axes([0.05, 0.56, 0.2, 0.03])
slider_rbr = Slider(ax_rbr, 'Red+Black+Red', 0, 10, valinit=n_type_rbr, valstep=1)
slider_rbr.on_changed(on_slider_rbr)

ax_r = plt.axes([0.05, 0.48, 0.2, 0.03])
slider_r = Slider(ax_r, 'Red single', 0, 20, valinit=n_type_r, valstep=1)
slider_r.on_changed(on_slider_r)

plt.show()
```
-----

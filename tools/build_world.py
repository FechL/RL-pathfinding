"""Generate a deterministic micromouse maze world and floor texture."""
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
LAYOUT = json.loads((ROOT / "worlds/maze_layout.json").read_text())


def edge(a, b):
    return tuple(sorted((tuple(a), tuple(b))))


def generate_walls():
    """Generate a connected DFS maze, then add loops and open the 2x2 goal."""
    width, height = LAYOUT["grid_size"]
    rng = random.Random(LAYOUT["seed"])
    start = tuple(LAYOUT["start_cell"])
    visited, stack, openings = {start}, [start], set()
    while stack:
        x, y = stack[-1]
        neighbors = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
        neighbors = [(nx, ny) for nx, ny in neighbors
                     if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in visited]
        if not neighbors:
            stack.pop()
            continue
        rng.shuffle(neighbors)
        next_cell = neighbors[0]
        openings.add(edge((x, y), next_cell))
        visited.add(next_cell)
        stack.append(next_cell)

    goals = {tuple(cell) for cell in LAYOUT["goal_cells"]}
    for x, y in goals:
        for neighbor in ((x+1, y), (x-1, y), (x, y+1), (x, y-1)):
            if neighbor in goals:
                openings.add(edge((x, y), neighbor))

    possible_loops = []
    for y in range(height):
        for x in range(width):
            for neighbor in ((x+1, y), (x, y+1)):
                if neighbor[0] < width and neighbor[1] < height:
                    connection = edge((x, y), neighbor)
                    if connection not in openings and not ({(x, y), neighbor} <= goals):
                        possible_loops.append(connection)
    rng.shuffle(possible_loops)
    openings.update(possible_loops[:LAYOUT["loop_openings"]])

    cell = LAYOUT["cell_size"]
    thickness = LAYOUT["wall_thickness"]
    wall_height = LAYOUT["wall_height"]
    x_min, y_min = -width*cell/2, -height*cell/2
    walls = []
    for y in range(height):
        for x in range(width):
            if x+1 < width and edge((x, y), (x+1, y)) not in openings:
                walls.append({
                    "name": f"V{x}_{y}",
                    "position": [x_min+(x+1)*cell, y_min+(y+0.5)*cell],
                    "size": [thickness, cell+thickness, wall_height],
                })
            if y+1 < height and edge((x, y), (x, y+1)) not in openings:
                walls.append({
                    "name": f"H{x}_{y}",
                    "position": [x_min+(x+0.5)*cell, y_min+(y+1)*cell],
                    "size": [cell+thickness, thickness, wall_height],
                })
    return walls


def cell_center(index):
    width, height = LAYOUT["grid_size"]
    cell = LAYOUT["cell_size"]
    return (-width*cell/2+(index[0]+0.5)*cell,
            -height*cell/2+(index[1]+0.5)*cell)


def mission_points():
    start = cell_center(LAYOUT["start_cell"])
    goal_cells = [tuple(cell) for cell in LAYOUT["goal_cells"]]
    goal = (sum(cell_center(cell)[0] for cell in goal_cells)/len(goal_cells),
            sum(cell_center(cell)[1] for cell in goal_cells)/len(goal_cells))
    return start, goal


def floor_texture():
    width, height = LAYOUT["grid_size"]
    cell = LAYOUT["cell_size"]
    world_w, world_h = width*cell, height*cell
    scale = 320
    image = Image.new("RGB", (round(world_w*scale), round(world_h*scale)), "#edf0e9")
    draw = ImageDraw.Draw(image)

    def xy(x, y):
        return round((x+world_w/2)*scale), round((world_h/2-y)*scale)

    def cell_box(index):
        x, y = index
        top_left = xy(-world_w/2+x*cell, -world_h/2+(y+1)*cell)
        bottom_right = xy(-world_w/2+(x+1)*cell, -world_h/2+y*cell)
        return (*top_left, *bottom_right)

    draw.rectangle(cell_box(LAYOUT["start_cell"]), fill="#bfe0d2")
    for goal_cell in LAYOUT["goal_cells"]:
        draw.rectangle(cell_box(goal_cell), fill="#f1d28e")

    for i in range(width+1):
        x = i*cell*scale
        draw.line((x, 0, x, world_h*scale), fill="#c8d1cc", width=2)
    for j in range(height+1):
        y = j*cell*scale
        draw.line((0, y, world_w*scale, y), fill="#c8d1cc", width=2)

    regular = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    bold = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    label_font = ImageFont.truetype(bold, 22)
    small_font = ImageFont.truetype(regular, 15)
    sx, sy = cell_center(LAYOUT["start_cell"])
    gx, gy = mission_points()[1]
    draw.text(xy(sx, sy), "START", fill="#245f52", font=label_font, anchor="mm")
    draw.text(xy(gx, gy), "GOAL", fill="#80571c", font=label_font, anchor="mm")
    ImageDraw.Draw(image).rectangle((2, 2, image.width-3, image.height-3), outline="#748681", width=5)

    folder = ROOT / "worlds/textures"
    folder.mkdir(exist_ok=True)
    image.resize((2048, 2048), Image.Resampling.LANCZOS).save(folder / "micromouse_floor.png")


def build_world():
    width, height = LAYOUT["grid_size"]
    cell = LAYOUT["cell_size"]
    world_w, world_h = width*cell, height*cell
    start, goal = mission_points()
    walls = generate_walls()
    parts = [f'''#VRML_SIM R2025a utf8
# Generated by tools/build_world.py from maze_layout.json.
EXTERNPROTO "../protos/NavigationArena.proto"
EXTERNPROTO "../protos/MazeWall.proto"
EXTERNPROTO "../protos/Micromouse.proto"

WorldInfo {{
  basicTimeStep 16
  title "{LAYOUT['title']}"
  info [ "{width} x {height} cell maze with dead ends, loops, and a 2 x 2 goal at the far end." "Micromouse uses an A* route baseline; four infrared sensors and differential-drive motors are available for future RL work." ]
}}
DEF OVERVIEW Viewpoint {{
  orientation -0.29 0.54 0.79 1.27
  position -3.7 -5.7 8.5
  fieldOfView 0.82
  description "Micromouse maze overview"
}}
Background {{ skyColor [ 0.88 0.92 0.95 ] }}
DirectionalLight {{ direction 0.3 0.4 -1 intensity 0.85 ambientIntensity 0.65 castShadows TRUE }}
DirectionalLight {{ direction -0.5 -0.2 -1 intensity 0.35 ambientIntensity 0.2 }}
DEF ARENA NavigationArena {{ floorSize {world_w} {world_h} cellSize {cell} }}
DEF MICROMOUSE Micromouse {{
  translation {start[0]} {start[1]} 0
  rotation 0 0 1 0
  name "Micromouse M1"
  controller "micromouse"
  supervisor TRUE
}}
DEF WALLS Group {{
  children [
''']
    for wall in walls:
        x, y = wall["position"]
        sx, sy, sz = wall["size"]
        parts.append(f'''    MazeWall {{
      translation {x:.4f} {y:.4f} {sz/2:.3f}
      size {sx:.4f} {sy:.4f} {sz:.3f}
      name "{wall['name']}"
    }}
''')
    goal_size = cell*2
    parts.append(f'''  ]
}}
DEF START Pose {{ translation {start[0]:.4f} {start[1]:.4f} 0 }}
DEF GOAL Pose {{ translation {goal[0]:.4f} {goal[1]:.4f} 0 }}
DEF GOAL_MARKER Pose {{
  translation {goal[0]:.4f} {goal[1]:.4f} 0.001
  children [ Shape {{
    appearance PBRAppearance {{ baseColor 0.92 0.66 0.23 transparency 0.25 roughness 1 }}
    geometry Box {{ size {goal_size:.3f} {goal_size:.3f} 0.001 }}
  }} ]
}}
''')
    (ROOT / "worlds/world.wbt").write_text("".join(parts))


if __name__ == "__main__":
    floor_texture()
    build_world()
    print(f"Built {len(generate_walls())} maze walls for {LAYOUT['title']}")

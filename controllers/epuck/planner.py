"""Finite A* search with continuous collision checks on every path segment."""
from dataclasses import dataclass
import heapq
import math

ROBOT_RADIUS = 0.037
SAFETY_MARGIN = 0.03
GRID_RESOLUTION = 0.05


@dataclass(frozen=True)
class Obstacle:
    name: str
    kind: str
    center: tuple
    size: tuple

    @property
    def circular(self):
        return self.kind == "drum"


def distance(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])


def point_clearance(point, obstacle):
    """Signed distance to the physical obstacle footprint."""
    dx = abs(point[0]-obstacle.center[0])
    dy = abs(point[1]-obstacle.center[1])
    if obstacle.circular:
        return math.hypot(dx, dy)-obstacle.size[0]/2
    ex, ey = dx-obstacle.size[0]/2, dy-obstacle.size[1]/2
    return math.hypot(max(ex, 0), max(ey, 0))+min(max(ex, ey), 0)


def physical_clearance(point, obstacles, bounds, robot_radius=ROBOT_RADIUS):
    xmin, xmax, ymin, ymax = bounds
    values = [point[0]-xmin, xmax-point[0], point[1]-ymin, ymax-point[1]]
    values.extend(point_clearance(point, ob) for ob in obstacles)
    return min(values)-robot_radius


def _segment_hits_box(a, b, center, half_size):
    lower, upper = 0.0, 1.0
    for axis in (0, 1):
        origin, delta, half = a[axis]-center[axis], b[axis]-a[axis], half_size[axis]
        if abs(delta) < 1e-12:
            if abs(origin) > half:
                return False
        else:
            first, last = sorted(((-half-origin)/delta, (half-origin)/delta))
            lower, upper = max(lower, first), min(upper, last)
            if lower > upper:
                return False
    return True


def segment_is_free(a, b, obstacles, bounds, radius=ROBOT_RADIUS+SAFETY_MARGIN):
    xmin, xmax, ymin, ymax = bounds
    for p in (a, b):
        if not (xmin+radius <= p[0] <= xmax-radius and ymin+radius <= p[1] <= ymax-radius):
            return False
    dx, dy = b[0]-a[0], b[1]-a[1]
    length_sq = dx*dx+dy*dy
    for ob in obstacles:
        if ob.circular:
            t = 0 if length_sq == 0 else max(0, min(1, ((ob.center[0]-a[0])*dx+(ob.center[1]-a[1])*dy)/length_sq))
            if distance((a[0]+t*dx, a[1]+t*dy), ob.center) <= ob.size[0]/2+radius:
                return False
        elif _segment_hits_box(a, b, ob.center, (ob.size[0]/2+radius, ob.size[1]/2+radius)):
            return False
    return True


def smooth_path(path, obstacles, bounds, radius):
    result, current = [path[0]], 0
    while current < len(path)-1:
        end = len(path)-1
        while not segment_is_free(path[current], path[end], obstacles, bounds, radius):
            end -= 1
        result.append(path[end])
        current = end
    return result


def plan_path(start, goal, obstacles, bounds, resolution=GRID_RESOLUTION,
              robot_radius=ROBOT_RADIUS, safety_margin=SAFETY_MARGIN):
    """Return exact endpoint waypoints and explored grid positions.

    Physical positions are continuous. Search samples an 8-connected integer
    lattice; boxes are conservatively expanded at their corners. This is A*,
    not an RL policy or a mathematically continuous optimal planner.
    """
    if resolution <= 0 or robot_radius <= 0 or safety_margin < 0:
        raise ValueError("Resolution/radius must be positive; margin must be nonnegative")
    start, goal, radius = tuple(start), tuple(goal), robot_radius+safety_margin
    if not segment_is_free(start, start, obstacles, bounds, radius) or not segment_is_free(goal, goal, obstacles, bounds, radius):
        return None, set()
    if segment_is_free(start, goal, obstacles, bounds, radius):
        return ([start] if start == goal else [start, goal]), set()
    xmin, xmax, ymin, ymax = bounds
    limits = (math.ceil((xmin+radius)/resolution), math.floor((xmax-radius)/resolution),
              math.ceil((ymin+radius)/resolution), math.floor((ymax-radius)/resolution))

    def position(cell):
        return (cell[0]*resolution, cell[1]*resolution)

    def inside(cell):
        return limits[0] <= cell[0] <= limits[1] and limits[2] <= cell[1] <= limits[3]

    queue, costs, parents, closed, explored = [], {}, {}, set(), set()
    seed = (round(start[0]/resolution), round(start[1]/resolution))
    for ix in range(seed[0]-1, seed[0]+2):
        for iy in range(seed[1]-1, seed[1]+2):
            cell, point = (ix, iy), position((ix, iy))
            if inside(cell) and segment_is_free(start, point, obstacles, bounds, radius):
                costs[cell], parents[cell] = distance(start, point), None
                heapq.heappush(queue, (costs[cell]+distance(point, goal), cell))
    while queue:
        _, current = heapq.heappop(queue)
        if current in closed:
            continue
        closed.add(current)
        point = position(current)
        explored.add(point)
        if distance(point, goal) <= resolution*1.5 and segment_is_free(point, goal, obstacles, bounds, radius):
            path = [goal]
            while current is not None:
                path.append(position(current))
                current = parents[current]
            path.append(start)
            path.reverse()
            path = [p for i, p in enumerate(path) if i == 0 or distance(p, path[i-1]) > 1e-9]
            return smooth_path(path, obstacles, bounds, radius), explored
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                cell = (current[0]+dx, current[1]+dy)
                if (dx == dy == 0) or cell in closed or not inside(cell):
                    continue
                neighbor = position(cell)
                if not segment_is_free(point, neighbor, obstacles, bounds, radius):
                    continue
                cost = costs[current]+distance(point, neighbor)
                if cost < costs.get(cell, math.inf):
                    costs[cell], parents[cell] = cost, current
                    heapq.heappush(queue, (cost+distance(neighbor, goal), cell))
    return None, explored


def path_length(path):
    return sum(distance(a, b) for a, b in zip(path, path[1:]))

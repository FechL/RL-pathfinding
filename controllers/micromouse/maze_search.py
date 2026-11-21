"""Depth-first search over open maze-cell connections."""
import math

from planner import segment_is_free


def depth_first_path(start,goal,walls,bounds,cell_size):
    """Try cell branches in order and backtrack until the goal cell is found."""
    xmin,xmax,ymin,ymax=map(float,bounds)
    cell_size=float(cell_size)
    width=round((xmax-xmin)/cell_size)
    height=round((ymax-ymin)/cell_size)

    def cell(point):
        x=min(width-1,max(0,int(math.floor((point[0]-xmin)/cell_size))))
        y=min(height-1,max(0,int(math.floor((point[1]-ymin)/cell_size))))
        return y*width+x

    def center(index):
        return (xmin+(index%width+.5)*cell_size,
                ymin+(index//width+.5)*cell_size)

    start_cell,goal_cell=cell(start),cell(goal)
    graph=[[] for _ in range(width*height)]
    directions=((1,0),(0,1),(-1,0),(0,-1))
    for index in range(width*height):
        x,y=index%width,index//width
        for dx,dy in directions:
            nx,ny=x+dx,y+dy
            if not (0<=nx<width and 0<=ny<height):
                continue
            neighbor=ny*width+nx
            if segment_is_free(center(index),center(neighbor),walls,bounds):
                graph[index].append(neighbor)

    parent={start_cell:None}
    visited={start_cell}
    stack=[(start_cell,iter(graph[start_cell]))]
    while stack and goal_cell not in visited:
        current,neighbors=stack[-1]
        next_cell=next((candidate for candidate in neighbors if candidate not in visited),None)
        if next_cell is None:
            stack.pop()
            continue
        parent[next_cell]=current
        visited.add(next_cell)
        stack.append((next_cell,iter(graph[next_cell])))

    if goal_cell not in visited:
        raise RuntimeError('DFS could not find a path from start to goal')
    cells=[];current=goal_cell
    while current is not None:
        cells.append(current);current=parent[current]
    cells.reverse()
    path=[tuple(start)]+[center(index) for index in cells[1:-1]]+[tuple(goal)]
    # Remove coincident endpoints when start and goal occupy the same cell.
    compact=[]
    for point in path:
        if not compact or math.dist(compact[-1],point)>.01:
            compact.append(point)
    return compact,len(visited)

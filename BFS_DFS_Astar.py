import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque
from queue import PriorityQueue
import time
import sys
from memory_profiler import memory_usage
from statistics import mean




def generate_maze(size=15):
    if size % 2 == 0:
        size += 1  # Ensure size is odd for maze generation

    # Create a grid where walls are True and cells are False
    maze = np.full((size, size), True)
    
    # Start with a random cell
    start_x, start_y = (random.randrange(1, size, 2), random.randrange(1, size, 2))
    maze[start_x, start_y] = False

    # List of walls to process, start with walls of the first cell
    walls = []
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        if 0 <= start_x+dx < size and 0 <= start_y+dy < size:
            walls.append((start_x+dx, start_y+dy))

    while walls:
        # Choose a random wall
        wall = random.choice(walls)
        walls.remove(wall)

        # Find the cells that this wall divides
        x, y = wall
        if x % 2 == 0:  # Vertical wall
            cell1 = (x-1, y)
            cell2 = (x+1, y)
        else:  # Horizontal wall
            cell1 = (x, y-1)
            cell2 = (x, y+1)

        # If the cells that the wall divides are in different states, remove the wall
        if 0 <= cell2[0] < size and 0 <= cell2[1] < size and maze[cell1] != maze[cell2]:
            maze[wall] = False
            new_open_cell = cell2 if maze[cell1] == False else cell1
            maze[new_open_cell] = False

            # Add the neighboring walls of the new cell to the wall list
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = new_open_cell[0]+dx, new_open_cell[1]+dy
                if 0 <= nx < size and 0 <= ny < size and maze[nx, ny] == True:
                    if (nx, ny) not in walls:
                        walls.append((nx, ny))

    # Create entrance and exit
    maze[1][0] = False  # Entrance
    maze[size-2][size-1] = False  # Exit

    return maze
def bfs_solve_maze_with_bool(maze, start, end, vis=False):
    start_time = time.time()  # Start timing
    
    queue = deque([start])
    visited = set()
    path = {}
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    nodes_explored = 0  # Initialize node counter
    
    if vis:
        # Set up the plot
        plt.figure(figsize=(8, 8))
        plt.imshow(maze, cmap='binary')
        plt.xticks([]), plt.yticks([])
        plt.title("Breadth First Search")  # Title for the plot


        # Annotate start and end
        start_annotation_x = start[1] - 0.5 if start[1] > 0 else start[1] + 0.5
        start_annotation_y = start[0] - 3  # Adjusting this line to correctly define start_annotation_y
        plt.annotate('Start', xy=(start[1], start[0]), xytext=(start_annotation_x, start_annotation_y),
                    arrowprops=dict(facecolor='green', shrink=0.05), fontsize=12, ha='center')

        # Annotate end outside the maze with an arrow pointing towards it
        end_annotation_x = end[1] + 0.5 if end[1] < maze.shape[1] - 1 else end[1] - 0.5
        end_annotation_y = end[0] + 2  # Adjusting for a more consistent position relative to the maze boundary
        plt.annotate('End', xy=(end[1], end[0]), xytext=(end_annotation_x, end_annotation_y),
                    arrowprops=dict(facecolor='red', shrink=0.05), fontsize=12, ha='center')
    while queue:
        current = queue.popleft()
        nodes_explored += 1  # Increment nodes explored when a node is processed

        if vis:
            plt.plot(current[1], current[0], 'ro', markersize=8)
            plt.pause(0.0001)  # Pause to display the update

        if current == end:
            # Reconstruct the path from end to start
            solution_path = []
            while current != start:
                solution_path.append(current)
                current = path[current]
            solution_path.append(start)
            solution_path.reverse()

            end_time = time.time()  # End timing
            if vis:
                # Visualize the final path
                for step in solution_path:
                    plt.plot(step[1], step[0], 'bo', markersize=4)
                    plt.pause(0.0001)
                plt.show()

            return solution_path, end_time - start_time, nodes_explored  # Return path, time taken, and nodes explored

        if current not in visited:
            visited.add(current)
            for dx, dy in directions:
                next_cell = (current[0] + dx, current[1] + dy)
                if 0 <= next_cell[0] < maze.shape[0] and 0 <= next_cell[1] < maze.shape[1] and maze[next_cell] == False:
                    if next_cell not in visited:
                        queue.append(next_cell)
                        path[next_cell] = current

    end_time = time.time()  # End timing
    if vis:
        plt.show()
    return None, end_time - start_time, nodes_explored


def dfs_solve_maze_with_bool(maze, start, end, vis=False):
    start_time = time.time()
    stack = [start]
    visited = set()
    path = {}
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    nodes_explored = 0
    solution_path = []

    if vis:
        # Set up the plot if visualization is enabled
        plt.figure(figsize=(8, 8))
        plt.imshow(maze, cmap='binary')
        plt.xticks([]), plt.yticks([])
        plt.title("Depth First Search")
        plt.annotate('Start', xy=(start[1], start[0]), xytext=(start[1] - 0.5, start[0] - 3),
                     arrowprops=dict(facecolor='green', shrink=0.05), ha='center')
        plt.annotate('End', xy=(end[1], end[0]), xytext=(end[1] + 0.5, end[0] + 3),
                     arrowprops=dict(facecolor='red', shrink=0.05), ha='center')

    while stack:
        current = stack.pop()
        if vis:
            plt.plot(current[1], current[0], 'ro', markersize=8)
            plt.pause(0.0001)

        if current == end:
            while current in path:
                solution_path.append(current)
                current = path[current]
            solution_path.append(start)
            solution_path.reverse()
            end_time = time.time()

            if vis:
                for step in solution_path:
                    plt.plot(step[1], step[0], 'bo', markersize=4)
                    plt.pause(0.0001)
                plt.show()

            return solution_path, end_time - start_time, nodes_explored

        if current not in visited:
            visited.add(current)
            nodes_explored += 1
            for dx, dy in directions:
                next_cell = (current[0] + dx, current[1] + dy)
                if 0 <= next_cell[0] < maze.shape[0] and 0 <= next_cell[1] < maze.shape[1] and maze[next_cell] == False:
                    if next_cell not in visited:
                        stack.append(next_cell)
                        path[next_cell] = current

    end_time = time.time()
    if vis:
        plt.show()
    return None, end_time - start_time, nodes_explored

def heuristic(a, b):
    return abs(b[0] - a[0]) + abs(b[1] - a[1])


def a_star_solve_maze_with_bool(maze, start, end, vis=False):
    start_time = time.time()
    open_set = PriorityQueue()
    open_set.put((0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, end)}
    nodes_explored = 0
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]


    if vis:
        plt.figure(figsize=(8, 8))
        plt.imshow(maze, cmap='binary')
        plt.xticks([]), plt.yticks([])
        plt.title("A Star Search")
        plt.annotate('Start', xy=(start[1], start[0]), xytext=(start[1] - 0.5, start[0] - 3),
                     arrowprops=dict(facecolor='green', shrink=0.05), ha='center')
        plt.annotate('End', xy=(end[1], end[0]), xytext=(end[1] + 0.5, end[0] + 3),
                     arrowprops=dict(facecolor='red', shrink=0.05), ha='center')

    while not open_set.empty():
        current = open_set.get()[1]
        nodes_explored += 1

        if vis:
            plt.plot(current[1], current[0], 'ro', markersize=8)
            plt.pause(0.0001)

        if current == end:
            solution_path = []
            while current in came_from:
                solution_path.append(current)
                current = came_from[current]
            solution_path.append(start)
            solution_path.reverse()
            end_time = time.time()

            if vis:
                for step in solution_path:
                    plt.plot(step[1], step[0], 'bo', markersize=4)
                    plt.pause(0.0001)
                plt.show()

            return solution_path, end_time - start_time, nodes_explored

        for dx, dy in directions:
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < maze.shape[0] and 0 <= neighbor[1] < maze.shape[1] and maze[neighbor] == 0:
                tentative_g_score = g_score[current] + 1
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, end)
                    open_set.put((f_score[neighbor], neighbor))


    end_time = time.time()
    if vis:
        plt.show()
    return None, end_time - start_time, nodes_explored # Return None and time taken if no path found


# Code for performance comparision of different algorithms on same maze
def main(arg1=35):
    maze_sizes=[25,45,65,85]

    for maze_size in maze_sizes:
        bfstimes=list()
        bfs_nodes=list()
        bfs_mem=list()
        dfs_times=list()
        dfs_nodes=list()
        dfs_mem=list()
        astartimes=list()
        astar_nodes=list()
        astar_mem=list()
        for i in  range(3):
            maze = generate_maze(maze_size)  # Adjust size as needed
            start_point = (1, 0)
            end_point = (maze.shape[0] - 2, maze.shape[1] - 1)
            def wrap_bfs():
                solution_path, bfs_time, bfs_explore = bfs_solve_maze_with_bool(maze, start_point, end_point)
                bfstimes.append(bfs_time)
                bfs_nodes.append(bfs_explore)
            def wrap_dfs():
                dfs_solution_path, dfs_time, dfs_explore = dfs_solve_maze_with_bool(maze, start_point, end_point)
                dfs_times.append(dfs_time)
                dfs_nodes.append(dfs_explore)
            def wrap_astar():
                astar, atime, astar_explore = a_star_solve_maze_with_bool(maze, start_point, end_point)
                astartimes.append(atime)
                astar_nodes.append(astar_explore)
            bfs_mem.append(mean(memory_usage(proc=wrap_bfs)))
            dfs_mem.append(mean(memory_usage(proc=wrap_dfs)))
            astar_mem.append(mean(memory_usage(proc=wrap_astar)))
        print("time taken by BFS for size "+str(maze_size)+" : "+str(sum(bfstimes) / len(bfstimes))+" seconds & Avg nodes explored are "+str(int(sum(bfs_nodes) / len(bfs_nodes))))
        print(f"Memory usage of BFS: {str(sum(bfs_mem) / len(bfs_mem))} MiB")
        print("time taken by DFS for size "+str(maze_size)+" : "+str(sum(dfs_times) / len(dfs_times))+" seconds & Avg nodes explored are "+str(int(sum(dfs_nodes) / len(dfs_nodes))))
        print(f"Memory usage of DFS: {str(sum(dfs_mem) / len(dfs_mem))} MiB")
        print("time taken by Astar for size "+str(maze_size)+" : "+str(sum(astartimes) / len(astartimes))+" seconds & Avg nodes explored are "+str(int(sum(astar_nodes) / len(astar_nodes))))
        print(f"Memory usage of AStar: {str(sum(astar_mem) / len(astar_mem))} MiB")
    time.sleep(2)


    #code for visualisation of different algorithms on same maze
    maze_size = int(arg1)
    maze = generate_maze(maze_size)

    start_point = (1, 0)
    end_point = (maze_size - 1, maze_size - 2)

    solution_path = bfs_solve_maze_with_bool(maze, start_point, end_point,True)
    dfs_solution_path = dfs_solve_maze_with_bool(maze, start_point, end_point, True)
    astar = a_star_solve_maze_with_bool(maze, start_point, end_point,True)

if __name__ == '__main__':
    # This code won't run if this file is imported.
    args = sys.argv[1:]

    main(*args)

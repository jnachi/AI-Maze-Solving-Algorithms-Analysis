import numpy as np
import random
import matplotlib.pyplot as plt
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

def get_next_state_value(state, action, maze):
    directions = {'up': (-1, 0), 'right': (0, 1), 'down': (1, 0), 'left': (0, -1)}
    next_state = (state[0] + directions[action][0], state[1] + directions[action][1])
    if 0 <= next_state[0] < maze.shape[0] and 0 <= next_state[1] < maze.shape[1] and maze[next_state] == 0:
        return next_state
    return state  # Return the original state if the action is not feasible

def extract_policy_value(value_map, maze, goal_state, discount_factor=0.9):
    policy = {}
    for state in np.ndindex(maze.shape):
        if maze[state] == 1 or state == goal_state:
            continue
        max_value = float('-inf')
        best_action = None
        for action in ['up', 'right', 'down', 'left']:
            next_state = get_next_state_value(state, action, maze)
            value = value_map[next_state]
            if value > max_value:
                max_value = value
                best_action = action
        policy[state] = best_action
    return policy

def value_iteration_with_bool(maze, goal_state, vis=False, discount_factor=0.9, threshold=0.0000001, plot_every_n_iterations=15):
    value_map = np.zeros(maze.shape)
    iteration = 0
    while True:
        delta = 0
        for state in np.ndindex(maze.shape):
            if maze[state] == 1 or state == goal_state:
                continue
            v = value_map[state]
            max_value = float('-inf')
            for action in ['up', 'right', 'down', 'left']:
                next_state = get_next_state_value(state, action, maze)
                value = value_map[next_state]
                max_value = max(max_value, value)
            new_v = -1 + discount_factor * max_value  # -1 accounts for step cost
            value_map[state] = new_v
            delta = max(delta, abs(v - new_v))
        
        # Optionally plot the policy at this iteration
        if vis==True:
            if iteration % plot_every_n_iterations == 0:
                temp_policy = extract_policy_value(value_map, maze, goal_state, discount_factor)
                plot_policy_on_maze_value(maze, temp_policy, iteration)
        
        iteration += 1
        if delta < threshold:
            break
    
    return value_map, iteration

def plot_policy_on_maze_value(maze, policy, iteration=None, start=(1, 0), end=None):
    if end is None:
        end = (maze.shape[0]-2, maze.shape[1]-1)  # Default end point if not provided
    
    plt.figure(figsize=(8, 8))
    plt.imshow(maze, cmap='Greys', interpolation='none', alpha=0.8)
    for state, action in policy.items():
        if maze[state] == 0:  # Only plot for open cells
            draw_action_value(state, action)

    # Annotate start
    start_annotation_x = start[1] - 0.5 if start[1] > 0 else start[1] + 0.5
    start_annotation_y = start[0] - 3
    plt.annotate('Start', xy=(start[1], start[0]), xytext=(start_annotation_x, start_annotation_y),
                 arrowprops=dict(facecolor='green', shrink=0.05), fontsize=12, ha='center')

    # Annotate end
    end_annotation_x = end[1] + 0.5 if end[1] < maze.shape[1] - 1 else end[1] - 0.5
    end_annotation_y = end[0] + 2
    plt.annotate('End', xy=(end[1], end[0]), xytext=(end_annotation_x, end_annotation_y),
                 arrowprops=dict(facecolor='red', shrink=0.05), fontsize=12, ha='center')

    title = 'Optimal Value Visualised on Maze'
    if iteration is not None:
        title += f' at Iteration {iteration}'
    plt.title(title)
    plt.show()

def draw_action_value(state, action):
    y, x = state
    if action == 'up':
        plt.arrow(x, y, 0, -0.5, head_width=0.1, head_length=0.2, fc='blue', ec='blue')
    elif action == 'down':
        plt.arrow(x, y, 0, 0.5, head_width=0.1, head_length=0.2, fc='blue', ec='blue')
    elif action == 'left':
        plt.arrow(x, y, -0.5, 0, head_width=0.1, head_length=0.2, fc='blue', ec='blue')
    elif action == 'right':
        plt.arrow(x, y, 0.5, 0, head_width=0.1, head_length=0.2, fc='blue', ec='blue')

def find_path_value(policy, start, goal, maze):
    path = [start]
    current = start
    while current != goal:
        action = policy.get(current)
        if action == 'up':
            next_state = (current[0]-1, current[1])
        elif action == 'down':
            next_state = (current[0]+1, current[1])
        elif action == 'left':
            next_state = (current[0], current[1]-1)
        elif action == 'right':
            next_state = (current[0], current[1]+1)
        if next_state == current:  # Prevent infinite loops
            break  # No valid move available from current position
        path.append(next_state)
        current = next_state
    return path

def visualize_path_value(maze, path):
    plt.figure(figsize=(8, 8))
    plt.imshow(maze, cmap='Greys', interpolation='none', alpha=0.8)
    ys, xs = zip(*path)
    plt.plot(xs, ys, color='red', linewidth=2, markersize=10, marker='o', label='Path')
    start = path[0]  # The first coordinate in your path list
    end = path[-1]  # The last coordinate in your path list

    # Calculate and adjust the x, y coordinates for the start and end annotations for better appearance
    start_annotation_x = start[1] - 0.5 if start[1] > 0 else start[1] + 0.5
    start_annotation_y = start[0] - 3  # Adjusting this line to correctly define start_annotation_y
    plt.annotate('Start', xy=(start[1], start[0]), xytext=(start_annotation_x, start_annotation_y),
                 arrowprops=dict(facecolor='green', shrink=0.05), fontsize=12, ha='center')

    end_annotation_x = end[1] + 0.5 if end[1] < maze.shape[1] - 1 else end[1] - 0.5
    end_annotation_y = end[0] + 2  # Adjusting for a more consistent position relative to the maze boundary
    plt.annotate('End', xy=(end[1], end[0]), xytext=(end_annotation_x, end_annotation_y),
                 arrowprops=dict(facecolor='red', shrink=0.05), fontsize=12, ha='center')

    plt.title('Optimal Path from Start to Goal')
    plt.legend()
    plt.show()

#########################################
def plot_policy_on_maze_policy(maze, policy, iteration, start=(1, 0), end=None):
    if end is None:
        end = (maze.shape[0]-2, maze.shape[1]-1)  # Default end point if not provided
    
    plt.figure(figsize=(8, 8))
    plt.imshow(maze, cmap='Greys', interpolation='none', alpha=0.8)
    for state, action in policy.items():
        if maze[state] == 0:  # Only plot for open cells
            draw_action_policy(state, action)
    
    # Annotate start
    start_annotation_x = start[1] - 0.5 if start[1] > 0 else start[1] + 0.5
    start_annotation_y = start[0] - 3
    plt.annotate('Start', xy=(start[1], start[0]), xytext=(start_annotation_x, start_annotation_y),
                 arrowprops=dict(facecolor='green', shrink=0.05), fontsize=12, ha='center')

    # Annotate end
    end_annotation_x = end[1] + 0.5 if end[1] < maze.shape[1] - 1 else end[1] - 0.5
    end_annotation_y = end[0] + 2
    plt.annotate('End', xy=(end[1], end[0]), xytext=(end_annotation_x, end_annotation_y),
                 arrowprops=dict(facecolor='red', shrink=0.05), fontsize=12, ha='center')

    plt.title(f'Policy Iteration: Iteration {iteration}')
    plt.show()
    plt.close()   # Close the plot automatically after the pause

def draw_action_policy(state, action):
    y, x = state
    if action == 'up':
        plt.arrow(x, y, 0, -0.5, head_width=0.1, head_length=0.2, fc='blue', ec='blue')
    elif action == 'down':
        plt.arrow(x, y, 0, 0.5, head_width=0.1, head_length=0.2, fc='blue', ec='blue')
    elif action == 'left':
        plt.arrow(x, y, -0.5, 0, head_width=0.1, head_length=0.2, fc='blue', ec='blue')
    elif action == 'right':
        plt.arrow(x, y, 0.5, 0, head_width=0.1, head_length=0.2, fc='blue', ec='blue')

def find_path_policy(policy, start, goal, maze):
    path = [start]
    current = start
    while current != goal:
        action = policy.get(current)
        if action == 'up':
            next_state = (current[0]-1, current[1])
        elif action == 'down':
            next_state = (current[0]+1, current[1])
        elif action == 'left':
            next_state = (current[0], current[1]-1)
        elif action == 'right':
            next_state = (current[0], current[1]+1)
        path.append(next_state)
        current = next_state
    return path

def visualize_path_policy(maze, path, start=(1, 0), end=None):
    if end is None:
        end = (maze.shape[0]-2, maze.shape[1]-1)  # Assuming the default end point if not provided
    
    plt.figure(figsize=(8, 8))
    plt.imshow(maze, cmap='Greys', interpolation='none', alpha=0.8)
    ys, xs = zip(*path)
    plt.plot(xs, ys, color='red', linewidth=2, markersize=10, marker='o')

    # Annotate start
    start_annotation_x = start[1] - 0.5 if start[1] > 0 else start[1] + 0.5
    start_annotation_y = start[0] - 3
    plt.annotate('Start', xy=(start[1], start[0]), xytext=(start_annotation_x, start_annotation_y),
                 arrowprops=dict(facecolor='green', shrink=0.05), fontsize=12, ha='center')

    # Annotate end
    end_annotation_x = end[1] + 0.5 if end[1] < maze.shape[1] - 1 else end[1] - 0.5
    end_annotation_y = end[0] + 2
    plt.annotate('End', xy=(end[1], end[0]), xytext=(end_annotation_x, end_annotation_y),
                 arrowprops=dict(facecolor='red', shrink=0.05), fontsize=12, ha='center')

    plt.title('Path from Start to Goal')
    plt.show()

def policy_evaluation(policy, utility, maze, discount_factor, goal_state):
    threshold = 0.001
    while True:
        delta = 0
        for state, action in policy.items():
            if maze[state] == 1 or state == goal_state:  # Skip walls and the goal state
                continue
            u = utility[state]
            reward = -1  # Constant step cost for non-goal states
            new_u = reward + discount_factor * utility[get_next_state_policy(state, action, maze)]
            utility[state] = new_u
            delta = max(delta, abs(u - new_u))
        if delta < threshold:
            break
    # Set the utility of the goal state to the highest positive reward
    utility[goal_state] = 0
    return utility

def policy_improvement(utility, maze, discount_factor, goal_state):
    new_policy = {}
    for state in np.ndindex(maze.shape):
        if maze[state] == 1 or state == goal_state:  # Skip walls and the goal state
            continue
        best_action = None
        best_value = float('-inf')
        for action in ['up', 'right', 'down', 'left']:
            next_state = get_next_state_policy(state, action, maze)
            value = utility[next_state]
            if value > best_value:
                best_value = value
                best_action = action
        new_policy[state] = best_action
    # Set the policy at the goal state to None (no action needed)
    new_policy[goal_state] = None
    return new_policy

def get_next_state_policy(state, action, maze):
    directions = {'up': (-1, 0), 'right': (0, 1), 'down': (1, 0), 'left': (0, -1)}
    next_state = (state[0] + directions[action][0], state[1] + directions[action][1])
    if 0 <= next_state[0] < maze.shape[0] and 0 <= next_state[1] < maze.shape[1] and maze[next_state] == 0:
        return next_state
    return state  # Return the original state if the action is not feasible

def policy_iteration_with_bool(maze, goal_state, vis=False, discount_factor=0.9):
    policy = {state: random.choice(['up', 'right', 'down', 'left'])
              for state in np.ndindex(maze.shape) if maze[state] == 0}
    utility = np.zeros(maze.shape)
    
    iteration = 0
    while True:
        utility = policy_evaluation(policy, utility, maze, discount_factor, goal_state)
        new_policy = policy_improvement(utility, maze, discount_factor, goal_state)
    
        plt.close()  # Close the plot automatically
        
        # Check if the policy has changed significantly
        if all(new_policy[state] == policy.get(state, None) for state in policy):
            break  # If not, then we have reached convergence
        policy = new_policy
        iteration += 1
        if vis==True:
            if(iteration%15==0):
                plot_policy_on_maze_policy(maze, policy, iteration)
    if vis==True:
        plot_policy_on_maze_policy(maze, policy, iteration)
    return policy, iteration

# Code for performance comparision of different algorithms on same maze

def main(arg1=35):
    maze_sizes=[25,45,65]
    print("Performance of Value iteration and Policy iteration on predefined maze sizes of 25,45,65 are")
    for maze_size in maze_sizes:
        value_times=list()
        v_iteration=list()
        v_mem=list()
        p_iteration=list()
        policy_times=list()
        p_mem=list()
        for i in range(3):
            maze = generate_maze(maze_size)
            #value iteration algorithm
            start_point = (1, 0) 
            goal_state = (maze_size-1, maze_size-2)  
            def wrap_value():
                start_time = time.time()
                value_map, viteration = value_iteration_with_bool(maze, goal_state, False)
                policy = extract_policy_value(value_map, maze, goal_state)
                path = find_path_value(policy, start_point, goal_state, maze)
                end_time = time.time()
                value_times.append(end_time-start_time)
                v_iteration.append(viteration)

            #policy iteration algorithm
            def wrap_policy():
                start_time = time.time()
                # policy, piteration = policy_iteration(maze,goal_state)
                policy, piteration = policy_iteration_with_bool(maze,goal_state,False)
                path = find_path_policy(policy, start_point, goal_state, maze)
                end_time = time.time()
                policy_times.append(end_time-start_time)
                p_iteration.append(piteration)
            v_mem.append(mean(memory_usage(proc=wrap_value)))
            p_mem.append(mean(memory_usage(proc=wrap_policy)))
        print(f"Time taken for Value iteration for size "+str(maze_size)+" is: "+(str(sum(value_times) / len(value_times)))+ "seconds & Iteration taken are "+str(int(sum(v_iteration) / len(v_iteration))))
        print(f"Memory usage of Value iteration: {str(sum(v_mem) / len(v_mem))} MiB")
        print(f"Time taken for Policy iteration for size "+str(maze_size)+" is: "+(str(sum(policy_times) / len(policy_times)))+ "seconds & Iterations taken are "+ str(int(sum(p_iteration) / len(p_iteration))))
        print(f"Memory usage of Policy iteration: {str(sum(p_mem) / len(p_mem))} MiB")
    time.sleep(2)


#code for visualisation of different algorithms on same maze

    maze_size = int(arg1)  # Adjust size as needed
    maze = generate_maze(maze_size)
    start_point = (1, 0)  # Adjust as needed
    goal_state = (maze_size-1, maze_size-2)  # Adjust the goal state as needed

    value_map, _ = value_iteration_with_bool(maze, goal_state, True)
    policy = extract_policy_value(value_map, maze, goal_state)
    # Plot the policy on the maze
    plot_policy_on_maze_value(maze, policy)
    # Find and visualize the path using the final policy
    path = find_path_value(policy, start_point, goal_state, maze)
    visualize_path_value(maze, path)


    policy, _ = policy_iteration_with_bool(maze,goal_state, True)
    # Find and visualize the path using the final policy
    path = find_path_policy(policy, start_point, goal_state, maze)
    visualize_path_policy(maze, path)

if __name__ == '__main__':
    # This code won't run if this file is imported.
    args = sys.argv[1:]

    main(*args)
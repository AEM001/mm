import numpy as np
import csv

# Constants
NUM_NODES = 100  # 10x10 grid, nodes 0-99
GRID_SIZE = 10
DIRECTIONS = 4  # 0: up, 1: right, 2: down, 3: left
DISTANCE_L = 50  # km
POSSIBLE_SPEEDS = [40, 60, 90, 120]  # km/h
START_NODE = 0
END_NODE = 99

# Action space: (direction, speed_index)
# speed_index refers to an index in POSSIBLE_SPEEDS

class QLearningEnv:
    def __init__(self, limits_row_file, limits_col_file):
        self.limits_row = self._load_limits(limits_row_file, is_row_limits=True)
        self.limits_col = self._load_limits(limits_col_file, is_row_limits=False)
        self.q_table = np.zeros((NUM_NODES, DIRECTIONS, len(POSSIBLE_SPEEDS)))
        self.current_node = START_NODE
        self.steps_taken = 0

    def _load_limits(self, filepath, is_row_limits):
        data = np.loadtxt(filepath, delimiter=',', dtype=int)
        if is_row_limits: # limits_row.csv
            assert data.shape == (10, 9), f"Expected shape (10,9) for {filepath}, got {data.shape}"
        else: # limits_col.csv
            assert data.shape == (9, 10), f"Expected shape (9,10) for {filepath}, got {data.shape}"
        return data

    def reset(self):
        self.current_node = START_NODE
        self.steps_taken = 0
        return self.current_node

    def get_node_coords(self, node_id):
        # 0 表示最底下那排(节点1-10)
        # 0 表示最左边(节点1,11,21…)
        # n // 10 gives row, n % 10 gives col, matching 0-indexed node_id
        r = node_id // GRID_SIZE  # Row 0 is the bottom row (nodes 0-9)
        c = node_id % GRID_SIZE   # Col 0 is the leftmost column (nodes 0, 10, 20,...)
        return r, c

    def get_speed_limit(self, current_node, direction):
        r, c = self.get_node_coords(current_node)
        limit = 0
        # Directions: 0: up (North), 1: right (East), 2: down (South), 3: left (West)
        if direction == 0:  # Up (North)
            if r < GRID_SIZE - 1: # Max row index is 9 (for 10 rows, 0-9)
                limit = self.limits_col[r, c] # r-th row in limits_col corresponds to links between row r and r+1
        elif direction == 1:  # Right (East)
            if c < GRID_SIZE - 1: # Max col index is 9 (for 10 columns, 0-9)
                limit = self.limits_row[r, c] # c-th col in limits_row corresponds to links between col c and c+1
        elif direction == 2:  # Down (South)
            if r > 0:
                limit = self.limits_col[r - 1, c] # (r-1)-th row in limits_col for links between row r-1 and r
        elif direction == 3:  # Left (West)
            if c > 0:
                limit = self.limits_row[r, c - 1] # (c-1)-th col in limits_row for links between col c-1 and c
        return limit

    def step(self, action_tuple):
        direction, speed_idx = action_tuple
        chosen_speed = POSSIBLE_SPEEDS[speed_idx]
        self.steps_taken += 1

        current_r, current_c = self.get_node_coords(self.current_node)
        speed_limit_on_path = self.get_speed_limit(self.current_node, direction)

        # Illegal action: No path or speed > limit
        if speed_limit_on_path == 0 or chosen_speed > speed_limit_on_path:
            reward = -200  # Large penalty for illegal move
            done = self.steps_taken > 150 # Episode can end if too many steps
            return self.current_node, reward, done, False # False indicates illegal move

        # Calculate next node based on new coordinate system and direction definitions
        # Directions: 0: up (North, r increases), 1: right (East, c increases), 2: down (South, r decreases), 3: left (West, c decreases)
        next_node = -1 # Placeholder for invalid next_node
        if direction == 0:  # Up (North)
            if current_r < GRID_SIZE - 1:
                next_node = self.current_node + GRID_SIZE
        elif direction == 1:  # Right (East)
            if current_c < GRID_SIZE - 1:
                next_node = self.current_node + 1
        elif direction == 2:  # Down (South)
            if current_r > 0:
                next_node = self.current_node - GRID_SIZE
        elif direction == 3:  # Left (West)
            if current_c > 0:
                next_node = self.current_node - 1

        # If next_node remained -1, it means the move was to a boundary where it shouldn't go based on speed_limit check
        # This should ideally be caught by speed_limit_on_path == 0, but as a safeguard:
        if next_node == -1 or next_node < 0 or next_node >= NUM_NODES: # Should not happen if speed_limit logic is correct
            # This case implies an issue, possibly trying to move off-grid despite limit checks
            # For robustness, treat as illegal move, though get_speed_limit should prevent this.
            reward = -300 # Even larger penalty for logic error
            done = self.steps_taken > 150
            return self.current_node, reward, done, False
        self.current_node = next_node

        time_taken = DISTANCE_L / chosen_speed
        reward = -time_taken

        done = (self.current_node == END_NODE) or (self.steps_taken > 150)
        return self.current_node, reward, done, True # True indicates legal move

    def get_valid_actions_indices(self, node_id):
        valid_actions = []
        for d_idx in range(DIRECTIONS):
            limit = self.get_speed_limit(node_id, d_idx)
            if limit > 0:
                for s_idx, speed in enumerate(POSSIBLE_SPEEDS):
                    if speed <= limit:
                        valid_actions.append((d_idx, s_idx))
        return valid_actions


def train(env, episodes=10000, alpha=0.2, gamma=0.95, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay_rate=2e-4):
    epsilon = epsilon_start
    episode_rewards = []
    episode_times = []

    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        total_time_this_episode = 0

        while not done:
            valid_actions = env.get_valid_actions_indices(state)
            if not valid_actions: # Should not happen in a connected graph with penalties
                # If stuck, penalize and end episode or break
                # This case might indicate an issue with graph connectivity or reward structure
                # For now, let's assume the graph allows reaching the end or max steps
                break 

            if np.random.rand() < epsilon:
                # Explore: choose a random valid action
                action_idx = np.random.choice(len(valid_actions))
                action = valid_actions[action_idx]
            else:
                # Exploit: choose the best action from Q-table among valid ones
                q_values_for_state = env.q_table[state]
                # Filter Q-values for valid actions only
                valid_q_values = { (d,s): q_values_for_state[d,s] for d,s in valid_actions }
                if not valid_q_values: # If all valid actions have Q-value 0 (e.g. at start)
                    action_idx = np.random.choice(len(valid_actions))
                    action = valid_actions[action_idx]
                else:
                    action = max(valid_q_values, key=valid_q_values.get)
            
            next_state, reward, done, legal_move = env.step(action)
            total_reward += reward
            if legal_move and reward != -200: # Only add time for legal, non-penalized moves
                 # reward is -time_taken
                total_time_this_episode += (-reward)

            # Q-Learning update
            old_value = env.q_table[state, action[0], action[1]]
            if done:
                next_max = 0 # No future reward if episode is done
            else:
                # Consider only valid actions for next_max
                next_valid_actions = env.get_valid_actions_indices(next_state)
                if not next_valid_actions:
                    next_max = -np.inf # Penalize if no valid actions from next state
                else:
                    q_values_for_next_state = env.q_table[next_state]
                    valid_next_q_values = [q_values_for_next_state[d,s] for d,s in next_valid_actions]
                    if not valid_next_q_values:
                        next_max = 0 # Or some other default if all are zero
                    else:
                        next_max = np.max(valid_next_q_values)
            
            new_value = old_value + alpha * (reward + gamma * next_max - old_value)
            env.q_table[state, action[0], action[1]] = new_value
            state = next_state

        epsilon = max(epsilon_end, epsilon - epsilon_decay_rate)
        episode_rewards.append(total_reward)
        if total_time_this_episode > 0 : # Avoid logging 0 time if episode ended due to penalty
            episode_times.append(total_time_this_episode)

        if (episode + 1) % 100 == 0:
            avg_time = np.mean(episode_times[-100:]) if episode_times else 0
            print(f"Episode {episode + 1}/{episodes}, Epsilon: {epsilon:.3f}, Avg Time (last 100): {avg_time:.2f}h")
            if avg_time > 0 and avg_time < 9 and episode > 3000: # Heuristic for convergence
                 if len(episode_times) > 200 and abs(np.mean(episode_times[-200:-100]) - avg_time) < 0.1:
                    print("Converged early.")
                    # break # Optional: break early if converged

    return env.q_table, episode_times

def extract_path(q_table, env):
    path = [START_NODE]
    time_taken_list = []
    current_node = START_NODE
    total_time = 0
    max_path_steps = NUM_NODES * 2 # Safety break for path extraction
    steps = 0

    while current_node != END_NODE and steps < max_path_steps:
        valid_actions = env.get_valid_actions_indices(current_node)
        if not valid_actions:
            print("Error: Stuck during path extraction, no valid actions.")
            return [], -1

        q_values_for_state = q_table[current_node]
        best_action = None
        max_q = -np.inf

        for d_idx, s_idx in valid_actions:
            if q_values_for_state[d_idx, s_idx] > max_q:
                max_q = q_values_for_state[d_idx, s_idx]
                best_action = (d_idx, s_idx)
        
        if best_action is None: # Should be covered by valid_actions check
            print("Error: Could not find a best action during path extraction.")
            return [], -1

        direction, speed_idx = best_action
        chosen_speed = POSSIBLE_SPEEDS[speed_idx]
        
        # Simulate step to get next node (without modifying env state)
        current_r, current_c = env.get_node_coords(current_node)
        next_r, next_c = current_r, current_c
        if direction == 0: next_r += 1 # UP, r increases
        elif direction == 1: next_c += 1
        elif direction == 2: next_r -= 1 # DOWN, r decreases
        elif direction == 3: next_c -= 1
        
        next_node = next_r * GRID_SIZE + next_c
        current_node = next_node
        path.append(current_node)
        
        time_for_segment = DISTANCE_L / chosen_speed
        time_taken_list.append(time_for_segment)
        total_time += time_for_segment
        steps += 1
        
        if current_node in path[:-1]: # Detect cycle
            print("Warning: Cycle detected during path extraction.")
            # return path, total_time # Or handle as error
            # For now, let it continue, but this indicates suboptimal Q-table or exploration issue

    if current_node != END_NODE:
        print("Error: Path extraction did not reach the end node.")
        return [], -1
        
    return path, total_time

if __name__ == '__main__':
    # Path to data files - assuming they are in a 'data' subdirectory relative to the script
    # Or provide absolute paths
    # For this example, let's assume they are in 'd:\Code\mm\data\'
    limits_row_file = 'd:\\Code\\mm\\data\\limits_row.csv'
    limits_col_file = 'd:\\Code\\mm\\data\\limits_col.csv'

    print("Initializing environment...")
    env = QLearningEnv(limits_row_file, limits_col_file)

    # Unit tests / Quick self-check
    print("--- Self-check for get_speed_limit ---")
    # Node 0 (1 in 1-based), Direction 1 (Right)
    print(f"Speed limit (Node 0, Right): {env.get_speed_limit(0, 1)}")
    print(f"Expected from limits_row[0,0]: {env.limits_row[0,0]}")
    # Node 0 (1 in 1-based), Direction 0 (Up)
    print(f"Speed limit (Node 0, Up): {env.get_speed_limit(0, 0)}")
    print(f"Expected from limits_col[0,0]: {env.limits_col[0,0]}")
    # Node 90 (91 in 1-based), Direction 1 (Right)
    print(f"Speed limit (Node 90, Right): {env.get_speed_limit(90, 1)}")
    print(f"Expected from limits_row[9,0]: {env.limits_row[9,0]}")
    print("-------------------------------------")

    print("Starting training...")
    # Reduced episodes for quicker testing, increase for better results
    q_table_trained, episode_times_history = train(env, episodes=6000, alpha=0.1, gamma=0.98, epsilon_decay_rate=0.0002)
    # q_table_trained, episode_times_history = train(env, episodes=10000, alpha=0.2, gamma=0.95, epsilon_decay_rate=2e-4)

    print("\nTraining finished.")
    
    # Plotting episode times (optional, requires matplotlib)
    try:
        import matplotlib.pyplot as plt
        plt.plot(episode_times_history)
        plt.title('Episode Travel Time Over Training')
        plt.xlabel('Episode')
        plt.ylabel('Total Time (hours)')
        plt.savefig('episode_times.png')
        print("Saved episode times plot to episode_times.png")
        # plt.show() # Uncomment to display plot if running in an interactive environment
    except ImportError:
        print("Matplotlib not found, skipping plot generation.")

    print("\nExtracting optimal path...")
    optimal_path, t1 = extract_path(q_table_trained, env)

    if optimal_path:
        print(f"\nRoute One Node Sequence (0-indexed):")
        print(" -> ".join(map(str, optimal_path)))
        print(f"\nRoute One Node Sequence (1-indexed for reference.md format):")
        print(" -> ".join(map(lambda x: str(x+1), optimal_path)))
        print(f"\nTotal time T1: {t1:.2f} hours")
    else:
        print("\nCould not find a valid path to the destination.")

    # Example of checking a specific Q-value (optional)
    # print("\nQ-value for state 0, action (down, 120km/h):", q_table_trained[0, 2, POSSIBLE_SPEEDS.index(120)])
    
    
# - limits_col.csv** (9x10)**:
#   - 代表**纵向**限速。
#   - 行 (0-8): 对应路段的起始节点的**行编号**。例如，limits_col 的第0行数据，对应的是从原始网格第0行节点（即节点0-9）出发**向上**的路段限速。
#   - 列 (0-9): 对应路段的**列编号**。例如，limits_col[r, c] 表示从网格坐标 (r, c) 的节点出发**向上**到 (r+1, c) 的路段限速。

# - limits_row.csv** (10x9)**:
#   - 代表**横向**限速。
#   - 行 (0-9): 对应路段的节点的**行编号**。例如，limits_row 的第0行数据，对应的是在原始网格第0行的节点（即节点0-9）之间**向右**的路段限速。
#   - 列 (0-8): 对应路段的起始节点的**列编号**。例如，limits_row[r, c] 表示从网格坐标 (r, c) 的节点出发**向右**到 (r, c+1) 的路段限速。

# 代码中的关键部分：

# 1. get_node_coords(self, node_id)** 方法**:
# // ... existing code ...
# def get_node_coords(self, node_id):
#     # 0 表示最底下那排(节点1-10)
#     # 0 表示最左边(节点1,11,21…)
#     # n // 10 gives row, n % 10 gives col, matching 0-indexed node_id
#     r = node_id // GRID_SIZE  # Row 0 is the bottom row (nodes 0-9)
#     c = node_id % GRID_SIZE   # Col 0 is the leftmost column (nodes 0, 10, 20,...)
#     return r, c
# // ... existing code ...
#   此方法将0-99的节点编号正确转换为0索引的网格行号 r (0-9) 和列号 c (0-9)，这与您的描述（如节点1-10为最底层，即行0）是一致的。

# 2. get_speed_limit(self, current_node, direction)** 方法**:
# // ... existing code ...
# def get_speed_limit(self, current_node, direction):
#     r, c = self.get_node_coords(current_node)
#     limit = 0
#     # Directions: 0: up (North), 1: right (East), 2: down (South), 3: left (West)
#     if direction == 0:  # Up (North)
#         if r < GRID_SIZE - 1: # Max row index is 9 (for 10 rows, 0-9)
#             limit = self.limits_col[r, c] # r-th row in limits_col corresponds to links between row r and r+1
#     elif direction == 1:  # Right (East)
#         if c < GRID_SIZE - 1: # Max col index is 9 (for 10 columns, 0-9)
#             limit = self.limits_row[r, c] # c-th col in limits_row corresponds to links between col c and c+1
#     elif direction == 2:  # Down (South)
#         if r > 0:
#             limit = self.limits_col[r - 1, c] # (r-1)-th row in limits_col for links between row r-1 and r
#     elif direction == 3:  # Left (West)
#         if c > 0:
#             limit = self.limits_row[r, c - 1] # (c-1)-th col in limits_row for links between col c-1 and c
#     return limit
# // ... existing code ...
#   - 向上 (direction 0) 从节点 (r,c): 代码使用 self.limits_col[r, c]。这表示 limits_col 的第 r 行、第 c 列存储的是从网格节点 (r,c) 向上的限速。这与您的描述相符。
#   - 向右 (direction 1) 从节点 (r,c): 代码使用 self.limits_row[r, c]。这表示 limits_row 的第 r 行、第 c 列存储的是从网格节点 (r,c) 向右的限速。这与您的描述相符。
#   - 向下 (direction 2) 从节点 (r,c): 相当于从节点 (r-1,c) 向上到 (r,c)。代码使用 self.limits_col[r-1, c]。这表示 limits_col 的第 r-1 行、第 c 列存储的是从网格节点 (r-1,c) 向上的限速。这与您的描述相符。
#   - 向左 (direction 3) 从节点 (r,c): 相当于从节点 (r,c-1) 向右到 (r,c)。代码使用 self.limits_row[r, c-1]。这表示 limits_row 的第 r 行、第 c-1 列存储的是从网格节点 (r,c-1) 向右的限速。这与您的描述相符。
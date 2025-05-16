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
            return self.current_node, reward, done, {"cost": 0, "time_spent":0, "legal_move": False} 

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

        if next_node == -1 or next_node < 0 or next_node >= NUM_NODES: 
            reward = -300 
            done = self.steps_taken > 150
            return self.current_node, reward, done, {"cost": 0, "time_spent":0, "legal_move": False}
        
        self.current_node = next_node

        # Cost calculation based on reference.md
        time_taken = DISTANCE_L / chosen_speed
        c_meal = 20 * time_taken
        q_fuel = (0.0625 * chosen_speed + 1.875) * (DISTANCE_L / 100.0)
        c_fuel = q_fuel * 7.76
        c_toll = 25.0 if speed_limit_on_path == 120 else 0.0
        c_edge = c_meal + c_fuel + c_toll
        reward = -c_edge

        done = (self.current_node == END_NODE) or (self.steps_taken > 150)
        return self.current_node, reward, done, {"cost": c_edge, "time_spent": time_taken, "legal_move": True}

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
    episode_rewards = [] # Will store total negative costs
    episode_costs = [] # Will store actual total costs

    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward_this_episode = 0 # Sum of negative costs
        total_cost_this_episode = 0

        while not done:
            valid_actions = env.get_valid_actions_indices(state)
            if not valid_actions: 
                break 

            if np.random.rand() < epsilon:
                action_idx = np.random.choice(len(valid_actions))
                action = valid_actions[action_idx]
            else:
                q_values_for_state = env.q_table[state]
                valid_q_values = { (d,s): q_values_for_state[d,s] for d,s in valid_actions }
                if not valid_q_values: 
                    action_idx = np.random.choice(len(valid_actions))
                    action = valid_actions[action_idx]
                else:
                    action = max(valid_q_values, key=valid_q_values.get)
            
            next_state, reward, done, info = env.step(action)
            total_reward_this_episode += reward
            if info["legal_move"]:
                 total_cost_this_episode += info["cost"]

            old_value = env.q_table[state, action[0], action[1]]
            if done:
                next_max = 0 
            else:
                next_valid_actions = env.get_valid_actions_indices(next_state)
                if not next_valid_actions:
                    next_max = -np.inf 
                else:
                    q_values_for_next_state = env.q_table[next_state]
                    valid_next_q_values = [q_values_for_next_state[d,s] for d,s in next_valid_actions]
                    if not valid_next_q_values:
                        next_max = 0 
                    else:
                        next_max = np.max(valid_next_q_values)
            
            new_value = old_value + alpha * (reward + gamma * next_max - old_value)
            env.q_table[state, action[0], action[1]] = new_value
            state = next_state

        epsilon = max(epsilon_end, epsilon - epsilon_decay_rate)
        episode_rewards.append(total_reward_this_episode)
        if info["legal_move"] and total_cost_this_episode > 0:
            episode_costs.append(total_cost_this_episode)

        if (episode + 1) % 100 == 0:
            avg_cost = np.mean(episode_costs[-100:]) if episode_costs else 0
            print(f"Episode {episode + 1}/{episodes}, Epsilon: {epsilon:.3f}, Avg Cost (last 100): {avg_cost:.2f}")
            # Convergence check (example, can be refined)
            if avg_cost > 0 and episode > 3000: 
                 if len(episode_costs) > 200 and abs(np.mean(episode_costs[-200:-100]) - avg_cost) < 1.0: # Cost convergence might be slower
                    print("Converged early by cost.")
                    # break 

    return env.q_table, episode_costs

def extract_path(q_table, env):
    path = [START_NODE]
    current_node = START_NODE
    total_cost_c1 = 0.0
    max_path_steps = NUM_NODES * 2 
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
        
        if best_action is None: 
            print("Error: Could not find a best action during path extraction.")
            return [], -1

        direction, speed_idx = best_action
        chosen_speed = POSSIBLE_SPEEDS[speed_idx]
        
        # Calculate cost for this segment
        current_segment_speed_limit = env.get_speed_limit(current_node, direction)
        time_for_segment = DISTANCE_L / chosen_speed
        c_meal_segment = 20 * time_for_segment
        q_fuel_segment = (0.0625 * chosen_speed + 1.875) * (DISTANCE_L / 100.0)
        c_fuel_segment = q_fuel_segment * 7.76
        c_toll_segment = 25.0 if current_segment_speed_limit == 120 else 0.0
        segment_cost = c_meal_segment + c_fuel_segment + c_toll_segment
        total_cost_c1 += segment_cost
        
        # Simulate step to get next node
        current_r, current_c = env.get_node_coords(current_node)
        next_r, next_c = current_r, current_c
        if direction == 0: next_r += 1 
        elif direction == 1: next_c += 1
        elif direction == 2: next_r -= 1 
        elif direction == 3: next_c -= 1
        
        next_node = next_r * GRID_SIZE + next_c
        current_node = next_node
        path.append(current_node)
        steps += 1
        
        if current_node in path[:-1]: 
            print("Warning: Cycle detected during path extraction.")
            # return path, total_cost_c1 # Or handle as error

    if current_node != END_NODE:
        print("Error: Path extraction did not reach the end node.")
        return [], -1
        
    return path, total_cost_c1

if __name__ == '__main__':
    limits_row_file = 'd:\\Code\\mm\\data\\limits_row.csv'
    limits_col_file = 'd:\\Code\\mm\\data\\limits_col.csv'

    print("Initializing environment for cost minimization...")
    env = QLearningEnv(limits_row_file, limits_col_file)

    print("Starting training for cost minimization...")
    # Hyperparameters from reference.md (or adjust as needed)
    # alpha: 0.15 – 0.25, gamma: 0.95 – 0.99, epsilon decay over ~6000 episodes
    q_table_trained_cost, episode_costs_history = train(env, episodes=6000, alpha=0.2, gamma=0.98, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay_rate=1.0/4000) # Adjusted decay for ~6k episodes

    print("\nTraining finished.")
    
    try:
        import matplotlib.pyplot as plt
        plt.plot(episode_costs_history)
        plt.title('Episode Cost Over Training')
        plt.xlabel('Episode')
        plt.ylabel('Total Cost (C1)')
        plt.savefig('episode_costs_c1.png')
        print("Saved episode costs plot to episode_costs_c1.png")
    except ImportError:
        print("Matplotlib not found, skipping plot generation.")

    print("\nExtracting optimal path for C1...")
    optimal_path_cost, min_total_cost_c1 = extract_path(q_table_trained_cost, env)

    if optimal_path_cost:
        print(f"\nRoute Two (Cost Minimization) Node Sequence (0-indexed):")
        print(" -> ".join(map(str, optimal_path_cost)))
        print(f"\nRoute Two Node Sequence (1-indexed for reference.md format):")
        print(" -> ".join(map(lambda x: str(x+1), optimal_path_cost)))
        print(f"\nMinimum Total Cost C1: {min_total_cost_c1:.2f}")
    else:
        print("\nCould not find a valid path to the destination for cost minimization.")
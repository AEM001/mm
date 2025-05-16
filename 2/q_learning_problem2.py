import numpy as np
import csv
import math
import os
import random
from datetime import datetime

# --- Constants based on plan.md and problem description ---
GRID_SIZE = 10
NUM_NODES = GRID_SIZE * GRID_SIZE
START_NODE = 0
END_NODE = NUM_NODES - 1
SEGMENT_LENGTH = 50  # km

T1 = 10.83  # hours, from problem 1's shortest_analysis.md
DEADLINE = 0.7 * T1  # hours

# Speeding options
SPEEDING_RATIOS = np.array([0.0, 0.2, 0.5, 0.7])
NUM_SPEEDING_LEVELS = len(SPEEDING_RATIOS)

# Radar and detection probabilities
PROB_MOBILE_RADAR = 1/9  # p_m
SINGLE_RADAR_DETECTION_RATES = { # p1(r)
    0.0: 0.0,
    0.2: 0.7,
    0.5: 0.9,
    0.7: 0.99
}

# Directions (0: Up, 1: Right, 2: Down, 3: Left)
DIRECTIONS = {'UP': 0, 'RIGHT': 1, 'DOWN': 2, 'LEFT': 3}
NUM_DIRECTIONS = 4
DX = [0, 1, 0, -1]  # Change in col for UP, RIGHT, DOWN, LEFT
DY = [1, 0, -1, 0]  # Change in row for UP, RIGHT, DOWN, LEFT (row 0 is bottom)

# Q-Learning Hyperparameters
ALPHA = 0.1  # Learning rate
GAMMA = 0.98  # Discount factor
EPSILON_START = 1.0
EPSILON_END = 0.05
NUM_EPISODES = 80000 # Increased episodes for Lagrange method
EPSILON_DECAY_RATE = (EPSILON_START - EPSILON_END) / (NUM_EPISODES * 0.8) # Decay over 80% of episodes

BETA_LAMBDA = 0.01 # Learning rate for lambda

MAX_STEPS_PER_EPISODE = NUM_NODES * 2 # Generous limit

# HUGE_PENALTY_TIMEOUT removed, handled by lambda
HUGE_PENALTY_ILLEGAL_MOVE = 200000

# Output file
OUTPUT_FILE_PATH = "d:/Code/mm/2/output_problem2.txt"
Q_TABLE_SAVE_PATH = "d:/Code/mm/2/q_table_problem2.npy" # Optional: to save/load Q-table

# --- Data Loading ---
def load_speed_limits(base_path="d:/Code/mm/data"):
    limits_row_path = os.path.join(base_path, "limits_row.csv")
    limits_col_path = os.path.join(base_path, "limits_col.csv")
    
    # limits_row.csv (10x9): Horizontal speed limits (Right)
    # Row i, Col j corresponds to limit from node (i,j) to (i,j+1)
    limits_row = []
    with open(limits_row_path, 'r') as f:
        reader = csv.reader(f)
        for row_data in reader:
            limits_row.append([int(x) for x in row_data])
    
    # limits_col.csv (9x10): Vertical speed limits (Up)
    # Row i, Col j corresponds to limit from node (i,j) to (i+1,j)
    # Note: The CSV is 9 rows, meaning from row 0 to row 8.
    # So, limits_col[r][c] is the limit from (r,c) to (r+1,c)
    limits_col = []
    with open(limits_col_path, 'r') as f:
        reader = csv.reader(f)
        for row_data in reader:
            limits_col.append([int(x) for x in row_data])
            
    return np.array(limits_row), np.array(limits_col)

# --- Helper Functions ---
def get_node_coords(node_id):
    # Node 0 is (0,0) at bottom-left. Row increases upwards.
    r = node_id // GRID_SIZE
    c = node_id % GRID_SIZE
    return r, c

def get_node_id(r, c):
    if 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE:
        return r * GRID_SIZE + c
    return -1 # Invalid node

def get_speed_limit_on_segment(current_node_id, direction_idx, limits_row, limits_col):
    r, c = get_node_coords(current_node_id)
    
    if direction_idx == DIRECTIONS['UP']: # Moving from (r,c) to (r+1,c)
        if r < GRID_SIZE - 1:
            return limits_col[r][c] 
    elif direction_idx == DIRECTIONS['RIGHT']: # Moving from (r,c) to (r,c+1)
        if c < GRID_SIZE - 1:
            return limits_row[r][c]
    elif direction_idx == DIRECTIONS['DOWN']: # Moving from (r,c) to (r-1,c)
        if r > 0:
            # This is equivalent to UP from (r-1,c) to (r,c)
            return limits_col[r-1][c]
    elif direction_idx == DIRECTIONS['LEFT']: # Moving from (r,c) to (r,c-1)
        if c > 0:
            # This is equivalent to RIGHT from (r,c-1) to (r,c)
            return limits_row[r][c-1]
    return -1 # Invalid move or boundary

def calculate_travel_cost(actual_speed_v, segment_limit_V_lim):
    if actual_speed_v <= 0: return float('inf') # Avoid division by zero
    time_t = SEGMENT_LENGTH / actual_speed_v  # hours
    
    # Fuel consumption per 50km (q_50km)
    # Formula from reference: (0.0625 * v + 1.875) is per 100km
    fuel_per_100km = (0.0625 * actual_speed_v + 1.875)
    q_50km = fuel_per_100km * 0.5 # liters
    
    cost_food_lodging = 20 * time_t
    cost_fuel = 7.76 * q_50km
    cost_toll = 25 if segment_limit_V_lim == 120 else 0
    
    return cost_food_lodging + cost_fuel + cost_toll

def get_fine_amount(V_lim, speeding_ratio_r):
    if speeding_ratio_r == 0: return 0
    
    # Based on problem description's traffic regulations
    if V_lim <= 50: # e.g., V_lim = 40
        if 0 < speeding_ratio_r <= 0.2: return 50
        if 0.2 < speeding_ratio_r <= 0.5: return 100
        if 0.5 < speeding_ratio_r <= 0.7: return 300
    elif 50 < V_lim <= 80: # e.g., V_lim = 60
        if 0 < speeding_ratio_r <= 0.2: return 100
        if 0.2 < speeding_ratio_r <= 0.5: return 150
        if 0.5 < speeding_ratio_r <= 0.7: return 500
    elif 80 < V_lim < 100: # e.g., V_lim = 90
        if 0 < speeding_ratio_r <= 0.2: return 150
        if 0.2 < speeding_ratio_r <= 0.5: return 200
        if 0.5 < speeding_ratio_r <= 0.7: return 1000
    elif V_lim >= 100: # e.g., V_lim = 120
        # "exceeding up to 50%" means r <= 0.5
        if 0 < speeding_ratio_r <= 0.5: return 200
        if 0.5 < speeding_ratio_r <= 0.7: return 1500
    return 0 # Should not happen if r is one of the defined ratios

def calculate_detection_probability(V_lim, speeding_ratio_r):
    if speeding_ratio_r == 0: return 0
    
    p1_r = SINGLE_RADAR_DETECTION_RATES.get(speeding_ratio_r, 0)
    if p1_r == 0: return 0

    has_fixed_radar = (V_lim >= 90)
    
    # Probability of being detected by at least one radar on the segment
    # N_mobile_expected = 20 * (1/180) = 1/9 for any given segment (assuming 180 segments total)
    # This was simplified to PROB_MOBILE_RADAR = 1/9 as the probability a mobile radar is on THIS segment.

    if not has_fixed_radar:
        # Only mobile radar possible. P(detected) = P(mobile radar present) * P(detected by it)
        return PROB_MOBILE_RADAR * p1_r
    else:
        # Fixed radar is present.
        # P(Detected) = 1 - P(Not detected by any)
        # P(Not detected by any) = P(Not detected by fixed) * P(Not detected by mobile if present OR mobile not present)
        # P(Not detected by fixed) = (1 - p1_r)
        # P(Mobile not present) = (1 - PROB_MOBILE_RADAR)
        # P(Mobile present AND not detected by it) = PROB_MOBILE_RADAR * (1 - p1_r)
        # P(Not detected by mobile radar system) = (1 - PROB_MOBILE_RADAR) + PROB_MOBILE_RADAR - PROB_MOBILE_RADAR * p1_r
        #                                      = 1 - PROB_MOBILE_RADAR + PROB_MOBILE_RADAR - PROB_MOBILE_RADAR * p1_r
        #                                      = 1 - PROB_MOBILE_RADAR * p1_r

        # P(Not Detected) = (1 - p1_r) * (1 - PROB_MOBILE_RADAR * p1_r)
        # P(Detected) = 1 - (1 - p1_r) * (1 - PROB_MOBILE_RADAR * p1_r)
        # This formula correctly accounts for the fixed radar and the *possibility* of a mobile radar.
        # Let's use the formula from plan.md which is P_det(r) = p1(r) * (1 + p_m - p_m * p1(r)) for segments with fixed radar.
        # This formula is: P(det by fixed) + P(det by mobile AND NOT by fixed)
        # = p1(r) + [ PROB_MOBILE_RADAR * p1(r) * (1-p1(r)) ] --- This is not quite right.

        # Let's re-derive P(Detected by at least one radar on a segment with a fixed radar)
        # Event A: Detected by fixed radar. P(A) = p1_r
        # Event B: Mobile radar is present AND detects. P(B) = PROB_MOBILE_RADAR * p1_r
        # We want P(A or B) = P(A) + P(B) - P(A and B)
        # P(A and B) = P(Detected by fixed AND Mobile radar present AND detected by mobile)
        #            = p1_r * PROB_MOBILE_RADAR * p1_r (assuming detection events are independent given presence)
        # So, P(Detected) = p1_r + PROB_MOBILE_RADAR * p1_r - p1_r * PROB_MOBILE_RADAR * p1_r
        #                 = p1_r * (1 + PROB_MOBILE_RADAR - PROB_MOBILE_RADAR * p1_r)
        # This matches the formula in plan.md.
        return p1_r * (1 + PROB_MOBILE_RADAR - PROB_MOBILE_RADAR * p1_r)

# The previous implementation was:
# prob_no_mobile = (1 - PROB_MOBILE_RADAR) * p1_r  <- This is P(Mobile not present AND detected by Fixed)
# prob_with_mobile = PROB_MOBILE_RADAR * (1 - (1 - p1_r)**2) <- This is P(Mobile present AND detected by (Fixed OR Mobile))
# return prob_no_mobile + prob_with_mobile
# This sums to: (1-p_m)p1_r + p_m(1-(1-p1_r)^2) = p1_r - p_m*p1_r + p_m(2p1_r - p1_r^2)
# = p1_r - p_m*p1_r + 2*p_m*p1_r - p_m*p1_r^2 = p1_r + p_m*p1_r - p_m*p1_r^2
# = p1_r * (1 + p_m - p_m*p1_r).
# So, the previous implementation was actually correct and equivalent to the plan.md formula. No change needed here.

# --- Helper Functions ---
def calculate_expected_fine(V_lim, speeding_ratio_r):
    fine_amount = get_fine_amount(V_lim, speeding_ratio_r)
    if fine_amount == 0: return 0
    
    detection_prob = calculate_detection_probability(V_lim, speeding_ratio_r)
    return detection_prob * fine_amount

def calculate_c_edge(segment_limit_V_lim, speeding_ratio_r):
    actual_speed_v = segment_limit_V_lim * (1 + speeding_ratio_r)
    
    travel_cost = calculate_travel_cost(actual_speed_v, segment_limit_V_lim)
    expected_fine = calculate_expected_fine(segment_limit_V_lim, speeding_ratio_r)
    
    # Calculate time taken for this segment as it's needed for Lagrangian reward
    if actual_speed_v <= 0:
        time_taken_segment = float('inf')
    else:
        time_taken_segment = SEGMENT_LENGTH / actual_speed_v
        
    return travel_cost + expected_fine, time_taken_segment

# get_time_bucket function is no longer needed and should be removed.
# def get_time_bucket(remaining_time):
#     if remaining_time < 0: return -1 
#     bucket = math.floor(remaining_time / TIME_BUCKET_WIDTH)
#     return max(0, min(bucket, NUM_TIME_BUCKETS - 1))


class QLearningAgent:
    def __init__(self, limits_row, limits_col):
        # Q-table: [node, direction, speeding_level]
        self.q_table = np.zeros((NUM_NODES, NUM_DIRECTIONS, NUM_SPEEDING_LEVELS))
        self.limits_row = limits_row
        self.limits_col = limits_col
        self.epsilon = EPSILON_START
        self.lambda_val = 0.0 # Initialize Lagrange multiplier

    def choose_action(self, current_node_id): # Removed current_time_bucket
        if random.uniform(0, 1) < self.epsilon:
            # Explore: choose a random valid action
            valid_actions = []
            for d_idx in range(NUM_DIRECTIONS):
                r, c = get_node_coords(current_node_id)
                next_r, next_c = r + DY[d_idx], c + DX[d_idx]
                if 0 <= next_r < GRID_SIZE and 0 <= next_c < GRID_SIZE: # Check boundary
                    for s_idx in range(NUM_SPEEDING_LEVELS):
                        valid_actions.append((d_idx, s_idx))
            if not valid_actions: return None 
            return random.choice(valid_actions)
        else:
            # Exploit: choose the best action from Q-table
            q_values_for_state = self.q_table[current_node_id, :, :] # Adjusted Q-table access
            
            best_action_val = -float('inf')
            best_action = None
            
            for d_idx in range(NUM_DIRECTIONS):
                r, c = get_node_coords(current_node_id)
                next_r, next_c = r + DY[d_idx], c + DX[d_idx]
                if not (0 <= next_r < GRID_SIZE and 0 <= next_c < GRID_SIZE):
                    continue

                for s_idx in range(NUM_SPEEDING_LEVELS):
                    if q_values_for_state[d_idx, s_idx] > best_action_val:
                        best_action_val = q_values_for_state[d_idx, s_idx]
                        best_action = (d_idx, s_idx)
            
            if best_action is None and (q_values_for_state.size == 0 or np.all(np.isneginf(q_values_for_state))): # check if all are -inf or empty
                 valid_actions = []
                 for d_idx_rand in range(NUM_DIRECTIONS):
                    r_rand, c_rand = get_node_coords(current_node_id)
                    next_r_rand, next_c_rand = r_rand + DY[d_idx_rand], c_rand + DX[d_idx_rand]
                    if 0 <= next_r_rand < GRID_SIZE and 0 <= next_c_rand < GRID_SIZE:
                        for s_idx_rand in range(NUM_SPEEDING_LEVELS):
                            valid_actions.append((d_idx_rand, s_idx_rand))
                 if valid_actions: return random.choice(valid_actions)
            return best_action


    def train(self):
        print(f"Starting training for {NUM_EPISODES} episodes with Lagrangian method...")
        episode_lagrangian_rewards = []
        episode_actual_costs = [] 
        episode_actual_times = []
        lambda_history = []
        successful_episodes_count = 0 # Count episodes reaching END_NODE within MAX_STEPS
        
        start_train_time = datetime.now()

        for episode in range(NUM_EPISODES):
            current_node_id = START_NODE
            done = False
            
            current_episode_lagrangian_reward = 0
            current_episode_actual_total_cost = 0 # C2
            current_episode_actual_total_time = 0 # T_episode

            for step in range(MAX_STEPS_PER_EPISODE):
                action = self.choose_action(current_node_id)
                if action is None: 
                    # This case should ideally not be hit if MAX_STEPS_PER_EPISODE is generous
                    # and illegal moves are penalized but don't end episode immediately unless necessary.
                    # If stuck (no valid actions from choose_action), penalize and end step.
                    # For simplicity, we break here, but a penalty could be applied.
                    break 
                
                direction_idx, speeding_idx = action
                speeding_r = SPEEDING_RATIOS[speeding_idx]

                r, c = get_node_coords(current_node_id)
                next_r, next_c = r + DY[direction_idx], c + DX[direction_idx]

                lagrangian_reward_for_step = 0
                
                if not (0 <= next_r < GRID_SIZE and 0 <= next_c < GRID_SIZE):
                    lagrangian_reward_for_step = -HUGE_PENALTY_ILLEGAL_MOVE # Keep this high penalty
                    next_node_id = current_node_id # Stay in the same node
                    # done = True # Or let it try to recover, but penalize heavily
                else:
                    next_node_id = get_node_id(next_r, next_c)
                    segment_V_lim = get_speed_limit_on_segment(current_node_id, direction_idx, self.limits_row, self.limits_col)
                    
                    if segment_V_lim == -1: # Should not happen with boundary checks, but as safeguard
                        lagrangian_reward_for_step = -HUGE_PENALTY_ILLEGAL_MOVE
                        next_node_id = current_node_id
                    else:
                        cost_edge, time_taken_segment = calculate_c_edge(segment_V_lim, speeding_r)
                        
                        if time_taken_segment == float('inf'): # e.g. actual_speed_v <= 0
                            lagrangian_reward_for_step = -HUGE_PENALTY_ILLEGAL_MOVE # Or a very large cost
                            # next_node_id remains current_node_id or handle as stuck
                        else:
                            immediate_reward = -cost_edge # r_t
                            lagrangian_reward_for_step = immediate_reward - self.lambda_val * time_taken_segment
                            
                            current_episode_actual_total_cost += cost_edge
                            current_episode_actual_total_time += time_taken_segment

                        if next_node_id == END_NODE:
                            done = True
                
                # Q-value update
                old_q_value = self.q_table[current_node_id, direction_idx, speeding_idx]
                
                max_future_q = -float('inf') 
                if not done and next_node_id != -1:
                    has_valid_future_action = False
                    # Check all actions from next_node_id
                    for next_d_idx in range(NUM_DIRECTIONS):
                        n_r_check, n_c_check = get_node_coords(next_node_id)
                        n_next_r_check, n_next_c_check = n_r_check + DY[next_d_idx], n_c_check + DX[next_d_idx]
                        if 0 <= n_next_r_check < GRID_SIZE and 0 <= n_next_c_check < GRID_SIZE:
                            for next_s_idx in range(NUM_SPEEDING_LEVELS):
                                if self.q_table[next_node_id, next_d_idx, next_s_idx] > max_future_q:
                                    max_future_q = self.q_table[next_node_id, next_d_idx, next_s_idx]
                                    has_valid_future_action = True
                    if not has_valid_future_action: # If stuck or only invalid moves from next state
                        max_future_q = 0 
                elif done: 
                     max_future_q = 0 # No future reward if episode ends

                new_q_value = old_q_value + ALPHA * (lagrangian_reward_for_step + GAMMA * max_future_q - old_q_value)
                self.q_table[current_node_id, direction_idx, speeding_idx] = new_q_value
                
                current_episode_lagrangian_reward += lagrangian_reward_for_step
                
                if done:
                    break
                
                current_node_id = next_node_id
            
            # End of episode
            episode_lagrangian_rewards.append(current_episode_lagrangian_reward)
            if done and current_node_id == END_NODE : # Successfully reached the end
                successful_episodes_count +=1
                episode_actual_costs.append(current_episode_actual_total_cost)
                episode_actual_times.append(current_episode_actual_total_time)
            
            # Update lambda
            self.lambda_val = max(0, self.lambda_val + BETA_LAMBDA * (current_episode_actual_total_time - DEADLINE))
            lambda_history.append(self.lambda_val)

            self.epsilon = max(EPSILON_END, EPSILON_START - episode * EPSILON_DECAY_RATE)

            if (episode + 1) % 1000 == 0 or episode == NUM_EPISODES - 1:
                avg_lag_reward = np.mean(episode_lagrangian_rewards[-1000:]) if episode_lagrangian_rewards else 0
                # For successful episodes in the last 1000, or all if fewer than 1000 total successful
                recent_success_costs = [c for c, t in zip(episode_actual_costs[-(successful_episodes_count % 1001):], 
                                                          episode_actual_times[-(successful_episodes_count % 1001):]) if t <= DEADLINE + 1e-9] # allow small tolerance
                recent_success_times = [t for t in episode_actual_times[-(successful_episodes_count % 1001):] if t <= DEADLINE + 1e-9]

                avg_cost_success = np.mean(recent_success_costs) if recent_success_costs else 0
                avg_time_success = np.mean(recent_success_times) if recent_success_times else 0
                
                # Success rate: episodes that reached END_NODE (within MAX_STEPS) AND met DEADLINE in the last 1000 block
                # This is tricky because lambda update happens based on T_episode regardless of DEADLINE success for that episode
                # Let's report raw success (reaching END_NODE) and then path extraction will verify DEADLINE
                current_block_total_episodes = (episode % 1000) + 1 if (episode + 1) % 1000 != 0 else 1000
                block_successful_episodes = 0
                temp_episode_idx = episode - current_block_total_episodes + 1
                temp_success_count_block = 0 # count successful episodes in current block of 1000
                
                # Count successful episodes in the current block of 1000 for success rate
                # This logic needs to be careful about how successful_episodes_count is managed
                # A simpler success rate: (successful_episodes_count in this block / episodes in this block)
                # We need a running count of successes per block.
                # Let's use a simpler success rate for now: total successful / total episodes in block
                
                # Simplified success rate for logging:
                success_rate = sum(1 for t in episode_actual_times[-current_block_total_episodes:] 
                                  if t <= DEADLINE + 1e-9) / current_block_total_episodes * 100 if episode_actual_times else 0
                
                print(f"Ep {episode+1}/{NUM_EPISODES}, Eps: {self.epsilon:.3f}, AvgRew: {avg_lag_reward:.2f}, "
                      f"AvgCost(S): {avg_cost_success:.2f}, AvgTime(S): {avg_time_success:.3f}h, "
                      f"SuccRate(last 1k): {success_rate:.1f}%, Lambda: {self.lambda_val:.3f}")
        
        train_time = datetime.now() - start_train_time
        print(f"Training finished in {train_time}.")
        
        # Save Q-table
        np.save(Q_TABLE_SAVE_PATH, self.q_table)
        print(f"Q-table saved to {Q_TABLE_SAVE_PATH}")
        
        return episode_lagrangian_rewards, episode_actual_costs, episode_actual_times, lambda_history

    def extract_optimal_path(self):
        """Extract the optimal path from the trained Q-table."""
        print("\nExtracting optimal path (Route Three)...")
        
        current_node_id = START_NODE
        path = [current_node_id]
        actions = []
        total_cost = 0
        total_time = 0
        
        # Set a maximum number of steps to prevent infinite loops
        max_steps = NUM_NODES * 2
        step_count = 0
        
        while current_node_id != END_NODE and step_count < max_steps:
            # Choose the best action from the Q-table (greedy policy)
            best_action = None
            best_q_value = -float('inf')
            
            for d_idx in range(NUM_DIRECTIONS):
                r, c = get_node_coords(current_node_id)
                next_r, next_c = r + DY[d_idx], c + DX[d_idx]
                
                if not (0 <= next_r < GRID_SIZE and 0 <= next_c < GRID_SIZE):
                    continue
                
                for s_idx in range(NUM_SPEEDING_LEVELS):
                    q_value = self.q_table[current_node_id, d_idx, s_idx]
                    if q_value > best_q_value:
                        best_q_value = q_value
                        best_action = (d_idx, s_idx)
            
            if best_action is None:
                print("Path extraction failed: No valid action found.")
                return None, None, None
            
            direction_idx, speeding_idx = best_action
            speeding_r = SPEEDING_RATIOS[speeding_idx]
            
            # Get the next node
            r, c = get_node_coords(current_node_id)
            next_r, next_c = r + DY[direction_idx], c + DX[direction_idx]
            next_node_id = get_node_id(next_r, next_c)
            
            # Calculate cost and time for this step
            segment_V_lim = get_speed_limit_on_segment(current_node_id, direction_idx, self.limits_row, self.limits_col)
            cost_edge, time_taken_segment = calculate_c_edge(segment_V_lim, speeding_r)
            
            # Update path, actions, cost, and time
            path.append(next_node_id)
            actions.append((direction_idx, speeding_idx))
            total_cost += cost_edge
            total_time += time_taken_segment
            
            # Move to the next node
            current_node_id = next_node_id
            step_count += 1
        
        if current_node_id != END_NODE:
            print("Path extraction failed: Timeout before reaching destination.")
            return None, None, None
        
        # Check if the path meets the time constraint
        if total_time > DEADLINE:
            print(f"Path extraction warning: Found path exceeds deadline ({total_time:.3f}h > {DEADLINE:.3f}h)")
            # We'll still return the path, but with a warning
        
        return path, total_cost, total_time

def run_problem2():
    """Main function to run Problem 2."""
    # Load speed limits
    limits_row, limits_col = load_speed_limits()
    print("Speed limits loaded.\n")
    
    # Create and train the Q-learning agent
    agent = QLearningAgent(limits_row, limits_col)
    agent.train()
    
    # Extract the optimal path
    path, total_cost, total_time = agent.extract_optimal_path()
    
    # Write results to output file
    with open(OUTPUT_FILE_PATH, 'w', encoding='utf-8') as f:  # 添加 encoding='utf-8'
        f.write(f"Problem 2: Minimum Expected Cost C2 with Time Constraint (Deadline: {DEADLINE:.3f} hours)\n\n")
        f.write("Speed limits loaded.\n\n")
        
        if path is not None and total_time <= DEADLINE:
            f.write(f"Route Three (Minimum Expected Cost with Time Constraint):\n")
            f.write(f"Path: {path}\n")
            f.write(f"Total Expected Cost C2: {total_cost:.2f}\n")
            f.write(f"Total Expected Time T2: {total_time:.3f} hours\n")
            f.write(f"Time Constraint: {total_time:.3f} <= {DEADLINE:.3f} hours ✓\n")
        else:
            f.write("Could not find a valid path (Route Three) meeting the criteria.\n")
        
        f.write(f"\nResults successfully written to {OUTPUT_FILE_PATH}")
    
    print(f"\nResults successfully written to {OUTPUT_FILE_PATH}")

if __name__ == "__main__":
    run_problem2()
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import pandas as pd
import os
from datetime import datetime

# --- Constants and Configuration ---
# Grid and Node Constants
GRID_SIZE = 10
NUM_NODES = GRID_SIZE * GRID_SIZE
START_NODE = 0  # 甲地 (1号路口)
END_NODE = NUM_NODES - 1  # 乙地 (100号路口)
SEGMENT_DISTANCE_KM = 50.0

# Time Constants
T1_HOURS = 10.8333
T_DEAD_HOURS = 0.7 * T1_HOURS
T_DEAD_SECONDS = T_DEAD_HOURS * 3600.0
TIME_SCALE_FACTOR = 3600.0 # To convert seconds to hours for lambda updates, as per plan.md tuning guide (TIME_SCALE)
                           # If lambda explodes, this might need adjustment.
                           # Using seconds directly for lambda calculation. NOW CHANGED TO HOURS.
T_DEAD_SCALED = T_DEAD_SECONDS / TIME_SCALE_FACTOR


# Action Constants
# (dir ∈0‑3, r_idx ∈0‑3) -> 0:E, 1:N, 2:W, 3:S
# r_idx: 0:0%, 1:20%, 2:50%, 3:70% overspeed
DIRECTIONS = [(0, 1), (-1, 0), (0, -1), (1, 0)]  # (dr, dc) for E, N, W, S
# For problem: E, N, W, S. My array indexing: row_idx (N-S), col_idx (W-E)
# So, E: (0,1), N: (-1,0), W: (0,-1), S: (1,0) matches common (row,col) systems.
NUM_DIRECTIONS = 4
SPEEDING_RATIOS = [0.0, 0.2, 0.5, 0.7]
NUM_SPEEDING_LEVELS = len(SPEEDING_RATIOS)
ACTION_SPACE_SIZE = NUM_DIRECTIONS * NUM_SPEEDING_LEVELS # 16

# RL Agent Hyperparameters (from plan.md)
STATE_DIM = NUM_NODES + 2  # 100 (one-hot node) + 1 (norm_remaining_time) + 1 (is_overtime_flag)
HIDDEN_DIM = 128
GAMMA = 0.99  # Discount factor for rewards
GAE_LAMBDA = 0.95 # GAE lambda, not the Lagrangian multiplier
ALPHA_ENT = 0.01  # Entropy regularization coefficient
BETA_LAMBDA_INITIAL = 5e-3 # Learning rate for Lagrangian multiplier lambda. Increased from 5e-4
BETA_LAMBDA_DECAY_STEP = 10000 # Env steps after which beta_lambda decays
BETA_LAMBDA_DECAY_FACTOR = 0.3
ACTOR_LR = 1e-3
CRITIC_LR = 5e-3 # Plan suggests Critic LR can be higher, e.g., 5e-3. Let's use same as Actor for now or make it separate.
CRITIC_LR_ACTUAL = 5e-3 # Using same as actor for simplicity first.
GRADIENT_CLIP = 5.0
C_BIG_PENALTY = 10000.0 # Penalty for exceeding T_dead

# Training Rhythm (from plan.md)
MAX_EPISODE_LENGTH = 200
SAMPLING_BATCH_SIZE = 2560  # Steps per batch for update
OPTIMIZATION_ITERATIONS_PER_BATCH = 1 # Number of updates per batch
TOTAL_ENV_STEPS = 150 * 1000

WARMUP_STEPS_LAMBDA_ZERO = 20 * 1000 # First 20k steps, lambda = 0
WARMUP_STEPS_CRITIC_FROZEN = 0 # First 10k steps, critic is frozen. NOW SET TO 0.

# Paths
DATA_DIR = "d:/Code/mm/data"
LIMITS_ROW_FILE = os.path.join(DATA_DIR, "limits_row.csv")
LIMITS_COL_FILE = os.path.join(DATA_DIR, "limits_col.csv")

# Misc
PETROL_PRICE_PER_LITER = 7.76
NUM_MOBILE_RADARS = 20
TOTAL_SEGMENTS = (GRID_SIZE * (GRID_SIZE - 1)) + ((GRID_SIZE - 1) * GRID_SIZE) # 10*9 + 9*10 = 180
EXPECTED_MOBILE_RADARS_PER_SEGMENT = NUM_MOBILE_RADARS / TOTAL_SEGMENTS

# --- Helper Functions ---
def get_node_coords(node_id):
    return node_id // GRID_SIZE, node_id % GRID_SIZE

def get_node_id(r, c):
    if 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE:
        return r * GRID_SIZE + c
    return -1

limits_row_data = None
limits_col_data = None

def load_speed_limits():
    global limits_row_data, limits_col_data
    try:
        limits_row_data = pd.read_csv(LIMITS_ROW_FILE, header=None).values
        limits_col_data = pd.read_csv(LIMITS_COL_FILE, header=None).values
    except Exception as e:
        print(f"Error loading speed limit data: {e}")
        print("Please ensure 'limits_row.csv' and 'limits_col.csv' are in the 'd:/Code/mm/data' directory.")
        exit()

def get_speed_limit_on_segment(current_node_id, direction_idx):
    r, c = get_node_coords(current_node_id)
    dr, dc = DIRECTIONS[direction_idx]
    next_r, next_c = r + dr, c + dc

    if not (0 <= next_r < GRID_SIZE and 0 <= next_c < GRID_SIZE):
        return -1 # Invalid move

    # Horizontal movement (East or West)
    if dr == 0: # East (dc=1) or West (dc=-1)
        # limits_row_data is (10, 9), for segments between (r,c) and (r,c+1)
        # If moving East from (r,c) to (r,c+1), use limits_row_data[r, c]
        # If moving West from (r,c) to (r,c-1), use limits_row_data[r, c-1]
        col_idx_in_limit_file = min(c, next_c)
        if 0 <= col_idx_in_limit_file < limits_row_data.shape[1]:
             return limits_row_data[r, col_idx_in_limit_file]
        return -1 # Should not happen if grid logic is correct
    # Vertical movement (North or South)
    elif dc == 0: # North (dr=-1) or South (dr=1)
        # limits_col_data is (9, 10), for segments between (r,c) and (r+1,c)
        # If moving South from (r,c) to (r+1,c), use limits_col_data[r, c]
        # If moving North from (r,c) to (r-1,c), use limits_col_data[r-1, c]
        row_idx_in_limit_file = min(r, next_r)
        if 0 <= row_idx_in_limit_file < limits_col_data.shape[0]:
            return limits_col_data[row_idx_in_limit_file, c]
        return -1 # Should not happen
    return -1


def calculate_fine_amount(V_lim, r):
    if r == 0.0: return 0.0
    
    # Based on problem description's fine table
    if V_lim < 50:
        if r <= 0.2: return 50
        if r <= 0.5: return 100 # 0.2 < r <= 0.5
        if r <= 0.7: return 300 # 0.5 < r <= 0.7
        return 500 # r > 0.7 (not used by our discrete r)
    elif V_lim <= 80: # 50 <= V_lim <= 80
        if r <= 0.2: return 100
        if r <= 0.5: return 150
        if r <= 0.7: return 500
        return 1000
    elif V_lim <= 100: # 80 < V_lim <= 100
        if r <= 0.2: return 150
        if r <= 0.5: return 200
        if r <= 0.7: return 1000
        return 1500
    else: # V_lim > 100
        if r <= 0.5: return 200 # Covers r=0.2 and r=0.5
        if r <= 0.7: return 1500
        return 2000

def get_detection_prob_per_radar(r):
    if r == 0.0: return 0.0
    if abs(r - 0.2) < 1e-6 : return 0.70
    if abs(r - 0.5) < 1e-6 : return 0.90
    if abs(r - 0.7) < 1e-6 : return 0.99
    return 0.0 # Should not happen for discrete r

def calculate_c_edge_and_time(V_lim, speeding_r):
    if V_lim <= 0: # Should not happen for valid segments
        return float('inf'), float('inf')

    actual_speed_v = V_lim * (1 + speeding_r)
    if actual_speed_v <= 1e-6: # Avoid division by zero if speed is effectively zero
        return float('inf'), float('inf')

    time_hours = SEGMENT_DISTANCE_KM / actual_speed_v
    time_seconds = time_hours * 3600.0

    # 1. Fuel cost
    # V_liters_per_100km = 0.0625 * actual_speed_v + 1.875
    # fuel_liters_for_segment = (V_liters_per_100km / 100.0) * SEGMENT_DISTANCE_KM
    # fuel_cost = fuel_liters_for_segment * PETROL_PRICE_PER_LITER
    
    # Simpler fuel cost from problem statement: "每百公里耗油量V=0.0625v+1.875（升）"
    # This seems to be the formula for consumption rate, not total consumption for the segment.
    # Let's use the formula as given:
    consumption_rate_L_per_100km = 0.0625 * actual_speed_v + 1.875
    fuel_cost = (consumption_rate_L_per_100km / 100.0) * SEGMENT_DISTANCE_KM * PETROL_PRICE_PER_LITER


    # 2. Catering, accommodation,游览费用: c=20t（元）, t in hours
    catering_cost = 20.0 * time_hours

    # 3. Highway tolls (高速公路)
    toll_fee = 0.0
    if V_lim == 120: # Assuming only 120km/h roads are highways
        toll_fee = 0.5 * SEGMENT_DISTANCE_KM

    # 4. Expected speeding fine
    expected_fine = 0.0
    if speeding_r > 1e-6: # If overspeeding
        num_fixed_radars = 1 if V_lim >= 90 else 0
        # Total expected radars on this segment
        # This is a simplification. A more complex model might consider radar locations.
        total_expected_radars_on_segment = num_fixed_radars + EXPECTED_MOBILE_RADARS_PER_SEGMENT
        
        prob_detection_per_radar = get_detection_prob_per_radar(speeding_r)
        
        # Prob of NOT being detected by one radar
        prob_not_detected_per_radar = 1.0 - prob_detection_per_radar
        
        # Prob of NOT being detected by ANY of the expected radars
        # (1 - p_detect_indiv)^k_radars
        prob_not_detected_by_any = prob_not_detected_per_radar ** total_expected_radars_on_segment
        
        prob_detected_by_at_least_one = 1.0 - prob_not_detected_by_any
        
        fine_amount_if_caught = calculate_fine_amount(V_lim, speeding_r)
        expected_fine = prob_detected_by_at_least_one * fine_amount_if_caught

    total_cost_edge = fuel_cost + catering_cost + toll_fee + expected_fine
    return total_cost_edge, time_seconds

# --- Environment ---
class RoadEnv:
    def __init__(self):
        self.current_node_id = START_NODE
        self.time_used_seconds = 0.0
        load_speed_limits() # Load speed limit data

    def reset(self):
        self.current_node_id = START_NODE
        self.time_used_seconds = 0.0
        return self._get_state()

    def _get_state(self):
        node_one_hot = np.zeros(NUM_NODES, dtype=np.float32)
        if 0 <= self.current_node_id < NUM_NODES: # Handle potential intermediate invalid node_id before reset
             node_one_hot[self.current_node_id] = 1.0
        
        remaining_time_seconds = T_DEAD_SECONDS - self.time_used_seconds
        
        # τ = (T_dead - t_used) / T_dead, as per plan.md
        # If t_used > T_dead, τ will be negative. This is informative.
        normalized_remaining_time = remaining_time_seconds / T_DEAD_SECONDS
        
        is_overtime_flag = 1.0 if self.time_used_seconds > T_DEAD_SECONDS else 0.0
        
        state = np.concatenate((node_one_hot, [normalized_remaining_time], [is_overtime_flag]))
        return torch.FloatTensor(state)

    def step(self, action_idx):
        direction_idx = action_idx // NUM_SPEEDING_LEVELS
        speeding_level_idx = action_idx % NUM_SPEEDING_LEVELS
        speeding_r = SPEEDING_RATIOS[speeding_level_idx]

        r, c = get_node_coords(self.current_node_id)
        dr, dc = DIRECTIONS[direction_idx]
        next_r, next_c = r + dr, c + dc
        
        next_node_id = get_node_id(next_r, next_c)
        
        reward = 0
        done = False
        cost_edge = 0
        time_edge_seconds = 0
        constraint_violation = 0

        if next_node_id == -1: # Moved off grid
            reward = -C_BIG_PENALTY # Severe penalty for invalid move
            cost_edge = C_BIG_PENALTY # Assign a high cost
            time_edge_seconds = T_DEAD_SECONDS # Assign a high time to discourage
            done = True # End episode for invalid move
            # current_node_id remains, state will reflect this penalty in next _get_state if not done
            # but for simplicity, we make it done.
        else:
            V_lim = get_speed_limit_on_segment(self.current_node_id, direction_idx)
            if V_lim == -1: # Should not happen if logic is correct, but as safeguard
                reward = -C_BIG_PENALTY 
                cost_edge = C_BIG_PENALTY
                time_edge_seconds = T_DEAD_SECONDS
                done = True
            else:
                cost_edge, time_edge_seconds = calculate_c_edge_and_time(V_lim, speeding_r)

                if cost_edge == float('inf') or time_edge_seconds == float('inf'):
                    reward = -C_BIG_PENALTY
                    cost_edge = C_BIG_PENALTY # ensure it's finite for storage
                    time_edge_seconds = T_DEAD_SECONDS
                    done = True # Stuck or invalid calculation
                else:
                    self.time_used_seconds += time_edge_seconds
                    self.current_node_id = next_node_id
                    
                    reward = -cost_edge # Reward is negative of cost

                    if self.current_node_id == END_NODE:
                        done = True
                    
                    if self.time_used_seconds > T_DEAD_SECONDS:
                        done = True # Episode ends if overtime
                        # reward -= C_BIG_PENALTY # Additional penalty for overtime - REMOVED
                        constraint_violation = self.time_used_seconds - T_DEAD_SECONDS
        
        next_state = self._get_state()
        # time_edge_scaled for lambda calculation
        time_edge_scaled = time_edge_seconds / TIME_SCALE_FACTOR 
        
        info = {
            'cost_edge': cost_edge, 
            'time_edge_seconds': time_edge_seconds,
            'time_edge_scaled': time_edge_scaled, # For lambda updates and advantage
            'constraint_violation_seconds': constraint_violation / TIME_SCALE_FACTOR if constraint_violation > 0 else 0,
            'raw_constraint_violation_seconds': constraint_violation # For logging actual seconds
        }
        return next_state, reward, done, info

# --- Actor-Critic Network ---
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_space_size, hidden_dim):
        super(ActorCritic, self).__init__()
        self.shared_fc1 = nn.Linear(state_dim, hidden_dim)
        self.shared_fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        self.actor_head = nn.Linear(hidden_dim, action_space_size)
        self.critic_head = nn.Linear(hidden_dim, 1)

        self._initialize_weights()

    def _initialize_weights(self):
        # Orthogonal initialization with gain=sqrt(2) for ReLU layers
        nn.init.orthogonal_(self.shared_fc1.weight, gain=np.sqrt(2))
        nn.init.constant_(self.shared_fc1.bias, 0)
        nn.init.orthogonal_(self.shared_fc2.weight, gain=np.sqrt(2))
        nn.init.constant_(self.shared_fc2.bias, 0)
        
        # Actor logits layer gain 0.01 for exploration
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        nn.init.constant_(self.actor_head.bias, 0)
        
        # Critic head
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0) # Standard gain for value function
        nn.init.constant_(self.critic_head.bias, 0)

    def forward(self, state):
        x = F.relu(self.shared_fc1(state))
        x = F.relu(self.shared_fc2(x))
        
        action_logits = self.actor_head(x)
        state_value = self.critic_head(x)
        
        return action_logits, state_value

# --- Primal-Dual Actor-Critic Agent ---
class PrimalDualACAgent:
    def __init__(self, state_dim, action_space_size, hidden_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.ac_net = ActorCritic(state_dim, action_space_size, hidden_dim).to(self.device)
        self.actor_optimizer = optim.Adam(
            list(self.ac_net.shared_fc1.parameters()) + 
            list(self.ac_net.shared_fc2.parameters()) + 
            list(self.ac_net.actor_head.parameters()), 
            lr=ACTOR_LR
        )
        self.critic_optimizer = optim.Adam(
            list(self.ac_net.shared_fc1.parameters()) + # Critic also uses shared layers
            list(self.ac_net.shared_fc2.parameters()) +
            list(self.ac_net.critic_head.parameters()), 
            lr=CRITIC_LR_ACTUAL
        )

        self.lambda_lagrangian = 0.0 # Lagrangian multiplier, initialized to 0
        self.beta_lambda = BETA_LAMBDA_INITIAL # Learning rate for lambda

        self.global_env_steps = 0

    def select_action(self, state_tensor):
        state_tensor = state_tensor.to(self.device)
        action_logits, _ = self.ac_net(state_tensor)
        action_probs = F.softmax(action_logits, dim=-1)
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob, dist.entropy()

    def update_parameters(self, rollouts, current_lambda_val, critic_frozen=False):
        states, actions, rewards, log_probs_old, entropies, values, dones, next_values, time_edges_scaled = rollouts
        
        # Convert to tensors
        states = torch.stack(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).to(self.device).unsqueeze(1)
        log_probs_old = torch.stack(log_probs_old).to(self.device).unsqueeze(1)
        entropies = torch.stack(entropies).to(self.device).unsqueeze(1)
        values = torch.stack(values).to(self.device) # Detached from graph during collection
        dones = torch.FloatTensor(dones).to(self.device).unsqueeze(1)
        next_values = torch.FloatTensor(next_values).to(self.device).unsqueeze(1) # V(s_t+N)
        time_edges_scaled = torch.FloatTensor(time_edges_scaled).to(self.device).unsqueeze(1)

        # Calculate GAE for A_lambda_t = G^R_t - λ * G^T_t - V_phi(s_t)
        # Effective reward for GAE: r_eff_t = reward_t - current_lambda_val * time_edge_scaled_t
        effective_rewards = rewards - current_lambda_val * time_edges_scaled
        
        advantages = torch.zeros_like(rewards).to(self.device)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            delta = effective_rewards[t] + GAMMA * next_values[t] * (1-dones[t]) - values[t]
            gae = delta + GAMMA * GAE_LAMBDA * (1-dones[t]) * gae
            advantages[t] = gae
        
        # Critic targets: V_target = Advantage_lambda + V(s_t)
        # Or, more directly, V_target is the GAE computed on r_eff plus current value estimate
        # V_target = G^R_t - λ * G^T_t. The GAE is an estimate of this minus V(s_t).
        # So, V_target = advantages + values (where advantages are for r_eff)
        value_targets = advantages + values 
        
        # Actor Loss
        actor_loss = -(advantages.detach() * log_probs_old).mean() - ALPHA_ENT * entropies.mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.ac_net.actor_head.parameters(), GRADIENT_CLIP) # Clip only actor head or all?
        nn.utils.clip_grad_norm_(self.ac_net.shared_fc1.parameters(), GRADIENT_CLIP)
        nn.utils.clip_grad_norm_(self.ac_net.shared_fc2.parameters(), GRADIENT_CLIP)
        self.actor_optimizer.step()

        # Critic Loss
        if not critic_frozen:
            # Re-evaluate V(s_t) with current network parameters for critic loss
            _, current_state_values = self.ac_net(states)
            critic_loss = F.mse_loss(current_state_values, value_targets.detach())
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.ac_net.critic_head.parameters(), GRADIENT_CLIP)
            nn.utils.clip_grad_norm_(self.ac_net.shared_fc1.parameters(), GRADIENT_CLIP) # Also clip shared for critic
            nn.utils.clip_grad_norm_(self.ac_net.shared_fc2.parameters(), GRADIENT_CLIP)
            self.critic_optimizer.step()
        else:
            critic_loss = torch.tensor(0.0) # Dummy value

        return actor_loss.item(), critic_loss.item()


    def update_lambda_lagrangian(self, completed_episode_total_times_scaled, t_dead_scaled):
        if not completed_episode_total_times_scaled:
            return

        avg_traj_time_scaled = np.mean(completed_episode_total_times_scaled)
        # λ ← [λ + β_λ * (T_traj_avg_scaled - T_dead_scaled)]^+
        self.lambda_lagrangian = max(0, self.lambda_lagrangian + self.beta_lambda * (avg_traj_time_scaled - t_dead_scaled))

        # Decay beta_lambda
        if self.global_env_steps > BETA_LAMBDA_DECAY_STEP and self.beta_lambda == BETA_LAMBDA_INITIAL:
             self.beta_lambda *= BETA_LAMBDA_DECAY_FACTOR
             print(f"Decayed beta_lambda to {self.beta_lambda} at step {self.global_env_steps}")
    
    def get_lambda(self):
        if self.global_env_steps < WARMUP_STEPS_LAMBDA_ZERO:
            return 0.0 # Lambda is fixed to 0 during warmup
        return self.lambda_lagrangian

# --- Training Loop ---
def train():
    print("Starting training...")
    start_time = datetime.now()

    env = RoadEnv()
    agent = PrimalDualACAgent(STATE_DIM, ACTION_SPACE_SIZE, HIDDEN_DIM)

    # Logging
    all_episode_rewards = []
    all_episode_costs = []
    all_episode_times_seconds = []
    all_episode_times_scaled = []
    lambda_history = []
    
    num_batches = TOTAL_ENV_STEPS // SAMPLING_BATCH_SIZE

    for batch_num in range(num_batches):
        # --- Collect Rollouts ---
        batch_states, batch_actions, batch_rewards = [], [], []
        batch_log_probs, batch_entropies, batch_values, batch_dones = [], [], [], []
        batch_next_values = [] # Stores V(s_{t+1}) or V(s_t) if done
        batch_time_edges_scaled = []
        
        # For lambda update
        completed_episode_total_times_scaled = []
        completed_episode_total_costs = []
        completed_episode_total_rewards = []
        completed_episode_success = []
        
        # For current episode tracking
        ep_states, ep_actions, ep_rewards = [], [], []
        ep_log_probs, ep_entropies, ep_values, ep_dones = [], [], [], []
        ep_next_values = []
        ep_time_edges_scaled = []
        
        ep_total_time_scaled = 0.0
        ep_total_cost = 0.0
        ep_total_reward = 0.0
        ep_success = False
        
        state = env.reset()
        episode_step = 0
        
        # Collect SAMPLING_BATCH_SIZE steps
        steps_collected = 0
        episodes_completed = 0
        
        while steps_collected < SAMPLING_BATCH_SIZE:
            # Get action from policy
            action, log_prob, entropy = agent.select_action(state)
            
            # Get value estimate
            _, value_st = agent.ac_net(state.to(agent.device))
            value_st = value_st.detach().cpu()
            
            # Take action in environment
            next_state, reward, done, info = env.step(action)
            
            # Store step data
            ep_states.append(state)
            ep_actions.append(action)
            ep_rewards.append(reward)
            ep_log_probs.append(log_prob)
            ep_entropies.append(entropy)
            ep_values.append(value_st)
            ep_dones.append(done)
            ep_time_edges_scaled.append(info['time_edge_scaled'])
            
            # Update episode stats
            ep_total_time_scaled += info['time_edge_scaled']
            ep_total_cost += info['cost_edge']
            ep_total_reward += reward
            
            # Check if reached goal
            if env.current_node_id == END_NODE and not done:
                ep_success = True
            
            # Get next state value for bootstrapping
            if not done:
                _, next_value = agent.ac_net(next_state.to(agent.device))
                next_value = next_value.detach().cpu()
            else:
                next_value = torch.zeros(1)
            
            ep_next_values.append(next_value)
            
            # Move to next state
            state = next_state
            episode_step += 1
            steps_collected += 1
            agent.global_env_steps += 1
            
            # Episode termination
            if done or episode_step >= MAX_EPISODE_LENGTH:
                # Add episode data to batch
                batch_states.extend(ep_states)
                batch_actions.extend(ep_actions)
                batch_rewards.extend(ep_rewards)
                batch_log_probs.extend(ep_log_probs)
                batch_entropies.extend(ep_entropies)
                batch_values.extend(ep_values)
                batch_dones.extend(ep_dones)
                batch_next_values.extend(ep_next_values)
                batch_time_edges_scaled.extend(ep_time_edges_scaled)
                
                # Store completed episode stats for lambda update
                completed_episode_total_times_scaled.append(ep_total_time_scaled)
                completed_episode_total_costs.append(ep_total_cost)
                completed_episode_total_rewards.append(ep_total_reward)
                completed_episode_success.append(ep_success)
                
                # Log episode stats
                all_episode_rewards.append(ep_total_reward)
                all_episode_costs.append(ep_total_cost)
                all_episode_times_seconds.append(ep_total_time_scaled * TIME_SCALE_FACTOR)
                all_episode_times_scaled.append(ep_total_time_scaled)
                
                # Reset for next episode
                state = env.reset()
                episode_step = 0
                episodes_completed += 1
                
                # Reset episode tracking
                ep_states, ep_actions, ep_rewards = [], [], []
                ep_log_probs, ep_entropies, ep_values, ep_dones = [], [], [], []
                ep_next_values = []
                ep_time_edges_scaled = []
                
                ep_total_time_scaled = 0.0
                ep_total_cost = 0.0
                ep_total_reward = 0.0
                ep_success = False
        
        # --- Update Policy and Value Function ---
        # Get current lambda value for advantage calculation
        current_lambda_val = agent.get_lambda()
        lambda_history.append(current_lambda_val)
        
        # Critic frozen during warmup
        critic_frozen = agent.global_env_steps < WARMUP_STEPS_CRITIC_FROZEN
        
        # Prepare rollouts for update
        rollouts = (
            batch_states, batch_actions, batch_rewards, 
            batch_log_probs, batch_entropies, batch_values, 
            batch_dones, batch_next_values, batch_time_edges_scaled
        )
        
        # Update actor and critic
        actor_loss, critic_loss = agent.update_parameters(
            rollouts, current_lambda_val, critic_frozen
        )
        
        # Update lambda (Lagrangian multiplier)
        if agent.global_env_steps >= WARMUP_STEPS_LAMBDA_ZERO:
            agent.update_lambda_lagrangian(completed_episode_total_times_scaled, T_DEAD_SCALED)
        
        # --- Logging ---
        if batch_num % 10 == 0 or batch_num == num_batches - 1:
            avg_reward = np.mean(all_episode_rewards[-episodes_completed:]) if episodes_completed > 0 else 0
            avg_cost = np.mean(all_episode_costs[-episodes_completed:]) if episodes_completed > 0 else 0
            avg_time = np.mean(all_episode_times_seconds[-episodes_completed:]) if episodes_completed > 0 else 0
            success_rate = np.mean(completed_episode_success) if completed_episode_success else 0
            
            print(f"Batch {batch_num}/{num_batches} | " +
                  f"Steps: {agent.global_env_steps} | " +
                  f"Avg Reward: {avg_reward:.2f} | " +
                  f"Avg Cost: {avg_cost:.2f} | " +
                  f"Avg Time: {avg_time:.2f}s | " +
                  f"Success Rate: {success_rate:.2f} | " +
                  f"Lambda: {current_lambda_val:.4f} | " +
                  f"Actor Loss: {actor_loss:.4f} | " +
                  f"Critic Loss: {critic_loss:.4f}")
    
    # Training complete
    end_time = datetime.now()
    training_duration = end_time - start_time
    print(f"Training completed in {training_duration}")
    
    # Save training history
    training_data = {
        'rewards': all_episode_rewards,
        'costs': all_episode_costs,
        'times': all_episode_times_seconds,
        'lambda_history': lambda_history
    }
    
    return agent, training_data

# --- Route Extraction ---
def extract_route3(agent, temperature=1.0):
    """
    Extract route 3 using the trained policy with greedy decoding.
    
    Args:
        agent: Trained PrimalDualACAgent
        temperature: Temperature for softmax sampling (lower = more greedy)
    
    Returns:
        Dictionary with route information
    """
    env = RoadEnv()
    state = env.reset()
    
    node_seq = [START_NODE]
    r_percent_seq = []
    cost_edges = []
    time_edges = []
    
    done = False
    total_cost = 0
    total_time_seconds = 0
    total_travel_cost = 0
    total_fine_cost = 0
    
    while not done:
        # Get action logits
        state_tensor = state.to(agent.device)
        action_logits, _ = agent.ac_net(state_tensor)
        
        # Apply temperature and get probabilities
        scaled_logits = action_logits / temperature
        action_probs = F.softmax(scaled_logits, dim=-1)
        
        # Greedy action selection
        action = torch.argmax(action_probs).item()
        
        # Take action
        next_state, reward, done, info = env.step(action)
        
        # Extract direction and speeding ratio
        direction_idx = action // NUM_SPEEDING_LEVELS
        speeding_level_idx = action % NUM_SPEEDING_LEVELS
        speeding_r = SPEEDING_RATIOS[speeding_level_idx]
        
        # Update node sequence
        if env.current_node_id != -1:  # Valid move
            node_seq.append(env.current_node_id)
            r_percent_seq.append(speeding_r)
            cost_edges.append(info['cost_edge'])
            time_edges.append(info['time_edge_seconds'])
            
            total_cost += info['cost_edge']
            total_time_seconds += info['time_edge_seconds']
            
            # Calculate travel cost vs fine cost
            r, c = get_node_coords(node_seq[-2])  # Previous node
            dr, dc = DIRECTIONS[direction_idx]
            V_lim = get_speed_limit_on_segment(node_seq[-2], direction_idx)
            
            if V_lim > 0:
                actual_speed = V_lim * (1 + speeding_r)
                
                # Calculate fuel cost
                consumption_rate = 0.0625 * actual_speed + 1.875
                fuel_cost = (consumption_rate / 100.0) * SEGMENT_DISTANCE_KM * PETROL_PRICE_PER_LITER
                
                # Calculate time cost
                time_hours = SEGMENT_DISTANCE_KM / actual_speed
                time_cost = 20.0 * time_hours
                
                # Calculate toll fee
                toll_fee = 0.5 * SEGMENT_DISTANCE_KM if V_lim == 120 else 0
                
                # Travel cost for this edge
                edge_travel_cost = fuel_cost + time_cost + toll_fee
                total_travel_cost += edge_travel_cost
                
                # Fine cost is the remainder
                edge_fine_cost = info['cost_edge'] - edge_travel_cost
                total_fine_cost += edge_fine_cost
        
        # Move to next state
        state = next_state
        
        # Check if we've reached the goal or exceeded time limit
        if env.current_node_id == END_NODE or total_time_seconds > T_DEAD_SECONDS:
            done = True
    
    # Check if we need to try again with lower temperature
    if total_time_seconds > 0.7 * T1_HOURS * 3600 and temperature > 0.1:
        print(f"Route exceeds time limit ({total_time_seconds/3600:.2f}h > {0.7*T1_HOURS:.2f}h), retrying with lower temperature")
        return extract_route3(agent, temperature=0.5)
    
    # Calculate T3 in hours
    T3 = total_time_seconds / 3600.0
    
    route_info = {
        'node_seq': node_seq,
        'r_percent_seq': r_percent_seq,
        'cost_edges': cost_edges,
        'time_edges': time_edges,
        'T3': T3,
        'C_travel': total_travel_cost,
        'E_fine': total_fine_cost,
        'C2': total_cost,
        'success': env.current_node_id == END_NODE and T3 <= 0.7 * T1_HOURS
    }
    
    return route_info

# --- Main Function ---
def main():
    # Train the agent
    agent, training_data = train()
    
    # Extract route 3
    route_info = extract_route3(agent)
    
    # Print results
    print("\n=== Route 3 Results ===")
    print(f"Total nodes visited: {len(route_info['node_seq'])}")
    print(f"Path: {route_info['node_seq']}")
    print(f"T3 (hours): {route_info['T3']:.4f}")
    print(f"C_travel: {route_info['C_travel']:.2f}")
    print(f"E_fine: {route_info['E_fine']:.2f}")
    print(f"C2: {route_info['C2']:.2f}")
    print(f"Success: {route_info['success']}")
    
    # Print speeding plan
    print("\n=== Speeding Plan ===")
    print("Segment\tFrom\tTo\tSpeed Limit\tSpeeding %\tActual Speed")
    
    for i in range(len(route_info['node_seq']) - 1):
        from_node = route_info['node_seq'][i]
        to_node = route_info['node_seq'][i + 1]
        
        # Determine direction
        from_r, from_c = get_node_coords(from_node)
        to_r, to_c = get_node_coords(to_node)
        
        dr = to_r - from_r
        dc = to_c - from_c
        
        direction = None
        for idx, (dir_dr, dir_dc) in enumerate(DIRECTIONS):
            if dir_dr == dr and dir_dc == dc:
                direction = idx
                break
        
        if direction is not None:
            speed_limit = get_speed_limit_on_segment(from_node, direction)
            speeding_r = route_info['r_percent_seq'][i]
            actual_speed = speed_limit * (1 + speeding_r)
            
            print(f"{i+1}\t{from_node+1}\t{to_node+1}\t{speed_limit}\t{speeding_r*100:.0f}%\t{actual_speed:.1f}")
    
    # Save results to file
    results = {
        'route': route_info,
        'training': training_data
    }
    
    # Save as pickle for easy loading later
    import pickle
    with open('route3_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print("\nResults saved to 'route3_results.pkl'")

if __name__ == "__main__":
    main()
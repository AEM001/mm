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
                           # 修正：使用小时作为时间尺度，避免lambda爆炸
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
ALPHA_ENT = 0.03  # 修正：增加初始熵系数，鼓励更多探索 (从0.01提高到0.03)
BETA_LAMBDA_INITIAL = 1e-2 # 修正：稍微提高lambda学习率，更快施加约束压力 (从5e-3提高到1e-2)
BETA_LAMBDA_DECAY_STEP = 100000 # Env steps after which beta_lambda decays. Increased from 50000
BETA_LAMBDA_DECAY_FACTOR = 0.3
ACTOR_LR = 1e-3
CRITIC_LR = 5e-3 # Plan suggests Critic LR can be higher, e.g., 5e-3. Let's use same as Actor for now or make it separate.
CRITIC_LR_ACTUAL = 5e-4 # Using same as actor for simplicity first. TRY REDUCING (e.g., from 1e-3 to 5e-4 or 1e-4)
GRADIENT_CLIP = 5.0
C_BIG_PENALTY = 200.0 # 修正：降低超时惩罚 (从3000降至1000，再降至200)
REACH_GOAL_REWARD = 20000.0 # 修正：大幅增加到达终点奖励 (从5000提高到20000)
C_OVERTIME_STEP_PENALTY = 50.0 # 新增：步进式超时惩罚

# Training Rhythm (from plan.md)
MAX_EPISODE_LENGTH = 200
SAMPLING_BATCH_SIZE = 2560  # Steps per batch for update
OPTIMIZATION_ITERATIONS_PER_BATCH = 1 # Number of updates per batch
TOTAL_ENV_STEPS = 150 * 1000

WARMUP_STEPS_LAMBDA_ZERO = 1000 # 修正：缩短Warmup阶段 (从5000提高到10000，再降至1000)
WARMUP_STEPS_CRITIC_FROZEN = 1000 # 修正：缩短Warmup阶段 (从5000提高到10000，再降至1000)

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

    # 1. Fuel cost - 修正燃油消耗计算
    # 每百公里耗油量V=0.0625v+1.875（升）
    consumption_rate_L_per_100km = 0.0625 * actual_speed_v + 1.875
    # 计算该路段的实际油耗（升）- 修正计算方式
    fuel_liters = (consumption_rate_L_per_100km / 100.0) * SEGMENT_DISTANCE_KM
    # 计算燃油成本
    fuel_cost = fuel_liters * PETROL_PRICE_PER_LITER


    # 2. Catering, accommodation,游览费用: c=20t（元）, t in hours
    catering_cost = 20.0 * time_hours

    # 3. Highway tolls (高速公路)
    toll_fee = 0.0
    if V_lim == 120: # Assuming only 120km/h roads are highways
        toll_fee = 0.5 * SEGMENT_DISTANCE_KM

    # 4. Expected speeding fine
    expected_fine = 0.0
    if speeding_r > 1e-6: # If overspeeding
        # 修正：简化雷达检测计算，对有固定雷达的段设k=2、无固定设k=1
        num_radars = 2 if V_lim >= 90 else 1
        
        prob_detection_per_radar = get_detection_prob_per_radar(speeding_r)
        
        # Prob of NOT being detected by one radar
        prob_not_detected_per_radar = 1.0 - prob_detection_per_radar
        
        # Prob of NOT being detected by ANY of the radars
        # (1 - p_detect_indiv)^k_radars
        prob_not_detected_by_any = prob_not_detected_per_radar ** num_radars
        
        prob_detected_by_at_least_one = 1.0 - prob_not_detected_by_any
        
        fine_amount_if_caught = calculate_fine_amount(V_lim, speeding_r)
        expected_fine = prob_detected_by_at_least_one * fine_amount_if_caught

    total_cost_edge = fuel_cost + catering_cost + toll_fee + expected_fine
    return total_cost_edge, time_seconds

# --- Environment ---
# 全局变量，用于跟踪是否处于warmup阶段
global_env_steps = 0

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
                    # 计算到终点的曼哈顿距离并给予奖励 (在更新节点前计算旧距离)
                    old_r, old_c = get_node_coords(self.current_node_id)
                    end_r, end_c = get_node_coords(END_NODE)
                    old_dist = abs(old_r - end_r) + abs(old_c - end_c)

                    self.time_used_seconds += time_edge_seconds
                    self.current_node_id = next_node_id

                    # 计算新距离 (在更新节点后计算新距离)
                    new_r, new_c = get_node_coords(self.current_node_id)
                    new_dist = abs(new_r - end_r) + abs(new_c - end_c)

                    # 如果更靠近终点，给予正奖励；否则给予负奖励
                    if new_dist < old_dist:
                        reward += 10.0 # 靠近终点奖励
                    elif new_dist > old_dist:
                        reward -= 10.0 # 远离终点惩罚
                    # 如果距离不变（比如撞墙或原地打转），不额外增减奖励

                    reward = -cost_edge # Reward is negative of cost

                    # 新增：步进式超时惩罚
                    if self.time_used_seconds > T_DEAD_SECONDS:
                        reward -= C_OVERTIME_STEP_PENALTY

                    if self.current_node_id == END_NODE:
                        done = True
                        reward += REACH_GOAL_REWARD # 添加到达终点的奖励

                    # 最终超时判断和惩罚 (只有在回合结束时应用 C_BIG_PENALTY)
                    if self.time_used_seconds > T_DEAD_SECONDS:
                        # 如果回合结束 (到达终点或达到最大步数) 且超时
                        if done:
                            # 在warmup阶段不施加最终超时惩罚
                            global global_env_steps
                            if global_env_steps >= WARMUP_STEPS_LAMBDA_ZERO:
                                reward -= C_BIG_PENALTY
                            constraint_violation = self.time_used_seconds - T_DEAD_SECONDS
                        else:
                             # 如果未结束但已超时，只记录违规时间，不立即结束回合
                             constraint_violation = self.time_used_seconds - T_DEAD_SECONDS
                             # done 保持 False，回合继续

        next_state = self._get_state()
        # time_edge_scaled for lambda calculation - 修正：使用小时作为时间尺度
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
        
        # 使用单个优化器处理所有参数，避免共享层被两次更新
        self.optimizer = optim.Adam(self.ac_net.parameters(), lr=ACTOR_LR)

        self.lambda_lagrangian = 0.0 # Lagrangian multiplier, initialized to 0
        self.beta_lambda = BETA_LAMBDA_INITIAL # Learning rate for lambda
        self.lambda_update_buffer = [] # 用于批量更新lambda的缓冲区

        # 熵系数自退火
        self.alpha_ent_initial = ALPHA_ENT # 使用上面修改后的0.03
        self.alpha_ent_final = 0.003  # 最终熵系数值保持不变

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
            # 修复GAE计算，确保done状态下不考虑下一状态的值
            delta = effective_rewards[t] + GAMMA * next_values[t] * (1-dones[t]) - values[t]
            gae = delta + GAMMA * GAE_LAMBDA * (1-dones[t]) * gae
            advantages[t] = gae
        
        # Critic targets: V_target = Advantage_lambda + V(s_t)
        # Or, more directly, V_target is the GAE computed on r_eff plus current value estimate
        # V_target = G^R_t - λ * G^T_t. The GAE is an estimate of this minus V(s_t).
        # So, V_target = advantages + values (where advantages are for r_eff)
        value_targets = advantages + values 
        
        # 使用单个优化器处理Actor和Critic的梯度，避免共享层参数更新冲突
        self.optimizer.zero_grad()
        
        # Actor Loss - 修复：重新前向计算获取新的log_probs，而不是使用断梯度的log_probs_old
        logits, _ = self.ac_net(states)
        dist = Categorical(F.softmax(logits, dim=-1))
        log_probs_new = dist.log_prob(actions.squeeze())
        
        # 使用自退火的熵系数
        current_alpha_ent = self.get_entropy_coef()
        
        # Critic Loss
        if not critic_frozen:
            # Re-evaluate V(s_t) with current network parameters for critic loss
            _, current_state_values = self.ac_net(states)
            critic_loss = F.mse_loss(current_state_values, value_targets.detach())
        else:
            critic_loss = torch.tensor(0.0) # Dummy value
            
        # 合并Actor和Critic的损失，一次性计算梯度
        actor_loss = -(advantages.detach() * log_probs_new).mean() - current_alpha_ent * dist.entropy().mean()
        total_loss = actor_loss + critic_loss
        
        # 一次性计算梯度并更新
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.ac_net.parameters(), GRADIENT_CLIP)
        self.optimizer.step()

        return actor_loss.item(), critic_loss.item()


    # 修正：修改函数签名，接收包含所有已结束回合时间的缓冲区
    def update_lambda_lagrangian(self, completed_episode_times_for_lambda_update, t_dead_scaled):
        # 修正：确保至少有20个完整轨迹才更新lambda，避免早期不稳定更新
        if not completed_episode_times_for_lambda_update:
            return

        # 使用折扣后的时间均值，确保与Critic/Advantage使用的时间口径一致
        # 只考虑批次内的平均轨迹时间，而不是累加所有时间
        avg_traj_time_scaled = np.mean(completed_episode_times_for_lambda_update)

        # 修正：使用更小的学习率(beta_lambda)更新lambda，并使用Softplus函数保持稳定性
        # Softplus: ln(1+e^x)
        # 修正：直接更新lambda_lagrangian，Softplus在get_lambda中应用，确保lambda始终非负
        lambda_raw = self.lambda_lagrangian + self.beta_lambda * (avg_traj_time_scaled - t_dead_scaled)
        # 修正：直接更新原始lambda值，Softplus在get_lambda中处理
        self.lambda_lagrangian = lambda_raw # 允许lambda_lagrangian为负，Softplus在get_lambda中处理非负约束

        # Decay beta_lambda
        if self.global_env_steps > BETA_LAMBDA_DECAY_STEP and self.beta_lambda == BETA_LAMBDA_INITIAL:
             self.beta_lambda *= BETA_LAMBDA_DECAY_FACTOR
             print(f"Decayed beta_lambda to {self.beta_lambda} at step {self.global_env_steps}")

    def get_lambda(self):
        if self.global_env_steps < WARMUP_STEPS_LAMBDA_ZERO:
            return 0.0 # Lambda is fixed to 0 during warmup
        return self.lambda_lagrangian
        
    def get_entropy_coef(self):
        # 线性退火：从初始值逐渐降低到最终值
        if self.global_env_steps >= TOTAL_ENV_STEPS:
            return self.alpha_ent_final
        
        # 线性插值
        # 修正：确保退火过程覆盖整个训练周期
        progress = min(1.0, self.global_env_steps / TOTAL_ENV_STEPS)
        return self.alpha_ent_initial + (self.alpha_ent_final - self.alpha_ent_initial) * progress

# --- Training Loop ---
def train():
    print("Starting training...")
    start_time = datetime.now()

    env = RoadEnv()
    agent = PrimalDualACAgent(STATE_DIM, ACTION_SPACE_SIZE, HIDDEN_DIM)

    # Load agent if checkpoint exists
    checkpoint_path = "primal_dual_ac_checkpoint.pth"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=agent.device)
        agent.ac_net.load_state_dict(checkpoint['model_state_dict'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        agent.lambda_lagrangian = checkpoint['lambda_lagrangian']
        agent.beta_lambda = checkpoint['beta_lambda']
        agent.global_env_steps = checkpoint['global_env_steps']
        print(f"Loaded checkpoint from {checkpoint_path} at step {agent.global_env_steps}")

    rollouts = []
    episode_rewards = []
    episode_costs = []
    episode_times_seconds = []
    episode_times_scaled = [] # For lambda updates
    episode_constraint_violations = []
    episode_successes = [] # Track if episode reached goal
    
    # 修正：使用新的缓冲区收集所有已结束回合的时间，无论是否成功
    completed_episode_times_for_lambda_update = []

    state = env.reset()
    episode_reward = 0
    episode_cost = 0
    episode_time_seconds = 0
    episode_time_scaled = 0
    episode_constraint_violation = 0
    episode_steps = 0
    current_episode_start_step = agent.global_env_steps

    # 修正：确保训练循环运行到 TOTAL_ENV_STEPS，移除5万步实验的临时限制
    while agent.global_env_steps < TOTAL_ENV_STEPS:
        action, log_prob, entropy = agent.select_action(state)
        next_state, reward, done, info = env.step(action)

        # Update global step count
        global global_env_steps
        global_env_steps = agent.global_env_steps # Ensure global variable is updated
        agent.global_env_steps += 1

        rollouts.append((state, action, reward, log_prob, entropy, agent.ac_net(state.to(agent.device))[1].detach(), done, agent.ac_net(next_state.to(agent.device))[1].detach(), info['time_edge_scaled']))

        episode_reward += reward
        episode_cost += info['cost_edge']
        episode_time_seconds += info['time_edge_seconds']
        episode_time_scaled += info['time_edge_scaled']
        episode_constraint_violation += info['constraint_violation_seconds']
        episode_steps += 1

        state = next_state

        if done or episode_steps >= MAX_EPISODE_LENGTH:
            # Collect episode stats
            episode_rewards.append(episode_reward)
            episode_costs.append(episode_cost)
            episode_times_seconds.append(episode_time_seconds)
            episode_times_scaled.append(episode_time_scaled)
            episode_constraint_violations.append(episode_constraint_violation)

            reached_goal = (env.current_node_id == END_NODE)
            overtime = (env.time_used_seconds > T_DEAD_SECONDS)
            episode_successes.append(reached_goal and not overtime)

            # 修正：将所有已结束回合的时间添加到新的缓冲区
            completed_episode_times_for_lambda_update.append(episode_time_scaled)

            # Log episode stats and update lambda if batch is ready
            if (agent.global_env_steps - current_episode_start_step) >= SAMPLING_BATCH_SIZE or agent.global_env_steps >= TOTAL_ENV_STEPS:
                 avg_reward = np.mean(episode_rewards) if episode_rewards else 0
                 avg_cost = np.mean(episode_costs) if episode_costs else 0
                 avg_time_seconds = np.mean(episode_times_seconds) if episode_times_seconds else 0
                 avg_time_scaled = np.mean(episode_times_scaled) if episode_times_scaled else 0
                 avg_constraint = np.mean(episode_constraint_violations) if episode_constraint_violations else 0
                 success_rate = np.mean(episode_successes) if episode_successes else 0

                 print(f"Step {agent.global_env_steps} | AvgReward {avg_reward:.2f} | AvgCost {avg_cost:.2f} | AvgTime {avg_time_seconds:.2f}s ({avg_time_scaled:.2f}h) | AvgConstraint {avg_constraint:.2f}h | SuccRate {success_rate:.2f} | Lambda {agent.lambda_lagrangian:.2f} | BetaLambda {agent.beta_lambda:.5f}")

                 # 修正：使用新的缓冲区更新lambda，并移除最小轨迹数限制
                 agent.update_lambda_lagrangian(completed_episode_times_for_lambda_update, T_DEAD_SCALED)

                 # Clear buffers for next batch
                 episode_rewards = []
                 episode_costs = []
                 episode_times_seconds = []
                 episode_times_scaled = []
                 episode_constraint_violations = []
                 episode_successes = []
                 # 修正：清空新的缓冲区
                 completed_episode_times_for_lambda_update = []
                 current_episode_start_step = agent.global_env_steps


            # Perform PPO update if enough steps collected
            if len(rollouts) >= SAMPLING_BATCH_SIZE or agent.global_env_steps >= TOTAL_ENV_STEPS:
                # Convert rollouts to tensors and update
                states, actions, rewards, log_probs_old, entropies, values, dones, next_values, time_edges_scaled = zip(*rollouts)

                # Calculate V_next for the last state in the batch if not done
                # This is already handled by collecting next_values in the loop

                # Update parameters
                critic_frozen = agent.global_env_steps < WARMUP_STEPS_CRITIC_FROZEN
                actor_loss, critic_loss = agent.update_parameters(
                    (list(states), list(actions), list(rewards), list(log_probs_old), list(entropies), list(values), list(dones), list(next_values), list(time_edges_scaled)),
                    agent.get_lambda(),
                    critic_frozen=critic_frozen
                )
                print(f"Step {agent.global_env_steps} | Actor Loss: {actor_loss:.4f} | Critic Loss: {critic_loss:.4f}")

                # Clear rollouts buffer
                rollouts = []

                # Save checkpoint periodically
                if agent.global_env_steps % 10000 == 0:
                    torch.save({
                        'global_env_steps': agent.global_env_steps,
                        'model_state_dict': agent.ac_net.state_dict(),
                        'optimizer_state_dict': agent.optimizer.state_dict(),
                        'lambda_lagrangian': agent.lambda_lagrangian,
                        'beta_lambda': agent.beta_lambda,
                    }, checkpoint_path)
                    print(f"Checkpoint saved at step {agent.global_env_steps}")


            # Reset environment for new episode
            state = env.reset()
            episode_reward = 0
            episode_cost = 0
            episode_time_seconds = 0
            episode_time_scaled = 0
            episode_constraint_violation = 0
            episode_steps = 0

    print("Training finished.")

# --- Main Execution ---
if __name__ == "__main__":
    # 修正：移除5万步实验的选项，直接运行训练
    train()
    
    # 如果需要提取路径，可以在训练完成后手动调用 extract_route3
    # print("\nExtracting final route...")
    # final_route_nodes, final_route_actions, final_route_times, final_route_costs, final_route_success = extract_route3(agent.ac_net)
    # print(f"Final Route Success: {final_route_success}")
    # print(f"Final Route Time: {sum(final_route_times)/3600:.2f} hours")
    # print(f"Final Route Cost: {sum(final_route_costs):.2f} yuan")
    # print("Final Route Nodes:", final_route_nodes)
    # print("Final Route Actions:", final_route_actions)

def extract_route3(agent, temperature=1.0, max_steps=50):
    """
    Extract route 3 using the trained policy with greedy decoding.
    
    Args:
        agent: Trained PrimalDualACAgent
        temperature: Temperature for softmax sampling (lower = more greedy)
        max_steps: Maximum steps to prevent infinite loops
    
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
    step_count = 0
    
    while not done and step_count < max_steps:
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
        step_count += 1
        
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
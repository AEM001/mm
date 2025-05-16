import pandas as pd
import numpy as np
from heapq import heappush, heappop
from collections import defaultdict
import os

def solve_part2():
    # --- Configuration ---
    # Script assumes it's located in d:/Code/mm/3/
    # Data is expected in d:/Code/mm/data/
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        # Fallback for interactive environments like Jupyter
        script_dir = os.getcwd()
    data_dir = os.path.join(script_dir, "..", "data")
    edges_file_path = os.path.join(data_dir, "edge_bracket_tc.csv")

    # Route one (0-indexed as per plan.md for Python usage)
    route1_nodes = [0, 1, 11, 12, 13, 23, 24, 34, 44, 54, 55, 56, 57, 58, 68, 78, 88, 89, 99]

    # --- Load and Process Data ---
    print(f"Loading data from: {edges_file_path}")
    try:
        # Assumed column names based on common problem descriptions.
        # Adjust ['eid', 'u', 'v', 'k', 't', 'c'] if your CSV uses different names.
        raw_data_df = pd.read_csv(edges_file_path)
    except FileNotFoundError:
        print(f"Error: Could not find data file at {edges_file_path}")
        print("Please ensure 'edge_bracket_tc.csv' exists in the 'data' directory.")
        return
    except Exception as e:
        print(f"Error loading data file: {e}")
        print("Please ensure the CSV file is correctly formatted.")
        return

    # Verify required columns exist
    # These are critical assumptions. Modify if your CSV headers are different.
    expected_cols = ['eid', 'u', 'v', 'k', 't', 'c'] 
    if not all(col in raw_data_df.columns for col in expected_cols):
        print(f"Error: CSV file must contain columns: {expected_cols}")
        print(f"Found columns: {raw_data_df.columns.tolist()}")
        print("Please check and adjust expected_cols in the script if names differ.")
        return

    print("Data loaded successfully. Processing...")

    # Create edges_df (eid, u, v)
    # Assumes 'eid' in the CSV is the unique edge identifier (0 to E-1, sequential).
    # Assumes 'u', 'v' in the CSV are 0-indexed node identifiers, consistent with route1_nodes.
    # If 'u', 'v' are 1-indexed in CSV, they need to be converted (e.g., row.u - 1).
    edges_df = raw_data_df[['eid', 'u', 'v']].drop_duplicates().sort_values('eid').reset_index(drop=True)

    if edges_df.empty:
        print("Error: No edges found after processing. Check 'eid', 'u', 'v' columns.")
        return
        
    num_edges = edges_df['eid'].max() + 1 
    if raw_data_df['k'].empty:
        print("Error: 'k' column is empty or not found, cannot determine number of brackets.")
        return
    num_brackets = raw_data_df['k'].max() + 1 # Assumes k is 0-indexed (e.g., 0,1,2,3 for 4 brackets)

    # Initialize t_mat and c_mat
    # t_mat stores time, c_mat stores cost
    # Dimensions: (num_edges, num_brackets)
    t_mat = np.full((num_edges, num_brackets), np.inf)
    c_mat = np.full((num_edges, num_brackets), np.inf)

    for _, row in raw_data_df.iterrows():
        eid = int(row['eid'])
        k_idx = int(row['k'])
        if eid >= num_edges or k_idx >= num_brackets:
            print(f"Warning: eid {eid} or k {k_idx} out of bounds. Max eid: {num_edges-1}, Max k: {num_brackets-1}. Skipping row: {row}")
            continue
        t_mat[eid, k_idx] = row['t']
        c_mat[eid, k_idx] = row['c']
    
    print("t_mat and c_mat constructed.")

    # --- Part 2: Calculate C_bound (from plan.md, section 2) ---
    print("Calculating C_bound...")

    # 2.1 Construct min_edge_cost for each edge
    min_edge_cost = np.min(c_mat, axis=1)  # Shape: (num_edges,)

    # 2.2 Build adjacency list: adj[u] = list of (v, eid)
    adj = defaultdict(list)
    for _, row in edges_df.iterrows(): # Use the derived edges_df
        eid, u, v = int(row.eid), int(row.u), int(row.v)
        adj[u].append((v, eid))
        # If graph is undirected and edges are listed once, add reverse edges:
        # adj[v].append((u, eid)) # Assuming directed as per plan's Dijkstra structure

    # 2.3 Dijkstra algorithm
    # Node IDs are assumed 0-indexed. Plan uses dist array of size 101 (for nodes 0-100).
    num_nodes_for_dist = 101 

    INF = 1e18
    dist = [INF] * num_nodes_for_dist
    
    if not route1_nodes:
        print("Error: route1_nodes is empty.")
        return
        
    start_node, goal_node = route1_nodes[0], route1_nodes[-1]

    if not (0 <= start_node < num_nodes_for_dist and 0 <= goal_node < num_nodes_for_dist):
        print(f"Error: Start node ({start_node}) or goal node ({goal_node}) is out of valid range [0, {num_nodes_for_dist-1}].")
        return

    dist[start_node] = 0
    hq = [(0, start_node)] # (current_distance, node_id)

    processed_nodes = 0
    while hq:
        d, u = heappop(hq)
        processed_nodes += 1

        if d > dist[u]:
            continue
        if u == goal_node:
            break 
        
        if u not in adj:
            continue # Node u has no outgoing edges

        for v, eid in adj[u]:
            if not (0 <= v < num_nodes_for_dist):
                print(f"Warning: Neighbor node {v} (from edge eid {eid}, u={u}) is out of range [0, {num_nodes_for_dist-1}]. Skipping.")
                continue
            if not (0 <= eid < len(min_edge_cost)):
                print(f"Warning: Edge ID {eid} (from u={u} to v={v}) is out of bounds for min_edge_cost (size {len(min_edge_cost)}). Skipping.")
                continue
            
            cost_uv = min_edge_cost[eid]
            if cost_uv == np.inf: # This edge (with its best bracket) is impassable
                continue

            if dist[u] + cost_uv < dist[v]:
                dist[v] = dist[u] + cost_uv
                heappush(hq, (dist[v], v))

    C_bound = dist[goal_node]
    print(f"Dijkstra processed {processed_nodes} states.")

    if C_bound == INF:
        print(f"Goal node {goal_node} is not reachable from start node {start_node} with the given edge costs.")
    else:
        print(f"C_bound = {C_bound:.4f} 元")

if __name__ == "__main__":
    solve_part2()
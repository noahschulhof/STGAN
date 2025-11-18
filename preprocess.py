import pandas as pd
import numpy as np

def create_required_files(df, output_dir="./"):
    """Create all required files for traffic forecasting model"""
    
    # Create all files
    data, nodes, timestamps = create_data_npy(df, output_dir)
    dist_matrix = create_node_dist(nodes, output_dir)
    adjacency_matrix = create_adjacency_matrix(dist_matrix, output_dir)
    create_subgraph_files(nodes, dist_matrix, adjacency_matrix, output_dir)
    create_time_features(timestamps, output_dir)
    
    print("\n=== All files created successfully ===")

def create_data_npy(df, output_dir):
    """Create data.npy with shape (timesteps, sensors, lanes, features)"""
    nodes = sorted(df['milemarker'].unique())
    timestamps = sorted(df['unix_time'].unique())
    
    num_timesteps = len(timestamps)
    num_nodes = len(nodes)
    num_lanes = 4
    num_features = 3  # speed, volume, occupancy
    
    data = np.zeros((num_timesteps, num_nodes, num_lanes, num_features))
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    
    for t, timestamp in enumerate(timestamps):
        for _, row in df[df['unix_time'] == timestamp].iterrows():
            node_idx = node_to_idx[row['milemarker']]
            for lane in range(1, 5):
                data[t, node_idx, lane-1, 0] = row[f'lane{lane}_speed']
                data[t, node_idx, lane-1, 1] = row[f'lane{lane}_volume']
                data[t, node_idx, lane-1, 2] = row[f'lane{lane}_occ']
    
    np.save(f"{output_dir}/data.npy", data)
    print(f"✓ data.npy: {data.shape} (timesteps, sensors, lanes, features)")
    return data, nodes, timestamps

def create_node_dist(nodes, output_dir):
    """Create node_dist.txt - Distance matrix between nodes"""
    node_positions = np.array(nodes).reshape(-1, 1)
    dist_matrix = np.abs(node_positions - node_positions.T)
    
    np.savetxt(f"{output_dir}/node_dist.txt", dist_matrix, fmt='%.6f')
    print(f"✓ node_dist.txt: {dist_matrix.shape}")
    return dist_matrix

def create_adjacency_matrix(dist_matrix, output_dir):
    """Create weighted adjacency matrix using Gaussian kernel"""
    sigma = np.std(dist_matrix[dist_matrix > 0])
    num_nodes = dist_matrix.shape[0]
    
    W = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        # Get 8 nearest neighbors (for 9-node subgraph)
        neighbors = np.argsort(dist_matrix[i])[1:9]  # exclude self
        for j in neighbors:
            W[i, j] = np.exp(-dist_matrix[i, j]**2 / sigma**2)
    
    np.savetxt(f"{output_dir}/adj_matrix.txt", W, fmt='%.6f')
    print(f"✓ adj_matrix.txt: {W.shape} ({np.count_nonzero(W)} edges)")
    return W

def create_subgraph_files(nodes, dist_matrix, adjacency_matrix, output_dir, n_nodes=9):
    """Create subgraph files with node indices"""
    num_nodes = len(nodes)
    node_subgraphs = []
    node_adjacent_list = []
    
    for i in range(num_nodes):
        closest_indices = np.argsort(dist_matrix[i])[:n_nodes]
        
        # Create subgraph adjacency matrix
        subgraph_adj = np.zeros((n_nodes, n_nodes))
        for idx_i, global_i in enumerate(closest_indices):
            for idx_j, global_j in enumerate(closest_indices):
                subgraph_adj[idx_i, idx_j] = adjacency_matrix[global_i, global_j]
        
        node_subgraphs.append(subgraph_adj)
        node_adjacent_list.append(closest_indices.tolist())
    
    # Save files
    np.save(f"{output_dir}/node_subgraph.npy", np.array(node_subgraphs))
    
    with open(f"{output_dir}/node_adjacent.txt", 'w') as f:
        for i, indices in enumerate(node_adjacent_list):
            f.write(f"{' '.join(map(str, indices))}\n")
    
    print(f"✓ node_subgraph.npy: ({num_nodes}, {n_nodes}, {n_nodes})")
    print(f"✓ node_adjacent.txt: {num_nodes} subgraphs")

def create_time_features(timestamps, output_dir):
    """Create time_features.txt - 31-dim (7 weekdays + 24 hours)"""
    time_features = []
    for timestamp in timestamps:
        dt = pd.to_datetime(timestamp, unit='s')
        
        # Create 31-dimensional vector: 7 weekdays + 24 hours
        time_vec = np.zeros(31, dtype=int)
        
        # Set weekday position (0-6)
        time_vec[dt.dayofweek] = 1
        
        # Set hour position (7-30) - offset by 7 for hours
        time_vec[7 + dt.hour] = 1
        
        time_features.append(time_vec)
    
    time_features = np.array(time_features)
    np.savetxt(f"{output_dir}/time_features.txt", time_features, fmt='%d')
    print(f"✓ time_features.txt: {time_features.shape} (7 weekdays + 24 hours = 31-dim)")
    return time_features

df = pd.read_csv('nashville_freeway_anomaly.csv')

create_required_files(df, 'traffic/data')
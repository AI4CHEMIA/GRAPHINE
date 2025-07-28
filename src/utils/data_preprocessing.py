
import numpy as np
import pandas as pd

def add_virtual_node(edges_index_df, node_f_df, out, edges_f, v_node_weight=0.1):
    """
    Adds a virtual node that connects to all existing nodes with a specified weight.

    Parameters:
    - edges_index_df: pd.DataFrame, shape (2, n_edges)
    - node_f_df: pd.DataFrame, shape (n_nodes, n_timestamps)
    - out: pd.DataFrame, same shape as node_f_df
    - edges_f: np.ndarray, shape (n_edges, 1)
    - v_node_weight: float, weight of edges from virtual node to each real node

    Returns:
    - Updated edges_index_df, node_f_df, out, edges_f
    """

    n_nodes = node_f_df.shape[0]

    # 1. Add edges from virtual node (new index = n_nodes) to all existing nodes
    virtual_node_index = n_nodes
    new_edges_source = np.full(n_nodes, virtual_node_index)
    new_edges_target = np.arange(n_nodes)

    # Shape: (2, n_nodes)
    virtual_edges = np.vstack([new_edges_source, new_edges_target])

    # Append to existing edges_index_df
    edges_index_df_new = pd.concat([
        edges_index_df,
        pd.DataFrame(virtual_edges, columns=[edges_index_df.shape[1]+i for i in range(n_nodes)])
    ], axis=1)

    # 2. Add a row of zeros to node_f_df and out
    new_node_features = pd.DataFrame([np.zeros(node_f_df.shape[1])], columns=node_f_df.columns)
    node_f_df_new = pd.concat([node_f_df, new_node_features], ignore_index=True)

    new_out = pd.DataFrame([np.zeros(out.shape[1])], columns=out.columns)
    out_new = pd.concat([out, new_out], ignore_index=True)

    # 3. Add corresponding weights for virtual edges
    virtual_edges_f = np.full((n_nodes, 1), v_node_weight)
    edges_f_new = np.vstack([edges_f, virtual_edges_f])

    return edges_index_df_new, node_f_df_new, out_new, edges_f_new

def load_and_preprocess_data(edges_path, node_features_path, output_path, v_node_weight=0.1):
    edges_index_df = pd.read_csv(edges_path)
    node_f_df = pd.read_csv(node_features_path)
    out = pd.read_csv(output_path)
    edges_f = np.ones((edges_index_df.shape[1], 1))

    n_edges_index_df, n_node_f_df, n_out, n_edges_f = add_virtual_node(edges_index_df, node_f_df, out, edges_f, v_node_weight=v_node_weight)

    edges_index = n_edges_index_df.to_numpy()
    node_f = n_out.to_numpy()
    node_f = node_f.T.reshape(n_node_f_df.shape[1], n_node_f_df.shape[0], 1)

    node_p = n_node_f_df.to_numpy()
    node_p = node_p.T.reshape(n_node_f_df.shape[1], n_node_f_df.shape[0],)

    return edges_index, n_edges_f, node_f, node_p



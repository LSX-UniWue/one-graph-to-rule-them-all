import torch
import torch_geometric
import pathpy as pp
import numpy as np

def get_edge_index(network):
    """
    Converts a pathpy network to a torch tensor capturing the adjacency list
    """
    num_edges = network.number_of_edges()
    if network.directed == False:
        num_edges *=2
    edge_index = torch.zeros([num_edges, 2], dtype=torch.long)
    
    for e in network.edges:
        edge_index[network.edges.index[e.uid], 0] = network.nodes.index[e.v.uid]
        edge_index[network.edges.index[e.uid], 1] = network.nodes.index[e.w.uid]
    if network.directed == False:
        for e in network.edges:
            edge_index[network.number_of_edges() + network.edges.index[e.uid], 0] = network.nodes.index[e.w.uid]
            edge_index[network.number_of_edges() + network.edges.index[e.uid], 1] = network.nodes.index[e.v.uid]
    
    return edge_index.t().contiguous()

def get_edge_weights(network):
    num_edges = network.number_of_edges()
    if network.directed == False:
        num_edges *=2
    edge_weights = torch.zeros([num_edges], dtype=torch.float)
    for e in network.edges:
        edge_weights[network.edges.index[e.uid]] = e['weight']
    if network.directed == False:
        for e in network.edges:
            edge_weights[network.number_of_edges() + network.edges.index[e.uid]] = e['weight']
    return edge_weights

def get_node_attributes(network, attribute, dtype=torch.float):
    """
    """
    x = np.random.choice([v.uid for v in network.nodes])
    tensor_dim = network.nodes[x][attribute].size(dim=0)
    node_features = torch.zeros([network.number_of_nodes(), tensor_dim], dtype=dtype)
    for v in network.nodes:
        for i in range(tensor_dim):
            node_features[network.nodes.index[v.uid], i] = v[attribute][i]

    return torch.tensor(node_features)

def network_to_pyg(net, x='x', y='y') -> torch_geometric.data.Data:
    """
    """
    data = torch_geometric.data.Data(
            ohe = torch.eye(net.number_of_nodes()).float(),
            y = get_node_attributes(net, y),
            word2vec = get_node_attributes(net, 'word2vec'),
            node2vec11 = get_node_attributes(net, 'node2vec-p1q1'),
            node2vec14 = get_node_attributes(net, 'node2vec-p1q4'),
            node2vec41 = get_node_attributes(net, 'node2vec-p4q1'),
            le = get_node_attributes(net, 'le'),
            edge_index = get_edge_index(net),
            edge_weights = get_edge_weights(net),
            node_names = [v.uid for v in net.nodes]
        )
    return data
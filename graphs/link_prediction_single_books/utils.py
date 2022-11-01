
import scipy as sp
def compute_laplacian_embedding(
    L,
    d,
    rescale_eigenvectors = False
    ):
    assert d+1<= L.shape[0], "can't compute embedding with number of dimensions larger than the number of nodes"
    # vals, vecs =  sp.linalg.eig(self.matrix) #.todense()
    vals, vecs = sp.sparse.linalg.eigs(
        L,
        k = d+1,
        which="SM",
        return_eigenvectors=True
        )

    #order vals and vecs
    idx = vals.argsort() 
    vals = vals[idx]
    vecs = vecs[:,idx].real
    
    # remove zero
    vals = vals[1:]
    vecs = vecs[:,1:].real
    # rescale contributions of eigenvectors
    if rescale_eigenvectors:
        vecs = 1/vals.real * vecs
    return vals, vecs


import infomap
def infomap_WW(
    csr_matrix,
    index_to_node,
    options = [],#["--directed"],
    include_isolated = True
    ):
    """
    That's a wrapper of infomap's wrapper (hence the WW) that runs infomap on a MultiOrderModel object
    Parameters
    -----------
    network: pp.Network
    options: iterable 
        list of options accepted by the infomap wrapper
    
    Returns
    -----------
    : dict 
    """
    #create wrapper object
    string_options = " ".join(options)
    string_options = string_options+" --silent"
    infomapWrapper = infomap.Infomap(string_options)
    #wrapper accepts only nodes with ints as ids -> map2int

    for i,j,value in zip(csr_matrix.nonzero()[0],csr_matrix.nonzero()[1],csr_matrix.data ):
        infomapWrapper.addLink(i,j, value)
    #find communities
    infomapWrapper.run()
    module_map = infomapWrapper.getModules()
    #put isolated nodes in own community
    if include_isolated:
        vs = csr_matrix.nonzero()[0]
        ws = csr_matrix.nonzero()[1]
        for isolated_node in set(range(csr_matrix.shape[0])) - set(vs).union(set(ws)):
            module_map[isolated_node] = max(module_map.values()) + 1 #"isolated_{}".format(isolated_node)
    #put back original node ids and return
    return  {index_to_node[k]:v for k,v in module_map.items()} #.items()

import random 
import matplotlib
# creates dictionary that gives same color to nodes in the same module (for input in pathpy visualization)
def module_to_color_string(module_map):
    module_to_color = {}
    module_ids = set(module_map.values())
    colors = list(matplotlib.colors.cnames.keys())
    random.shuffle(colors)
    node_color = {}
    for enum,module in enumerate(module_ids):
        #for node in module:
        module_to_color[module] = colors[enum]
    return {node:module_to_color[module] for (node,module) in module_map.items()}








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

def network_to_pyg(net): #, x='x', y='y') -> torch_geometric.data.Data:
    """
    """
    data = torch_geometric.data.Data(
            ohe = torch.eye(net.number_of_nodes()).float(),
            # y = get_node_attributes(net, y),
            word2vec = get_node_attributes(net, 'word2vec'),
            node2vec11 = get_node_attributes(net, 'node2vec-p1q1'),
            # node2vec14 = get_node_attributes(net, 'node2vec-p1q4'),
            # node2vec41 = get_node_attributes(net, 'node2vec-p4q1'),
            le = get_node_attributes(net, 'le'),
            edge_index = get_edge_index(net),
            edge_weights = get_edge_weights(net)
        )
    return data
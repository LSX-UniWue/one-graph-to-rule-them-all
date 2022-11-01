import pathpy as pp
import numpy as np
import pandas as pd
import torch
import torch_geometric
from sklearn.metrics import roc_auc_score, accuracy_score
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, GATConv 
from torch_geometric.utils import negative_sampling
from torch_geometric.transforms import RandomLinkSplit
from sklearn.model_selection import cross_validate
import json
from tqdm import tqdm
from utils import network_to_pyg
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, recall_score, precision_score
import warnings
warnings.filterwarnings('ignore')


def load_node_representations(one, network_name = "legendarium"):
    # add the node2vec embeddings as node features
    df = pd.read_csv(f'{network_name}_node2vec-p1q1.csv')
    for v in one.nodes:
        v['node2vec-p1q1'] = torch.from_numpy(df[df['characters'] == v.uid].iloc[:, :-1].values).squeeze()

    # add the node2vec embeddings as node features
    # df = pd.read_csv('node2vec-p1q4.csv')
    # for v in one.nodes:
    #     v['node2vec-p1q4'] = torch.from_numpy(df[df['characters'] == v.uid].iloc[:, :-1].values).squeeze()

    # # add the node2vec embeddings as node features
    # df = pd.read_csv('node2vec-p4q1.csv')
    # for v in one.nodes:
    #     v['node2vec-p4q1'] = torch.from_numpy(df[df['characters'] == v.uid].iloc[:, :-1].values).squeeze()

    # add the word2vec as node features
    df = pd.read_csv(f'{network_name}_words_and_vectors.csv')
    for v in one.nodes:
        v['word2vec'] = torch.from_numpy(df[df['words'] == v.uid].iloc[:, :-1].values).squeeze()

    # add the Laplacian Embeddings as node features
    df = pd.read_csv(f'{network_name}_LE_embedding.csv')
    for v in one.nodes:
        v['le'] = torch.from_numpy(df[df['characters'] == v.uid].iloc[:, :-1].values).squeeze()

    # adding the weights
    df = pd.read_csv(f'{network_name}_new_data.csv').loc[:, ['v', 'w']]
    weights = df.value_counts().to_dict()
    for e in one.edges:
        e['weight'] = weights[(e.v.uid, e.w.uid)]
    return one



def load_data(network_name = "legendarium"):
    assert network_name in ["legendarium", "lotr","hobbit","silmarillion"]
    # load the data set
    df = pd.read_csv(f'{network_name}_new_data.csv') 
    # load the empty network
    n = pp.Network(directed=False)
    # add the nodes
    for i in range(df.shape[0]):
        n.add_edge(df.loc[i, 'v'], df.loc[i, 'w'])
    n = load_node_representations(n, network_name = network_name)

    print(n)

    # convert the network to PyG data set
    data = network_to_pyg(n)
    return data








class Net_GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def encode(self, x, edge_index, edge_weight):
        x = self.conv1(x, edge_index, edge_weight).relu()
        return self.conv2(x, edge_index, edge_weight)

    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()


class Net_GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels)
        self.conv2 = GATConv(hidden_channels, out_channels)

    def encode(self, x, edge_index): #, edge_weight):
        # x = self.conv1(x, edge_index, edge_weight).relu()
        # return self.conv2(x, edge_index, edge_weight)
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()





def train(model, train_data, criterion, optimizer, X):
    model.train()
    optimizer.zero_grad()
    z = model.encode(X, train_data.edge_index , train_data.edge_weights)

    # We perform a new round of negative sampling for every training epoch:
    neg_edge_index = negative_sampling(
        edge_index=train_data.edge_index, num_nodes=train_data.num_nodes,
        num_neg_samples=train_data.edge_label_index.size(1), method='sparse')

    edge_label_index = torch.cat(
        [train_data.edge_label_index, neg_edge_index],
        dim=-1,
    )
    edge_label = torch.cat([
        train_data.edge_label,
        train_data.edge_label.new_zeros(neg_edge_index.size(1))
    ], dim=0)

    out = model.decode(z, edge_label_index).view(-1)
    loss = criterion(out, edge_label)
    loss.backward()
    optimizer.step()
    return loss



def train_GAT(model, train_data, criterion, optimizer, X):
    # MODIFIED for GAT because we get "assert self.lin_edge is not None " when passing edge weights 
    model.train()
    optimizer.zero_grad()
    z = model.encode(X, train_data.edge_index) #, train_data.edge_weights)

    # We perform a new round of negative sampling for every training epoch:
    neg_edge_index = negative_sampling(
        edge_index=train_data.edge_index, num_nodes=train_data.num_nodes,
        num_neg_samples=train_data.edge_label_index.size(1), method='sparse')

    edge_label_index = torch.cat(
        [train_data.edge_label_index, neg_edge_index],
        dim=-1,
    )
    edge_label = torch.cat([
        train_data.edge_label,
        train_data.edge_label.new_zeros(neg_edge_index.size(1))
    ], dim=0)

    out = model.decode(z, edge_label_index).view(-1)
    loss = criterion(out, edge_label)
    loss.backward()
    optimizer.step()
    return loss

def proba(x):
    return [0 if (i <= 0.5) else 1 for i in x]

@torch.no_grad()
def test(model, data, X):
    model.eval()
    z = model.encode(X, data.edge_index, data.edge_weights)
    out = model.decode(z, data.edge_label_index).view(-1).sigmoid()
    roc = roc_auc_score(data.edge_label.cpu().numpy(), out.cpu().numpy())
    acc = accuracy_score(data.edge_label.cpu().numpy(), proba(out.cpu().numpy()))
    return roc, acc


@torch.no_grad()
def test_GAT(model, data, X):
    model.eval()
    z = model.encode(X, data.edge_index)
    out = model.decode(z, data.edge_label_index).view(-1).sigmoid()
    roc = roc_auc_score(data.edge_label.cpu().numpy(), out.cpu().numpy())
    acc = accuracy_score(data.edge_label.cpu().numpy(), proba(out.cpu().numpy()))
    return roc, acc

##########################

def run_experiment_link_prediction_GCN(d = 20, network_name = "legendarium", n_epochs = 15000, nfolds = 10):
    data = load_data(network_name=network_name)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    results = {
        'node2vec41': [],
        'node2vec14': [],
        'node2vec11': [], 
        'word2vec': [], 
        'ohe': [], 
        'le': []
    }

    # for method in tqdm(['node2vec41','node2vec14','node2vec11', 'word2vec', 'ohe', 'le']):
    for method in tqdm(['node2vec11', 'word2vec', 'ohe', 'le']): # running only the best performing node2vec
        for _ in tqdm(range(nfolds)):
            transform = RandomLinkSplit(is_undirected=True, num_val=0.01, num_test=0.3, add_negative_train_samples=True)
            train_data, val_data, test_data = transform(data.to(device))
            
            X = train_data[method] 
            model = Net_GCN(
                X.shape[1],
                d,
                d).to(device)
            optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
            criterion = torch.nn.BCEWithLogitsLoss()

            best_val_auc = final_test_auc = 0
            # final_test_acc = 0 
            for _ in range(1, n_epochs):
                _ = train(model, train_data, criterion, optimizer, X)
                val_auc, val_acc = test(model, val_data, X)
                test_auc, test_acc = test(model, test_data, X)
                if val_auc > best_val_auc:
                    # best_val = val_auc
                    final_test_auc = test_auc
                    # final_test_acc = test_acc

            results[method].append(final_test_auc)
    return results



def run_experiment_link_prediction_GAT(d = 20, network_name = "legendarium", n_epochs = 15000, nfolds = 10):
    data = load_data(network_name=network_name)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    results = {
        'node2vec41': [],
        'node2vec14': [],
        'node2vec11': [], 
        'word2vec': [], 
        'ohe': [], 
        'le': []
    }
    # for method in tqdm(['node2vec41','node2vec14','node2vec11', 'word2vec', 'ohe', 'le']):
    for method in tqdm(['node2vec11', 'word2vec', 'ohe', 'le']):
        for _ in tqdm(range(nfolds)):
            
            transform = RandomLinkSplit(is_undirected=True, num_val=0.01, num_test=0.3, add_negative_train_samples=True)
            train_data, val_data, test_data = transform(data.to(device))
            
            X = train_data[method] 
            model = Net_GAT(
                X.shape[1],
                d,
                d).to(device)
            optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
            criterion = torch.nn.BCEWithLogitsLoss()

            best_val_auc = final_test_auc = 0
            # final_test_acc = 0 
            for _ in range(1, n_epochs):
                _ = train_GAT(model, train_data, criterion, optimizer, X)

                val_auc, val_acc = test_GAT(model, val_data, X)
                test_auc, test_acc = test_GAT(model, test_data, X)
                if val_auc > best_val_auc:
                    # best_val = val_auc
                    final_test_auc = test_auc
                    # final_test_acc = test_acc

            results[method].append(final_test_auc)
    return results
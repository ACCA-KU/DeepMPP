import torch.nn as nn
import torch
from dgl.nn import SumPooling

from D4CMPP.networks.src.GCN import GCNs
from D4CMPP.networks.src.SolventInteractionMap import SolventLayer

class network(nn.Module):
    def __init__(self, config):
        super(network, self).__init__()
        
        hidden_dim = config.get('hidden_dim', 64)
        conv_layers = config.get('conv_layers', 4)
        linear_layers = config.get('linear_layers', 2)
        self.node_embedding = nn.Linear(config['node_dim'], hidden_dim)
        self.node_embedding_solv = nn.Linear(config['node_dim'], 64)

        self.GCNs = GCNs(hidden_dim, hidden_dim, hidden_dim, nn.ReLU(), conv_layers, 0.2, False, True) # in_feats, hidden_feats, out_feats, activation, n_layers, dropout=0.2, batch_norm=False, residual_sum = False):
        self.SolventLayer = SolventLayer(config)

    def forward(self, graph, node_feats,solv_graph,solv_node_feats,**kwargs):
        h = self.node_embedding(node_feats)
        h = self.GCNs(graph, h)

        h = self.SolventLayer(graph, h, solv_graph, solv_node_feats)

        return h
    
    def loss_fn(self, scores, targets):
        return nn.MSELoss()(targets[~torch.isnan(targets)],scores[~torch.isnan(targets)])

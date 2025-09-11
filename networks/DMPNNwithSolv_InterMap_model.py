
import dgl
import torch
import torch.nn as nn
from dgl.nn import SumPooling

from D4CMPP.networks.src.DMPNN import DMPNNLayer
from D4CMPP.networks.src.SolventInteractionMap import SolventLayer


class network(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = config.get('hidden_dim', 64)
        conv_layers = config.get('conv_layers', 4)
        dropout = config.get('dropout', 0.2)

        self.embedding_node_lin = nn.Linear(config['node_dim'], hidden_dim, bias=True)
        self.embedding_edge_lin = nn.Linear(config['edge_dim'], hidden_dim, bias=True)
        self.init_h_func = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim, bias=True),
            nn.LeakyReLU()
        )
        self.W_a = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim, bias=True),
            nn.LeakyReLU()
        )
        self.dropout_layer = nn.Dropout(dropout)
        self.DMPNNLayer = nn.ModuleList([DMPNNLayer(hidden_dim,hidden_dim,hidden_dim,nn.LeakyReLU(),0.2) for _ in range(conv_layers)])
        self.SolvLayer = SolventLayer(config)
        

    def forward(self, graph, node_feats, edge_feats, solv_graph, solv_node_feats, **kwargs):
        node = self.embedding_node_lin(node_feats)
        edge = self.embedding_edge_lin(edge_feats)

        direct_feats = None
        backward_feats = None
        for layer in self.DMPNNLayer:
            hidden_feats, direct_feats, backward_feats = layer(graph, node, edge, direct_feats, backward_feats)

        output = self.SolvLayer(graph, hidden_feats, solv_graph, solv_node_feats)
        return output
    
    def loss_fn(self, scores, targets):
        return nn.MSELoss()(targets[~torch.isnan(targets)],scores[~torch.isnan(targets)])



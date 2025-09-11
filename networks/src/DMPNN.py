
"""
This codes are modified from the project "GC-GNN" (https://github.com/gsi-lab/GC-GNN)
The original codes are under the MIT License. (https://github.com/gsi-lab/GC-GNN/blob/main/networks/DMPNN.py)
"""
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

class DMPNNLayer(nn.Module):
    def __init__(self, in_feats, edge_feats, out_feats, activation, dropout=0.2):
        super(DMPNNLayer, self).__init__()
        self.activation = activation

        self.W_m = nn.Linear(in_feats+edge_feats, out_feats, bias=True)
        self.dropout_layer = nn.Dropout(dropout)

    
    def initial_message_func(self, edges):
        return {'direct_feats': self.W_m(torch.cat([edges.data['edge_feats'], edges.src['node_feats']], 1))
                }
    def initial_backward_message_func(self, edges):
        return {'backward_feats': self.W_m(torch.cat([edges.data['edge_feats'], edges.dst['node_feats']], 1))
                }

    def direct_message_func(self, edges):
        return {'direct_feats': self.W_m(torch.cat([edges.data['edge_feats'], edges.data['prev_direct_feats']], 1))
                }
    def backward_message_func(self, edges):
        return {'backward_feats': self.W_m(torch.cat([edges.data['edge_feats'], edges.data['prev_backward_feats']], 1))
                }


    def just_message_func(self, edges):
        return {'mail': edges.data['direct_feats'] }
    def reducer_sum(self, nodes):
        return {'full_feats': torch.sum(nodes.mailbox['mail'], 1)}

    def edge_func(self, edges):
        return {'new_direct_feats': edges.src['full_feats'] - edges.data['backward_feats'],
                'new_backward_feats': edges.dst['full_feats'] - edges.data['direct_feats']}
        


    def forward(self, graph, node_feats, edge_feats, direct_feats=None, backward_feats=None):
        with graph.local_scope():
            graph.edata['edge_feats'] = edge_feats
            if backward_feats is not None and direct_feats is not None:
                graph.edata['prev_backward_feats'] = backward_feats
                graph.edata['prev_direct_feats'] = direct_feats
                graph.apply_edges(self.direct_message_func)
                graph.apply_edges(self.backward_message_func)
                graph.update_all(self.just_message_func, self.reducer_sum)
            else:
                graph.ndata['node_feats'] = node_feats
                graph.apply_edges(self.initial_message_func)
                graph.apply_edges(self.initial_backward_message_func)
                graph.update_all(self.just_message_func, self.reducer_sum)
            graph.apply_edges(self.edge_func)

            new_node_feats = graph.ndata['full_feats']
            new_edge_feats = graph.edata['new_direct_feats']
            new_backward_feats = graph.edata['new_backward_feats']
            if self.activation is not None:
                new_node_feats = self.activation(new_node_feats)
            new_node_feats = self.dropout_layer(new_node_feats)

            return new_node_feats, new_edge_feats, new_backward_feats


from D4CMPP.networks.src.GCN import GCNs
from D4CMPP.networks.src.Linear import Linears
import torch.nn as nn
import torch
from torch.nn.utils.rnn import pad_sequence

from dgl.nn import SumPooling

class SolventLayer(nn.Module):
    def __init__(self, config):
        super(SolventLayer, self).__init__()
        

        hidden_dim = config.get('hidden_dim', 64)
        linear_layers = min(config.get('linear_layers', 2),6)
        self.node_embedding_solv = nn.Linear(config['node_dim'], hidden_dim)

        self.h_embedding_layer = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.solv_embedding_layer = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.h_batch_norm = nn.BatchNorm1d(hidden_dim)
        self.solv_batch_norm = nn.BatchNorm1d(hidden_dim)

        self.GCNs_solv = GCNs(hidden_dim, hidden_dim, hidden_dim, nn.ReLU(), 4, 0.2, False, True)
        self.reduce_func = SumPooling()


        self.Linears3 = Linears(hidden_dim*4,config['target_dim'], nn.ReLU(), linear_layers, 0.2, False, False,True) # in_feats, out_feats, activation, n_layers, dropout=0.2, batch_norm=False, residual_sum = False

    def forward(self, graph, hidden_feats,solv_graph,solv_node_feats,**kwargs):
        h = hidden_feats

        h_solv = self.node_embedding_solv(solv_node_feats)
        h_solv = self.GCNs_solv(solv_graph, h_solv)
        

        _h = self.h_embedding_layer(h)
        _h = self.h_batch_norm(_h)
        _h_solv = self.solv_embedding_layer(h_solv)
        _h_solv = self.solv_batch_norm(_h_solv)

        mol_node_counts = graph.batch_num_nodes().tolist()   # [n_atoms_mol1, n_atoms_mol2, ...]
        solv_node_counts = solv_graph.batch_num_nodes().tolist() # [n_atoms_solv1, n_atoms_solv2, ...]
        B= len(mol_node_counts)
        h_list = torch.split(_h, mol_node_counts, dim=0)
        h_solv_list = torch.split(_h_solv, solv_node_counts, dim=0)

        _h = pad_sequence(h_list, batch_first=True, padding_value=0.0)
        _h_solv = pad_sequence(h_solv_list, batch_first=True, padding_value=0.0)

        # interaction_map = nn.Tanh()(h @ h_solv.T)
        interaction_map = torch.tanh(torch.bmm( _h, _h_solv.transpose(1, 2)))
        h_2 = torch.bmm(interaction_map, _h_solv)  
        h_solv2 = torch.bmm(interaction_map.transpose(1, 2), _h)

        h_2_list = []
        h_solv2_list = []
        for i in range(B):
            n_mol = mol_node_counts[i]
            n_solv = solv_node_counts[i]
            h_2_list.append(h_2[i, :n_mol])
            h_solv2_list.append(h_solv2[i, :n_solv])
        h_2 = torch.cat(h_2_list, dim=0)
        h_solv2 = torch.cat(h_solv2_list, dim=0)

        h_solv_reduce = self.reduce_func(solv_graph, torch.cat([h_solv, h_solv2], dim=1))
        h_reduce = self.reduce_func(graph, torch.cat([h, h_2], dim=1))

        h = torch.cat([h_reduce,h_solv_reduce],axis=1)
        h = self.Linears3(h)
        return h
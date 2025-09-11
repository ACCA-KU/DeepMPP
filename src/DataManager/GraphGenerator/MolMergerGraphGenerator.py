import dgl
import torch
import rdkit.Chem as Chem
import numpy as np
import traceback
from D4CMPP.src.utils.featureizer import get_atom_features, get_bond_features, InvalidAtomError

class MolGraphGenerator:
    def __init__(self):
        self.af = get_atom_features
        self.bf = get_bond_features
        self.set_feature_dim()

    # Set the feature dimensions by generating a dummy graph
    def set_feature_dim(self):
        self.node_dim = self.af(Chem.MolFromSmiles('C')).shape[1]
        self.edge_dim = self.bf(Chem.MolFromSmiles('C-C')).shape[1]

    # Get the graph from the SMILES
    def get_graph(self,smi,**kwargs):
        # The smiles have to consist of two molecules
        if smi.count('.') != 1:
            raise ValueError("The SMILES should contain two molecules separated by a dot (.)")
        smi1, smi2 = smi.split('.')

        mol1 = Chem.MolFromSmiles(smi1)
        mol2 = Chem.MolFromSmiles(smi2)
        mol2 = Chem.AddHs(mol2)


        if mol1 is None or mol2 is None:
            raise Exception("Invalid SMILES: failed to generate mol object")
        src1, dst1 = self.generate_mol_graph(mol1)
        g1 = dgl.graph((src1, dst1), num_nodes=mol1.GetNumAtoms())
        g1 = self.add_feature(g1,mol1)

        src2, dst2 = self.generate_mol_graph(mol2)
        g2 = dgl.graph((src2, dst2), num_nodes=mol2.GetNumAtoms())
        g2 = self.add_feature(g2,mol2)

        # Connect most positive and negative partial charge atoms
        Chem.rdPartialCharges.ComputeGasteigerCharges(mol1)
        Chem.rdPartialCharges.ComputeGasteigerCharges(mol2)
        
        min_charge1 = None
        max_charge1 = None
        for i,atom in enumerate(mol1.GetAtoms()):
            charge = atom.GetProp("_GasteigerCharge")  # 문자열로 반환됨
            if min_charge1 is None or float(charge) < min_charge1:
                min_charge1 = i
            if max_charge1 is None or float(charge) > max_charge1:
                max_charge1 = i
        min_charge2 = None
        max_charge2 = None
        for i,atom in enumerate(mol2.GetAtoms()):
            charge = atom.GetProp("_GasteigerCharge")
            if min_charge2 is None or float(charge) < min_charge2:
                min_charge2 = i
            if max_charge2 is None or float(charge) > max_charge2:
                max_charge2 = i

        src2 = [i+max(src1)+1 for i in src2]
        dst2 = [i+max(src1)+1 for i in dst2]
        g = dgl.graph((src1+src2, dst1+dst2), num_nodes=mol1.GetNumAtoms()+mol2.GetNumAtoms())
        g.add_edges(max(src1)+1+min_charge2, max_charge1)
        g.add_edges(max_charge1, max(src1)+1+min_charge2)
        g.add_edges(min_charge1, max(src1)+1+max_charge2)
        g.add_edges(max(src1)+1+max_charge2, min_charge1)

        g.ndata['f'] = torch.cat([g1.ndata['f'], g2.ndata['f']], dim=0)
        g.edata['f'] = torch.cat([g1.edata['f'], g2.edata['f'],torch.zeros((4, g1.edata['f'].shape[1]))], dim=0)

        return g
    
    def get_empty_graph(self):
        return dgl.graph(([],[]))

    # Add the features to the graph
    def add_feature(self, g, mol):
        atom_feature = self.af(mol)
        g.ndata['f'] = torch.tensor(atom_feature).float()

        bond_feature = self.bf(mol)
        edata = torch.tensor(bond_feature).float()
        edata = torch.cat([edata,edata],dim=0)
        g.edata['f'] = edata
        return g
    
    # Generate the graph from the molecule object
    def generate_mol_graph(self,mol):
        num_atoms = mol.GetNumAtoms()
        if num_atoms == 1:
            return ([0],[0])
        src, dst = [], []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            src.append(start)
            dst.append(end)

        return (src+dst, dst+src)
    
    def generate_graph(self,mol):
        mol_data = self.generate_mol_graph(mol)            
        g = dgl.graph(mol_data, num_nodes=mol.GetNumAtoms())
        return g
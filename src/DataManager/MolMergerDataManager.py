from .Dataset.GraphDataset import GraphDataset
from .GraphGenerator.MolMergerGraphGenerator import MolGraphGenerator
from .MolDataManager import MolDataManager

class MolMergerDataManager(MolDataManager):
    def __init__(self, config):
        super().__init__(config)
    def import_others(self):
        self.graph_type = 'mol'
        self.gg = MolGraphGenerator()
        self.dataset = GraphDataset
        self.unwrapper = self.dataset.unwrapper
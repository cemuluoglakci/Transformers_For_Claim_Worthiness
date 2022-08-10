import torch

class Constants(object):
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.parent_dir = None
        self.seed_list = [7, 42] # seed_list = [7, 42, 127]
        self.fold_count = 3 #5
        self.patience = 5
        self.loss_function = None
        self.metric_types = None

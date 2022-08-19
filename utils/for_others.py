import torch

from utils import for_plankton as fplankton
from utils import prepare_train_test_data as pdata

torch.manual_seed(0)


class CreateDataForOthers:
    def __init__(self):
        self.checkpoint_path = None
        self.classes = None
        self.test_dataloader = None
        self.train_dataloader = None
        self.val_dataloader = None
        self.X_val = None
        self.X_test = None
        self.X_train = None
        self.class_weights_tensor = None
        self.params = None
        return

    def make_train_test_for_others(self, train_main):

        prep_test_data = pdata.CreateDataset()
        prep_test_data.LoadData_for_others(train_main)
        prep_test_data.CreatedataSetsForOthers(train_main)

        loaded_data = fplankton.CreateDataForPlankton()
        loaded_data.make_train_test_for_others(train_main, prep_test_data)
        loaded_data.create_data_loaders_for_others(train_main)

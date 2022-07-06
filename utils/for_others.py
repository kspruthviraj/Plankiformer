from utils import prepare_data_for_testing as pdata_test
from utils import for_plankton_test as fplankton_test

import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from torch.utils.data import Dataset
from torchvision import datasets
from pathlib import Path


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

        train_PATH = train_main.params.datapaths
        test_PATH = train_main.params.test_path

        train_PATH = ' '.join(map(str, train_PATH))
        test_PATH = ' '.join(map(str, test_PATH))

        prep_test_data = pdata_test.CreateDataset()
        prep_test_data.LoadData_for_others(train_main)
        prep_test_data.CreatedataSets(train_main)

        loaded_data = fplankton_test.CreateDataForOthers()
        loaded_data.make_data_for_others(train_main, prep_test_data)
        loaded_data.create_data_loaders_for_others(train_main)

###########
# IMPORTS #
###########

import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset


class CreateDataForPlankton:
    def __init__(self):
        self.classes = None
        self.classes_int = None
        self.Filenames = None
        self.val_dataloader = None
        self.test_dataloader = None
        self.train_dataloader = None
        self.checkpoint_path = None
        self.y_val = None
        self.y_test = None
        self.y_train = None
        self.testFilenames = None
        self.trainFilenames = None
        self.valFilenames = None
        self.X_val = None
        self.X_test = None
        self.X_train = None
        self.class_weights_tensor = None
        self.params = None
        return

    def make_train_test_for_model(self, train_main, test_main, prep_data):
        Data = prep_data.Data
        # self.class_weights_tensor = prep_data.tt.class_weights_tensor
        # self.Filenames = prep_data.Filenames
        self.classes = np.load(test_main.params.main_param_path + '/classes.npy')
        self.Filenames = prep_data.Filenames
        self.class_weights_tensor = torch.load(test_main.params.main_param_path + '/class_weights_tensor.pt')

        if train_main.params.ttkind == 'mixed':
            self.trainFilenames = Data[0]
            trainXimage = Data[1]
            trainXfeat = Data[9]
            trX = [trainXimage, trainXfeat]
        elif train_main.params.ttkind == 'image' and train_main.params.compute_extrafeat == 'yes':
            self.trainFilenames = Data[0]
            trainXimage = Data[1]
            trainXfeat = Data[9]
            trX = [trainXimage, trainXfeat]
        elif train_main.params.ttkind == 'feat':
            self.trainFilenames = Data[0]
            trainXfeat = Data[9]
            trX = [trainXfeat]
        else:
            self.trainFilenames = Data[0]
            trX = Data[1]

        data_train = trX.astype(np.float64)
        data_train = 255 * data_train
        self.X_train = data_train.astype(np.uint8)

        return

    def make_train_test_for_model_with_y(self, train_main, test_main, prep_data):
        Data = prep_data.Data
        # classes = prep_data.classes
        self.classes = np.load(test_main.params.main_param_path + '/classes.npy')
        self.Filenames = prep_data.Filenames
        self.class_weights_tensor = torch.load(test_main.params.main_param_path + '/class_weights_tensor.pt')

        if train_main.params.ttkind == 'mixed':
            self.trainFilenames = Data[0]
            trainXimage = Data[1]
            trainXfeat = Data[9]
            trY = Data[2]
            trX = [trainXimage, trainXfeat]
        elif train_main.params.ttkind == 'image' and train_main.params.compute_extrafeat == 'yes':
            self.trainFilenames = Data[0]
            trainXimage = Data[1]
            trY = Data[2]
            trainXfeat = Data[9]
            trX = [trainXimage, trainXfeat]
        elif train_main.params.ttkind == 'feat':
            self.trainFilenames = Data[0]
            trY = Data[2]
            trainXfeat = Data[9]
            trX = [trainXfeat]
        else:
            self.trainFilenames = Data[0]
            trX = Data[1]
            trY = Data[2]

        data_train = trX.astype(np.float64)
        data_train = 255 * data_train
        self.X_train = data_train.astype(np.uint8)
        # self.y_train = np.array([classes_int[y_train_max[i]] for i in range(len(y_train_max))], dtype=object)
        self.y_train = np.concatenate([np.where(self.classes == uid) if np.where(self.classes == uid) else print(
            'The folder should match the trained classes') for uid in trY]).ravel()

        return

    def create_data_loaders(self, train_main):
        # self.checkpoint_path = test_main.params.model_path

        test_dataset = CreateDataset(X=self.X_train)
        self.test_dataloader = DataLoader(test_dataset, 32, shuffle=False, num_workers=0,
                                          pin_memory=True)

    def create_data_loaders_with_y(self, test_main):
        # self.checkpoint_path = test_main.params.model_path

        test_dataset = CreateDataset_with_y(X=self.X_train, y=self.y_train)
        self.test_dataloader = DataLoader(test_dataset, 32, shuffle=False, num_workers=0,
                                          pin_memory=True)
        torch.save(self.test_dataloader, test_main.params.main_param_path + '/test_dataloader.pt')


class CreateDataset(Dataset):
    """Characterizes a dataset for PyTorch"""

    def __init__(self, X):
        """Initialization"""
        self.X = X

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.X)

    def __getitem__(self, index):
        """Generates one sample of data"""
        # Select sample
        image = self.X[index]
        X = self.transform(image)
        sample = X
        return sample

    transform = T.Compose([
        T.ToPILImage(),
        T.Resize(224),
        T.ToTensor()])
    transform_y = T.Compose([T.ToTensor()])


class CreateDataset_with_y(Dataset):
    """Characterizes a dataset for PyTorch"""

    def __init__(self, X, y):
        """Initialization"""
        self.X = X
        self.y = y

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.X)

    def __getitem__(self, index):
        """Generates one sample of data"""
        # Select sample
        image = self.X[index]
        label = self.y[index]
        X = self.transform(image)
        y = label
        #         y = self.transform_y(label)
        #         sample = {'image': X, 'label': label}
        sample = [X, y]
        return sample

    transform = T.Compose([
        T.ToPILImage(),
        T.Resize(224),
        T.ToTensor()])
    transform_y = T.Compose([T.ToTensor()])

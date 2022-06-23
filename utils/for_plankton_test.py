###########
# IMPORTS #
###########

import numpy as np
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset


class CreateDataForPlankton:
    def __init__(self):
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

    def make_train_test_for_model(self, train_main, prep_data):
        Data = prep_data.Data
        self.class_weights_tensor = prep_data.tt.class_weights_tensor
        self.Filenames = prep_data.Filenames

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

    def create_data_loaders(self, train_main, test_main):
        self.checkpoint_path = test_main.params.model_path

        test_dataset = CreateDataset(X=self.X_train)
        self.test_dataloader = DataLoader(test_dataset, train_main.params.batch_size, shuffle=True, num_workers=4,
                                          pin_memory=True)


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

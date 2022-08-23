###########
# IMPORTS #
###########
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
torch.manual_seed(0)


class CreateDataForPlankton:
    def __init__(self):
        self.Filenames_val = None
        self.Filenames_test = None
        self.Filenames_train = None
        self.classes_int = None
        self.classes = None
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
        if os.path.exists(train_main.params.outpath + '/Data.pickle'):
            Data = pd.read_pickle(train_main.params.outpath + '/Data.pickle')
            classes = np.load(train_main.params.outpath + '/classes.npy')
            self.class_weights_tensor = torch.load(train_main.params.outpath + '/class_weights_tensor.pt')
        # if train_main.params.saved_data == 'yes':
        #     Data = pd.read_pickle(train_main.params.outpath + '/Data.pickle')
        #     classes = np.load(train_main.params.outpath + '/classes.npy')
        #     self.class_weights_tensor = torch.load(train_main.params.outpath + '/class_weights_tensor.pt')
        else:
            self.class_weights_tensor = prep_data.class_weights_tensor
            self.Filenames = prep_data.Filenames
            classes = prep_data.classes
            Data = prep_data.Data

        if train_main.params.test_set == 'no':
            if train_main.params.ttkind == 'mixed':
                self.trainFilenames = Data[0]
                trainXimage = Data[1]
                trY = Data[2]
                trainXfeat = Data[9]
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

        elif train_main.params.test_set == 'yes':
            if train_main.params.valid_set == 'no':
                if train_main.params.ttkind == 'mixed':
                    self.trainFilenames = Data[0]
                    trainXimage = Data[1]
                    trY = Data[2]
                    self.testFilenames = Data[3]
                    testXimage = Data[4]
                    teY = Data[5]
                    trainXfeat = Data[9]
                    testXfeat = Data[10]
                    trX = [trainXimage, trainXfeat]
                    teX = [testXimage, testXfeat]
                elif train_main.params.ttkind == 'image' and train_main.params.compute_extrafeat == 'yes':
                    self.trainFilenames = Data[0]
                    trainXimage = Data[1]
                    trY = Data[2]
                    self.testFilenames = Data[3]
                    testXimage = Data[4]
                    teY = Data[5]
                    trainXfeat = Data[9]
                    testXfeat = Data[10]
                    trX = [trainXimage, trainXfeat]
                    teX = [testXimage, testXfeat]
                elif train_main.params.ttkind == 'feat':
                    self.trainFilenames = Data[0]
                    trainXfeat = Data[9]
                    trY = Data[2]
                    testXfeat = Data[10]
                    teY = Data[5]
                    trX = [trainXfeat]
                    teX = [testXfeat]
                else:
                    self.trainFilenames = Data[0]
                    trX = Data[1]
                    trY = Data[2]
                    self.testFilenames = Data[3]
                    teX = Data[4]
                    teY = Data[5]

            elif train_main.params.valid_set == 'yes':
                if train_main.params.ttkind == 'mixed':
                    self.trainFilenames = Data[0]
                    trainXimage = Data[1]
                    trY = Data[2]
                    self.testFilenames = Data[3]
                    testXimage = Data[4]
                    teY = Data[5]
                    self.valFilenames = Data[6]
                    valXimage = Data[7]
                    veY = Data[8]
                    trainXfeat = Data[9]
                    testXfeat = Data[10]
                    valXfeat = Data[11]
                    trX = [trainXimage, trainXfeat]
                    teX = [testXimage, testXfeat]
                    veX = [valXimage, valXfeat]
                elif train_main.params.ttkind == 'image' and train_main.params.compute_extrafeat == 'yes':
                    self.trainFilenames = Data[0]
                    trainXimage = Data[1]
                    trY = Data[2]
                    self.testFilenames = Data[3]
                    testXimage = Data[4]
                    teY = Data[5]
                    self.valFilenames = Data[6]
                    valXimage = Data[7]
                    veY = Data[8]
                    trainXfeat = Data[9]
                    testXfeat = Data[10]
                    valXfeat = Data[11]
                    trX = [trainXimage, trainXfeat]
                    teX = [testXimage, testXfeat]
                    veX = [valXimage, valXfeat]
                elif train_main.params.ttkind == 'feat':
                    self.trainFilenames = Data[0]
                    self.testFilenames = Data[3]
                    self.valFilenames = Data[6]
                    trainXfeat = Data[9]
                    testXfeat = Data[10]
                    valXfeat = Data[11]
                    trY = Data[2]
                    teY = Data[5]
                    veY = Data[8]
                    trX = [trainXfeat]
                    teX = [testXfeat]
                    veX = [valXfeat]
                else:
                    self.trainFilenames = Data[0]
                    trX = Data[1]
                    trY = Data[2]
                    self.testFilenames = Data[3]
                    teX = Data[4]
                    teY = Data[5]
                    self.valFilenames = Data[6]
                    veX = Data[7]
                    veY = Data[8]

        classes_int = np.unique(np.argmax(trY, axis=1))
        self.classes = classes
        self.classes_int = classes_int

        if train_main.params.test_set == 'no':
            y_train_max = trY.argmax(axis=1)  # The class that the classifier would bet on
            self.y_train = np.array([classes_int[y_train_max[i]] for i in range(len(y_train_max))], dtype=object)

            data_train = trX.astype(np.float64)
            data_train = 255 * data_train
            self.X_train = data_train.astype(np.uint8)

        elif train_main.params.test_set == 'yes' and train_main.params.valid_set == 'no':
            y_train_max = trY.argmax(axis=1)  # The class that the classifier would bet on
            self.y_train = np.array([classes_int[y_train_max[i]] for i in range(len(y_train_max))], dtype=object)

            y_test_max = teY.argmax(axis=1)  # The class that the classifier would bet on
            self.y_test = np.array([classes_int[y_test_max[i]] for i in range(len(y_test_max))], dtype=object)

            data_train = trX.astype(np.float64)
            data_train = 255 * data_train
            self.X_train = data_train.astype(np.uint8)

            data_test = teX.astype(np.float64)
            data_test = 255 * data_test
            self.X_test = data_test.astype(np.uint8)

        elif train_main.params.test_set == 'yes' and train_main.params.valid_set == 'yes':
            y_train_max = trY.argmax(axis=1)  # The class that the classifier would bet on
            self.y_train = np.array([classes_int[y_train_max[i]] for i in range(len(y_train_max))], dtype=object)

            y_test_max = teY.argmax(axis=1)  # The class that the classifier would bet on
            self.y_test = np.array([classes_int[y_test_max[i]] for i in range(len(y_test_max))], dtype=object)

            y_val_max = veY.argmax(axis=1)  # The class that the classifier would bet on
            self.y_val = np.array([classes_int[y_val_max[i]] for i in range(len(y_val_max))], dtype=object)

            data_train = trX.astype(np.float64)
            data_train = 255 * data_train
            self.X_train = data_train.astype(np.uint8)

            data_test = teX.astype(np.float64)
            data_test = 255 * data_test
            self.X_test = data_test.astype(np.uint8)

            data_val = veX.astype(np.float64)
            data_val = 255 * data_val
            self.X_val = data_val.astype(np.uint8)

        return

    def create_data_loaders(self, train_main):
        self.checkpoint_path = train_main.params.outpath + 'trained_models/' + train_main.params.init_name + '/'
        Path(self.checkpoint_path).mkdir(parents=True, exist_ok=True)

        if train_main.params.test_set == 'yes' and train_main.params.valid_set == 'yes':
            train_dataset = AugmentedDataset(X=self.X_train, y=self.y_train)
            self.train_dataloader = DataLoader(train_dataset, train_main.params.batch_size, shuffle=True, num_workers=4,
                                               pin_memory=True)

            test_dataset = CreateDataset(X=self.X_test, y=self.y_test)
            self.test_dataloader = DataLoader(test_dataset, train_main.params.batch_size, shuffle=True, num_workers=4,
                                              pin_memory=True)

            val_dataset = CreateDataset(X=self.X_val, y=self.y_val)
            self.val_dataloader = DataLoader(val_dataset, train_main.params.batch_size, shuffle=True, num_workers=4,
                                             pin_memory=True)
        elif train_main.params.test_set == 'no':
            train_dataset = AugmentedDataset(X=self.X_train, y=self.y_train)
            self.train_dataloader = DataLoader(train_dataset, train_main.params.batch_size, shuffle=True, num_workers=4,
                                               pin_memory=True)
        elif train_main.params.test_set == 'yes' and train_main.params.valid_set == 'no':
            train_dataset = AugmentedDataset(X=self.X_train, y=self.y_train)
            self.train_dataloader = DataLoader(train_dataset, train_main.params.batch_size, shuffle=True, num_workers=4,
                                               pin_memory=True)

            test_dataset = CreateDataset(X=self.X_test, y=self.y_test)
            self.test_dataloader = DataLoader(test_dataset, train_main.params.batch_size, shuffle=True, num_workers=4,
                                              pin_memory=True)

    def make_train_test_for_others(self, prep_data):
        self.class_weights_tensor = prep_data.class_weights_tensor
        self.Filenames = prep_data.Filenames
        classes = prep_data.classes
        Data = prep_data.Data

        # Data = pd.read_pickle(train_main.params.outpath + '/Data.pickle')
        # classes = np.load(train_main.params.outpath + '/classes.npy')

        self.trainFilenames = Data[0]
        trX = Data[1]
        trY = Data[2]
        self.testFilenames = Data[3]
        teX = Data[4]
        teY = Data[5]
        self.valFilenames = Data[6]
        veX = Data[7]
        veY = Data[8]

        classes_int = np.unique(np.argmax(trY, axis=1))
        self.classes = classes
        self.classes_int = classes_int

        y_train_max = trY.argmax(axis=1)  # The class that the classifier would bet on
        self.y_train = np.array([classes_int[y_train_max[i]] for i in range(len(y_train_max))], dtype=object)

        y_test_max = teY.argmax(axis=1)  # The class that the classifier would bet on
        self.y_test = np.array([classes_int[y_test_max[i]] for i in range(len(y_test_max))], dtype=object)

        y_val_max = veY.argmax(axis=1)  # The class that the classifier would bet on
        self.y_val = np.array([classes_int[y_val_max[i]] for i in range(len(y_val_max))], dtype=object)

        data_train = trX.astype(np.float64)
        data_train = 255 * data_train
        self.X_train = data_train.astype(np.uint8)

        data_test = teX.astype(np.float64)
        data_test = 255 * data_test
        self.X_test = data_test.astype(np.uint8)

        data_val = veX.astype(np.float64)
        data_val = 255 * data_val
        self.X_val = data_val.astype(np.uint8)

        return

    def create_data_loaders_for_others(self, train_main):
        self.checkpoint_path = train_main.params.outpath + 'trained_models/' + train_main.params.init_name + '/'
        Path(self.checkpoint_path).mkdir(parents=True, exist_ok=True)

        train_dataset = AugmentedDataset(X=self.X_train, y=self.y_train)
        self.train_dataloader = DataLoader(train_dataset, train_main.params.batch_size, shuffle=True, num_workers=4,
                                           pin_memory=True)

        test_dataset = CreateDataset(X=self.X_test, y=self.y_test)
        self.test_dataloader = DataLoader(test_dataset, train_main.params.batch_size, shuffle=True, num_workers=4,
                                          pin_memory=True)

        val_dataset = CreateDataset(X=self.X_val, y=self.y_val)
        self.val_dataloader = DataLoader(val_dataset, train_main.params.batch_size, shuffle=True, num_workers=4,
                                         pin_memory=True)


class AugmentedDataset(Dataset):
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
        sample = [X, y]
        return sample

    transform = T.Compose([
        T.ToPILImage(),
        T.Resize(224),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandAugment(),
        T.TrivialAugmentWide(),
        # T.AugMix(),
        T.RandomResizedCrop(size=(224, 224)),
        T.RandomErasing(),
        # T.Grayscale(),
        T.RandomInvert(),
        T.RandomAutocontrast(),
        T.RandomEqualize(),
        T.RandomAdjustSharpness(sharpness_factor=2),
        T.ColorJitter(brightness=0.5, hue=0.3),
        T.GaussianBlur(kernel_size=(3, 9), sigma=(0.1, 2)),
        T.RandomPerspective(distortion_scale=0.8, p=0.1),
        T.RandomRotation(degrees=(0, 180)),
        T.RandomAffine(degrees=(30, 90), translate=(0.1, 0.3), scale=(0.5, 0.9)),
        T.ToTensor()])
    transform_y = T.Compose([T.ToTensor()])


class CreateDataset(Dataset):
    """Characterizes a dataset for PyTorch"""

    def __init__(self, X, y):
        'Initialization'
        self.X = X
        self.y = y

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.X)

    def __getitem__(self, index):
        'Generates one sample of data'
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

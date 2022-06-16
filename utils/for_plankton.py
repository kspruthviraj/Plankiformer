#
# Script to prepare the data
#
#

###########
# IMPORTS #
###########

from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from sklearn.utils import compute_class_weight
from torch.utils.data import DataLoader, Dataset


class CreateDataForPlankton:
    def __init__(self):
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
        self.class_weights = None
        self.params = None
        return

    def make_train_test_for_model(self, class_main):
        Data = pd.read_pickle(class_main.params.outpath + '/Data.pickle')
        classes = np.load(class_main.params.outpath + '/classes.npy')

        if class_main.params.balance_weight == 'yes':
            self.class_weights = pd.read_pickle(class_main.params.outpath + '/class_weights.pickle')

        if class_main.params.test_set == 'no':
            if class_main.params.ttkind == 'mixed':
                self.trainFilenames = Data[0]
                trainXimage = Data[1]
                trY = Data[2]
                trainXfeat = Data[9]
                trX = [trainXimage, trainXfeat]
            elif class_main.params.ttkind == 'image' and class_main.params.compute_extrafeat == 'yes':
                self.trainFilenames = Data[0]
                trainXimage = Data[1]
                trY = Data[2]
                trainXfeat = Data[9]
                trX = [trainXimage, trainXfeat]
            elif class_main.params.ttkind == 'feat':
                self.trainFilenames = Data[0]
                trY = Data[2]
                trainXfeat = Data[9]
                trX = [trainXfeat]
            else:
                self.trainFilenames = Data[0]
                trX = Data[1]
                trY = Data[2]

        elif class_main.params.test_set == 'yes':
            if class_main.params.valid_set == 'no':
                if class_main.params.ttkind == 'mixed':
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
                elif class_main.params.ttkind == 'image' and class_main.params.compute_extrafeat == 'yes':
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
                elif class_main.params.ttkind == 'feat':
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

            elif class_main.params.valid_set == 'yes':
                if class_main.params.ttkind == 'mixed':
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
                elif class_main.params.ttkind == 'image' and class_main.params.compute_extrafeat == 'yes':
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
                elif class_main.params.ttkind == 'feat':
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

        if class_main.params.test_set == 'no':
            y_train_max = trY.argmax(axis=1)  # The class that the classifier would bet on
            self.y_train = np.array([classes_int[y_train_max[i]] for i in range(len(y_train_max))],dtype=object)

            data_train = trX.astype(np.float64)
            data_train = 255 * data_train
            self.X_train = data_train.astype(np.uint8)

        elif class_main.params.test_set == 'yes' and class_main.params.valid_set == 'no':
            y_train_max = trY.argmax(axis=1)  # The class that the classifier would bet on
            self.y_train = np.array([classes_int[y_train_max[i]] for i in range(len(y_train_max))],dtype=object)

            y_test_max = teY.argmax(axis=1)  # The class that the classifier would bet on
            self.y_test = np.array([classes_int[y_test_max[i]] for i in range(len(y_test_max))], dtype=object)

            data_train = trX.astype(np.float64)
            data_train = 255 * data_train
            self.X_train = data_train.astype(np.uint8)

            data_test = teX.astype(np.float64)
            data_test = 255 * data_test
            self.X_test = data_test.astype(np.uint8)

        elif class_main.params.test_set == 'yes' and class_main.params.valid_set == 'yes':
            y_train_max = trY.argmax(axis=1)  # The class that the classifier would bet on
            self.y_train = np.array([classes_int[y_train_max[i]] for i in range(len(y_train_max))],dtype=object)

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

        y_integers = np.argmax(trY, axis=1)
        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_integers), y=y_integers)
        self.class_weights = torch.Tensor(class_weights)

        return

    def create_data_loaders(self, class_main):
        self.checkpoint_path = class_main.params.outpath + 'trained_models/' + class_main.params.init_name + '/'
        Path(self.checkpoint_path).mkdir(parents=True, exist_ok=True)

        train_dataset = AugmentedDataset(X=self.X_train, y=self.y_train)
        self.train_dataloader = DataLoader(train_dataset, class_main.params.batch_size, shuffle=True, num_workers=4,
                                           pin_memory=True)

        test_dataset = CreateDataset(X=self.X_test, y=self.y_test)
        self.test_dataloader = DataLoader(test_dataset, class_main.params.batch_size, shuffle=True, num_workers=4,
                                          pin_memory=True)

        val_dataset = CreateDataset(X=self.X_val, y=self.y_val)
        self.val_dataloader = DataLoader(val_dataset, class_main.params.batch_size, shuffle=True, num_workers=4,
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
        T.GaussianBlur(kernel_size=(3, 9), sigma=(0.1, 2)),
        #         T.RandomPerspective(distortion_scale=0.8, p=0.1),
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


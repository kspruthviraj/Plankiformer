###########
# IMPORTS #
###########

import pickle

import numpy as np
import torch
from joblib import dump
from sklearn.preprocessing import StandardScaler

from utils import create_data as cdata

from pathlib import Path

import numpy as np
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset


class CreateDataset:
    def __init__(self, initMode='default', verbose=True):
        self.Filenames = None
        self.classes = None
        self.Data = None
        self.initMode = initMode
        self.fsummary = None
        self = None
        self.verbose = verbose
        self.params = None
        self.paramsDict = None
        self.data = None
        self.trainSize = None
        self.testSize = None
        self.model = None
        self.opt = None

        return

    def LoadData(self, train_main):
        """
        Loads dataset using the function in the Cdata class.
        Acts differently in case it is the first time or not that the data is loaded

        The flag `training_data` is there because of how the taxonomists created the data directories. In the folders that I use for training there is an extra subfolder called `training_data`. This subfolder is absent, for example, in the validation directories.
        """

        # Default values
        datapaths = train_main.params.datapaths
        L = train_main.params.L
        class_select = train_main.params.class_select  # class_select==None has the explicit
        # meaning of selecting all the classes
        classifier = train_main.params.classifier
        compute_extrafeat = train_main.params.compute_extrafeat
        resize_images = train_main.params.resize_images
        balance_weight = train_main.params.balance_weight
        datakind = train_main.params.datakind
        training_data = train_main.params.training_data

        # Initialize or Load Data Structure
        if self.data is None:
            self.data = cdata.Cdata(datapaths, L, class_select, classifier, compute_extrafeat, resize_images,
                                    balance_weight, datakind, training_data=training_data)
        else:
            self.data.Load(datapaths, L, class_select, classifier, compute_extrafeat, resize_images, balance_weight,
                           datakind, training_data=training_data)

        return

    def CreateTrainTestSets(self, train_main, ttkind=None, classifier=None, save_data=None, balance_weight=None,
                            testSplit=None,
                            valid_set=None, test_set=None, compute_extrafeat=None, random_state=12345):
        """
        Creates train and test sets using the CtrainTestSet class
        """

        # Set default value for ttkind
        if ttkind is None:
            ttkind = train_main.params.ttkind
        else:
            self.params.ttkind = ttkind

        # Set default value for testSplit
        if testSplit is None:
            testSplit = train_main.params.testSplit
        else:
            self.params.testSplit = testSplit

        if valid_set is None:
            valid_set = train_main.params.valid_set
        else:
            self.params.valid_set = valid_set

        if test_set is None:
            test_set = train_main.params.test_set
        else:
            self.params.test_set = test_set

        if classifier is None:
            classifier = train_main.params.classifier

        if save_data is None:
            save_data = train_main.params.save_data

        if balance_weight is None:
            balance_weight = train_main.params.balance_weight

        if compute_extrafeat is None:
            compute_extrafeat = train_main.params.compute_extrafeat

        self = cdata.CTrainTestSet(self.data.X, self.data.y, self.data.filenames,
                                      ttkind=ttkind, classifier=classifier, balance_weight=balance_weight,
                                      testSplit=testSplit, valid_set=valid_set, test_set=test_set,
                                      compute_extrafeat=compute_extrafeat, random_state=random_state)

        # To save the data
        if train_main.params.ttkind == 'mixed':
            scaler = StandardScaler()
            scaler.fit(self.trainXfeat)
            dump(scaler, train_main.params.outpath + '/Features_scaler_used_for_MLP.joblib')
            if train_main.params.test_set == 'yes':
                self.trainXfeat = scaler.transform(self.trainXfeat)
                self.testXfeat = scaler.transform(self.testXfeat)
                if train_main.params.valid_set == 'yes':
                    self.valXfeat = scaler.transform(self.valXfeat)
                    self.Data = [self.trainFilenames, self.trainXimage, self.trainY,
                                 self.testFilenames, self.testXimage, self.testY,
                                 self.valFilenames, self.valXimage, self.valY,
                                 self.trainXfeat, self.testXfeat, self.valXfeat]
                elif train_main.params.valid_set == 'no':
                    self.Data = [self.trainFilenames, self.trainXimage, self.trainY,
                                 self.testFilenames, self.testXimage, self.testY,
                                 [], [], [],
                                 self.trainXfeat, self.testXfeat, []]
            elif train_main.params.test_set == 'no':
                self.trainXfeat = scaler.transform(self.trainXfeat)
                self.Data = [self.trainFilenames, self.trainXimage, self.trainY,
                             [], [], [],
                             [], [], [],
                             self.trainXfeat, [], []]

        elif train_main.params.ttkind == 'feat':
            scaler = StandardScaler()
            scaler.fit(self.trainX)
            dump(scaler, self.params.outpath + '/Features_scaler_used_for_MLP.joblib')
            if train_main.params.test_set == 'yes':
                self.trainX = scaler.transform(self.trainX)
                self.testX = scaler.transform(self.testX)
                if train_main.params.valid_set == 'yes':
                    self.valXfeat = scaler.transform(self.valXfeat)
                    self.Data = [self.trainFilenames, [], self.trainY,
                                 self.testFilenames, [], self.testY,
                                 self.valFilenames, [], self.valY,
                                 self.trainX, self.testX, self.valX]
                elif train_main.params.valid_set == 'no':
                    self.Data = [self.trainFilenames, [], self.trainY,
                                 self.testFilenames, [], self.testY,
                                 [], [], [],
                                 self.trainX, self.testX, []]

            elif train_main.params.test_set == 'no':
                self.trainX = scaler.transform(self.trainX)
                self.Data = [self.trainFilenames, [], self.trainY,
                             [], [], [],
                             [], [], [],
                             self.trainX, [], []]

        elif train_main.params.ttkind == 'image' and train_main.params.compute_extrafeat == 'no':
            if train_main.params.test_set == 'yes':
                if train_main.params.valid_set == 'yes':
                    self.Data = [self.trainFilenames, self.trainX, self.trainY,
                                 self.testFilenames, self.testX, self.testY,
                                 self.valFilenames, self.valX, self.valY,
                                 [], [], []]
                elif train_main.params.valid_set == 'no':
                    self.Data = [self.trainFilenames, self.trainX, self.trainY,
                                 self.testFilenames, self.testX, self.testY,
                                 [], [], [],
                                 [], [], []]
            elif train_main.params.test_set == 'no':
                self.Data = [self.trainFilenames, self.trainX, self.trainY,
                             [], [], [],
                             [], [], [],
                             [], [], []]

        elif train_main.params.ttkind == 'image' and train_main.params.compute_extrafeat == 'yes':
            scaler = StandardScaler()
            scaler.fit(self.trainXfeat)
            dump(scaler, self.params.outpath + '/Aqua_Features_scaler_used_for_MLP.joblib')
            if train_main.params.test_set == 'yes':
                self.trainXfeat = scaler.transform(self.trainXfeat)
                self.testXfeat = scaler.transform(self.testXfeat)
                if self.params.valid_set == 'yes':
                    self.valXfeat = scaler.transform(self.valXfeat)

                    self.Data = [self.trainFilenames, self.trainXimage, self.trainY,
                                 self.testFilenames, self.testXimage, self.testY,
                                 self.valFilenames, self.valXimage, self.valY,
                                 self.trainXfeat, self.testXfeat, self.valXfeat]
                elif train_main.params.valid_set == 'no':
                    self.Data = [self.trainFilenames, self.trainXimage, self.trainY,
                                 self.testFilenames, self.testXimage, self.testY,
                                 [], [], [],
                                 self.trainXfeat, self.testXfeat, []]
            elif train_main.params.test_set == 'no':
                self.trainXfeat = scaler.transform(self.trainXfeat)
                self.Data = [self.trainFilenames, self.trainXimage, self.trainY,
                             [], [], [],
                             [], [], [],
                             self.trainXfeat, [], []]
        else:
            print("Set the right data type")

        if train_main.params.save_data == 'yes':
            with open(train_main.params.outpath + '/Data.pickle', 'wb') as a:
                pickle.dump(self.Data, a)

        if train_main.params.balance_weight == 'yes':
            with open(train_main.params.outpath + '/class_weights.pickle', 'wb') as cw:
                pickle.dump(self.class_weights, cw)
            torch.save(self.class_weights_tensor, train_main.params.outpath + '/class_weights_tensor.pt')

        # To Save classes and filenames
        np.save(train_main.params.outpath + '/classes.npy', self.lb.classes_)
        classes = self.lb.classes_
        self.classes = classes

        if train_main.params.test_set == 'yes' and train_main.params.valid_set == 'yes':
            self.Filenames = [self.trainFilenames, self.testFilenames, self.valFilenames]
        elif train_main.params.test_set == 'yes' and train_main.params.valid_set == 'no':
            self.Filenames = [self.trainFilenames, self.testFilenames]
        elif train_main.params.test_set == 'no':
            self.Filenames = [self.trainFilenames]
        else:
            self.Filenames = ['']
        with open(train_main.params.outpath + '/Files_used_for_training_testing.pickle', 'wb') as b:
            pickle.dump(self.Filenames, b)

        return


class CreateTrainTestDataloader:
    def __init__(self):
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

    def make_train_test_for_model(self, train_main):
        self.class_weights_tensor = self.class_weights_tensor
        self.Filenames = self.Filenames
        classes = self.classes
        Data = self.Data

        # Data = pd.read_pickle(train_main.params.outpath + '/Data.pickle')
        # classes = np.load(train_main.params.outpath + '/classes.npy')

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

            test_dataset = CreateDataloader(X=self.X_test, y=self.y_test)
            self.test_dataloader = DataLoader(test_dataset, train_main.params.batch_size, shuffle=True, num_workers=4,
                                              pin_memory=True)

            val_dataset = CreateDataloader(X=self.X_val, y=self.y_val)
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

            test_dataset = CreateDataloader(X=self.X_test, y=self.y_test)
            self.test_dataloader = DataLoader(test_dataset, train_main.params.batch_size, shuffle=True, num_workers=4,
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


class CreateDataloader(Dataset):
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


###########
# IMPORTS #
###########

import pickle

import numpy as np
import torch
from joblib import dump
from sklearn.preprocessing import StandardScaler

from utils import create_data as cdata


class CreateDataset:
    def __init__(self, initMode='default', verbose=True):
        self.class_weights_tensor = None
        self.Filenames_val = None
        self.Filenames_test = None
        self.Filenames_train = None
        self.tt2 = None
        self.tt1 = None
        self.data2 = None
        self.data1 = None
        self.Filenames = None
        self.classes = None
        self.Data = None
        self.initMode = initMode
        self.fsummary = None
        self.tt = None
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

    def LoadData_for_others(self, train_main):
        """
        Loads dataset using the function in the Cdata class.
        Acts differently in case it is the first time or not that the data is loaded

        The flag `training_data` is there because of how the taxonomists created the data directories. In the folders that I use for training there is an extra subfolder called `training_data`. This subfolder is absent, for example, in the validation directories.
        """

        # Default values
        train_PATH = train_main.params.datapaths
        testpath = train_main.params.test_path

        train_PATH = ' '.join(map(str, train_PATH))
        testpath = ' '.join(map(str, testpath))

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
        if self.data1 is None:
            self.data1 = cdata.Cdata(testpath, L, class_select, classifier, compute_extrafeat, resize_images,
                                     balance_weight, datakind, training_data=training_data)
        else:
            self.data1.Load(testpath, L, class_select, classifier, compute_extrafeat, resize_images, balance_weight,
                            datakind, training_data=training_data)

        # Initialize or Load Data Structure
        if self.data2 is None:
            self.data2 = cdata.Cdata(train_PATH, L, class_select, classifier, compute_extrafeat, resize_images,
                                     balance_weight, datakind, training_data=training_data)
        else:
            self.data2.Load(train_PATH, L, class_select, classifier, compute_extrafeat, resize_images, balance_weight,
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

        self.tt = cdata.CTrainTestSet(self.data.X, self.data.y, self.data.filenames,
                                      ttkind=ttkind, classifier=classifier, balance_weight=balance_weight,
                                      testSplit=testSplit, valid_set=valid_set, test_set=test_set,
                                      compute_extrafeat=compute_extrafeat, random_state=random_state)

        # To save the data
        if train_main.params.ttkind == 'mixed':
            scaler = StandardScaler()
            scaler.fit(self.tt.trainXfeat)
            dump(scaler, train_main.params.outpath + '/Features_scaler_used_for_MLP.joblib')
            if test_set == 'yes':
                self.tt.trainXfeat = scaler.transform(self.tt.trainXfeat)
                self.tt.testXfeat = scaler.transform(self.tt.testXfeat)
                if train_main.params.valid_set == 'yes':
                    self.tt.valXfeat = scaler.transform(self.tt.valXfeat)
                    self.Data = [self.tt.trainFilenames, self.tt.trainXimage, self.tt.trainY,
                                 self.tt.testFilenames, self.tt.testXimage, self.tt.testY,
                                 self.tt.valFilenames, self.tt.valXimage, self.tt.valY,
                                 self.tt.trainXfeat, self.tt.testXfeat, self.tt.valXfeat]
                elif train_main.params.valid_set == 'no':
                    self.Data = [self.tt.trainFilenames, self.tt.trainXimage, self.tt.trainY,
                                 self.tt.testFilenames, self.tt.testXimage, self.tt.testY,
                                 [], [], [],
                                 self.tt.trainXfeat, self.tt.testXfeat, []]
            elif test_set == 'no':
                self.tt.trainXfeat = scaler.transform(self.tt.trainXfeat)
                self.Data = [self.tt.trainFilenames, self.tt.trainXimage, self.tt.trainY,
                             [], [], [],
                             [], [], [],
                             self.tt.trainXfeat, [], []]

        elif train_main.params.ttkind == 'feat':
            scaler = StandardScaler()
            scaler.fit(self.tt.trainX)
            dump(scaler, self.params.outpath + '/Features_scaler_used_for_MLP.joblib')
            if test_set == 'yes':
                self.tt.trainX = scaler.transform(self.tt.trainX)
                self.tt.testX = scaler.transform(self.tt.testX)
                if train_main.params.valid_set == 'yes':
                    self.tt.valXfeat = scaler.transform(self.tt.valXfeat)
                    self.Data = [self.tt.trainFilenames, [], self.tt.trainY,
                                 self.tt.testFilenames, [], self.tt.testY,
                                 self.tt.valFilenames, [], self.tt.valY,
                                 self.tt.trainX, self.tt.testX, self.tt.valX]
                elif train_main.params.valid_set == 'no':
                    self.Data = [self.tt.trainFilenames, [], self.tt.trainY,
                                 self.tt.testFilenames, [], self.tt.testY,
                                 [], [], [],
                                 self.tt.trainX, self.tt.testX, []]

            elif test_set == 'no':
                self.tt.trainX = scaler.transform(self.tt.trainX)
                self.Data = [self.tt.trainFilenames, [], self.tt.trainY,
                             [], [], [],
                             [], [], [],
                             self.tt.trainX, [], []]

        elif train_main.params.ttkind == 'image' and train_main.params.compute_extrafeat == 'no':
            if test_set == 'yes':
                if train_main.params.valid_set == 'yes':
                    self.Data = [self.tt.trainFilenames, self.tt.trainX, self.tt.trainY,
                                 self.tt.testFilenames, self.tt.testX, self.tt.testY,
                                 self.tt.valFilenames, self.tt.valX, self.tt.valY,
                                 [], [], []]
                elif train_main.params.valid_set == 'no':
                    self.Data = [self.tt.trainFilenames, self.tt.trainX, self.tt.trainY,
                                 self.tt.testFilenames, self.tt.testX, self.tt.testY,
                                 [], [], [],
                                 [], [], []]
            elif test_set == 'no':
                self.Data = [self.tt.trainFilenames, self.tt.trainX, self.tt.trainY,
                             [], [], [],
                             [], [], [],
                             [], [], []]

        elif train_main.params.ttkind == 'image' and train_main.params.compute_extrafeat == 'yes':
            scaler = StandardScaler()
            scaler.fit(self.tt.trainXfeat)
            dump(scaler, self.params.outpath + '/Aqua_Features_scaler_used_for_MLP.joblib')
            if test_set == 'yes':
                self.tt.trainXfeat = scaler.transform(self.tt.trainXfeat)
                self.tt.testXfeat = scaler.transform(self.tt.testXfeat)
                if self.params.valid_set == 'yes':
                    self.tt.valXfeat = scaler.transform(self.tt.valXfeat)

                    self.Data = [self.tt.trainFilenames, self.tt.trainXimage, self.tt.trainY,
                                 self.tt.testFilenames, self.tt.testXimage, self.tt.testY,
                                 self.tt.valFilenames, self.tt.valXimage, self.tt.valY,
                                 self.tt.trainXfeat, self.tt.testXfeat, self.tt.valXfeat]
                elif train_main.params.valid_set == 'no':
                    self.Data = [self.tt.trainFilenames, self.tt.trainXimage, self.tt.trainY,
                                 self.tt.testFilenames, self.tt.testXimage, self.tt.testY,
                                 [], [], [],
                                 self.tt.trainXfeat, self.tt.testXfeat, []]
            elif test_set == 'no':
                self.tt.trainXfeat = scaler.transform(self.tt.trainXfeat)
                self.Data = [self.tt.trainFilenames, self.tt.trainXimage, self.tt.trainY,
                             [], [], [],
                             [], [], [],
                             self.tt.trainXfeat, [], []]
        else:
            print("Set the right data type")

        if train_main.params.save_data == 'yes':
            with open(train_main.params.outpath + '/Data.pickle', 'wb') as a:
                pickle.dump(self.Data, a)

        if train_main.params.balance_weight == 'yes':
            with open(train_main.params.outpath + '/class_weights.pickle', 'wb') as cw:
                pickle.dump(self.tt.class_weights, cw)
            torch.save(self.tt.class_weights_tensor, train_main.params.outpath + '/class_weights_tensor.pt')

        # To Save classes and filenames
        np.save(train_main.params.outpath + '/classes.npy', self.tt.lb.classes_)
        classes = self.tt.lb.classes_
        self.classes = classes

        if test_set == 'yes' and train_main.params.valid_set == 'yes':
            self.Filenames = [self.tt.trainFilenames, self.tt.testFilenames, self.tt.valFilenames]
        elif test_set == 'yes' and train_main.params.valid_set == 'no':
            self.Filenames = [self.tt.trainFilenames, self.tt.testFilenames]
        elif test_set == 'no':
            self.Filenames = [self.tt.trainFilenames]
        else:
            self.Filenames = ['']
        with open(train_main.params.outpath + '/Files_used_for_training_testing.pickle', 'wb') as b:
            pickle.dump(self.Filenames, b)

        return

    def CreatedataSetsForOthers(self, train_main, ttkind=None, classifier=None, balance_weight=None,
                                compute_extrafeat=None):
        """
        Creates train and test sets using the CtrainTestSet class
        """

        # Set default value for ttkind
        if ttkind is None:
            ttkind = train_main.params.ttkind
        else:
            self.params.ttkind = ttkind

        # Set default value for testSplit

        if classifier is None:
            classifier = train_main.params.classifier

        if balance_weight is None:
            balance_weight = train_main.params.balance_weight

        if compute_extrafeat is None:
            compute_extrafeat = train_main.params.compute_extrafeat

        self.tt1 = cdata.CTrainTestSet(self.data1.X, self.data1.y, self.data1.filenames,
                                       ttkind=ttkind, classifier=classifier, balance_weight=balance_weight,
                                       testSplit=0.2, valid_set='no', test_set='yes',
                                       compute_extrafeat=compute_extrafeat)

        self.tt2 = cdata.CTrainTestSet(self.data2.X, self.data2.y, self.data2.filenames,
                                       ttkind=ttkind, classifier=classifier, balance_weight=balance_weight,
                                       testSplit=0, valid_set='no', test_set='no',
                                       compute_extrafeat=compute_extrafeat)

        if train_main.params.ttkind == 'image' and train_main.params.compute_extrafeat == 'no':
            self.Data = [self.tt1.trainFilenames, self.tt1.trainX, self.tt1.trainY,
                         self.tt1.testFilenames, self.tt1.testX, self.tt1.testY,
                         self.tt2.trainFilenames, self.tt2.trainX, self.tt2.trainY,
                         [], [], []]

        else:
            print("Set the right data type")

        self.Filenames_train = [self.tt1.trainFilenames]
        self.Filenames_val = [self.tt1.testFilenames]
        self.Filenames_test = [self.tt2.trainFilenames]

        if train_main.params.save_data == 'yes':
            with open(train_main.params.outpath + '/Data.pickle', 'wb') as a:
                pickle.dump(self.Data, a)

        if train_main.params.balance_weight == 'yes':
            with open(train_main.params.outpath + '/class_weights.pickle', 'wb') as cw:
                pickle.dump(self.tt1.class_weights, cw)
            torch.save(self.tt1.class_weights_tensor, train_main.params.outpath + '/class_weights_tensor.pt')
        self.class_weights_tensor = self.tt1.class_weights_tensor

        # To Save classes and filenames
        np.save(train_main.params.outpath + '/classes.npy', self.tt1.lb.classes_)
        classes = self.tt1.lb.classes_
        self.classes = classes

        self.Filenames = [self.tt1.trainFilenames, self.tt1.testFilenames, self.tt2.trainFilenames]

        with open(train_main.params.outpath + '/Files_used_for_train_val_test.pickle', 'wb') as b:
            pickle.dump(self.Filenames, b)

        return

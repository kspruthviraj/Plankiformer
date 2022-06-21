###########
# IMPORTS #
###########

import pickle

import joblib
import numpy as np
from joblib import dump
from sklearn.preprocessing import StandardScaler

from utils import create_data as cdata
from utils.create_data import DropCols, RemoveUselessCols


class CreateDataset:
    def __init__(self, initMode='default', verbose=True):
        self.valid_set = None
        self.test_set = None
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

    def LoadData(self, class_main, pred_main):
        """
        Loads dataset using the function in the Cdata class.
        Acts differently in case it is the first time or not that the data is loaded

        The flag `training_data` is there because of how the taxonomists created the data directories. In the folders that I use for training there is an extra subfolder called `training_data`. This subfolder is absent, for example, in the validation directories.
        """

        # Default values
        datapaths = pred_main.params.datapaths
        L = class_main.params.L
        class_select = class_main.params.class_select  # class_select==None has the explicit
        # meaning of selecting all the classes
        classifier = class_main.params.classifier
        compute_extrafeat = class_main.params.compute_extrafeat
        resize_images = class_main.params.resize_images
        balance_weight = class_main.params.balance_weight
        datakind = class_main.params.datakind
        training_data = class_main.params.training_data

        # Initialize or Load Data Structure
        if self.data is None:
            self.data = cdata.Cdata(datapaths, L, class_select, classifier, compute_extrafeat, resize_images,
                                    balance_weight, datakind, training_data=training_data)
        else:
            self.data.Load(datapaths, L, class_select, classifier, compute_extrafeat, resize_images, balance_weight,
                           datakind, training_data=training_data)

        return

    def CreateTrainTestSets(self, class_main, ttkind=None, classifier=None, save_data=None, balance_weight=None,
                            testSplit=None, valid_set=None, test_set=None, compute_extrafeat=None, random_state=12345):
        """
        Creates train and test sets using the CtrainTestSet class
        """

        # Set default value for ttkind
        if ttkind is None:
            ttkind = class_main.params.ttkind
        else:
            self.params.ttkind = ttkind

        # Set default value for testSplit
        testSplit = 0

        self.valid_set = 'no'
        self.test_set = 'no'

        if classifier is None:
            classifier = class_main.params.classifier

        if balance_weight is None:
            balance_weight = class_main.params.balance_weight

        if compute_extrafeat is None:
            compute_extrafeat = class_main.params.compute_extrafeat

        self.tt = cdata.CTrainTestSet(self.data.X, self.data.y, self.data.filenames,
                                      ttkind=ttkind, classifier=classifier, balance_weight=balance_weight,
                                      testSplit=testSplit, valid_set=valid_set, test_set=self.test_set,
                                      compute_extrafeat=compute_extrafeat, random_state=random_state)

        # To store the data
        if class_main.params.ttkind == 'mixed':
            scaler = joblib.load(class_main.params.outpath + '/Features_scaler_used_for_MLP.joblib')
            self.tt.trainXfeat = scaler.transform(self.tt.trainXfeat)
            self.Data = [self.tt.trainFilenames, self.tt.trainXimage, self.tt.trainY,
                         [], [], [],
                         [], [], [],
                         self.tt.trainXfeat, [], []]

        elif class_main.params.ttkind == 'feat':
            scaler = joblib.load(class_main.params.outpath + '/Features_scaler_used_for_MLP.joblib')
            self.tt.trainX = scaler.transform(self.tt.trainX)
            self.Data = [self.tt.trainFilenames, [], self.tt.trainY,
                         [], [], [],
                         [], [], [],
                         self.tt.trainX, [], []]

        elif class_main.params.ttkind == 'image' and class_main.params.compute_extrafeat == 'no':
            self.Data = [self.tt.trainFilenames, self.tt.trainX, self.tt.trainY,
                         [], [], [],
                         [], [], [],
                         [], [], []]

        elif class_main.params.ttkind == 'image' and class_main.params.compute_extrafeat == 'yes':
            scaler = joblib.load(class_main.params.outpath + '/Features_scaler_used_for_MLP.joblib')
            self.tt.trainXfeat = scaler.transform(self.tt.trainXfeat)
            self.Data = [self.tt.trainFilenames, self.tt.trainXimage, self.tt.trainY,
                         [], [], [],
                         [], [], [],
                         self.tt.trainXfeat, [], []]
        else:
            print("Set the right data type")

        return

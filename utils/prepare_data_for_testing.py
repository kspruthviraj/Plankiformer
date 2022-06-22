###########
# IMPORTS #
###########

import joblib

from utils import create_data as cdata


class CreateDataset:
    def __init__(self, initMode='default', verbose=True):
        self.Filenames = None
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

    def LoadData(self, test_main, train_main):
        """
        Loads dataset using the function in the Cdata class.
        Acts differently in case it is the first time or not that the data is loaded

        The flag `training_data` is there because of how the taxonomists created the data directories. In the folders that I use for training there is an extra subfolder called `training_data`. This subfolder is absent, for example, in the validation directories.
        """

        # Default values
        testpath = test_main.params.test_path
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
            self.data = cdata.Cdata(testpath, L, class_select, classifier, compute_extrafeat, resize_images,
                                    balance_weight, datakind, training_data=training_data)
        else:
            self.data.Load(testpath, L, class_select, classifier, compute_extrafeat, resize_images, balance_weight,
                           datakind, training_data=training_data)

        return

    def CreateTrainTestSets(self, train_main, ttkind=None, classifier=None, balance_weight=None,
                            valid_set=None, compute_extrafeat=None, random_state=12345):
        """
        Creates train and test sets using the CtrainTestSet class
        """

        # Set default value for ttkind
        if ttkind is None:
            ttkind = train_main.params.ttkind
        else:
            self.params.ttkind = ttkind

        # Set default value for testSplit
        testSplit = 0

        self.valid_set = 'no'
        self.test_set = 'no'

        if classifier is None:
            classifier = train_main.params.classifier

        if balance_weight is None:
            balance_weight = train_main.params.balance_weight

        if compute_extrafeat is None:
            compute_extrafeat = train_main.params.compute_extrafeat

        self.tt = cdata.CTrainTestSet(self.data.X, self.data.y, self.data.filenames,
                                      ttkind=ttkind, classifier=classifier, balance_weight=balance_weight,
                                      testSplit=testSplit, valid_set=valid_set, test_set=self.test_set,
                                      compute_extrafeat=compute_extrafeat, random_state=random_state)

        # To store the data
        if train_main.params.ttkind == 'mixed':
            scaler = joblib.load(train_main.params.outpath + '/Features_scaler_used_for_MLP.joblib')
            self.tt.trainXfeat = scaler.transform(self.tt.trainXfeat)
            self.Data = [self.tt.trainFilenames, self.tt.trainXimage, self.tt.trainY,
                         [], [], [],
                         [], [], [],
                         self.tt.trainXfeat, [], []]

        elif train_main.params.ttkind == 'feat':
            scaler = joblib.load(train_main.params.outpath + '/Features_scaler_used_for_MLP.joblib')
            self.tt.trainX = scaler.transform(self.tt.trainX)
            self.Data = [self.tt.trainFilenames, [], self.tt.trainY,
                         [], [], [],
                         [], [], [],
                         self.tt.trainX, [], []]

        elif train_main.params.ttkind == 'image' and train_main.params.compute_extrafeat == 'no':
            self.Data = [self.tt.trainFilenames, self.tt.trainX, self.tt.trainY,
                         [], [], [],
                         [], [], [],
                         [], [], []]

        elif train_main.params.ttkind == 'image' and train_main.params.compute_extrafeat == 'yes':
            scaler = joblib.load(train_main.params.outpath + '/Features_scaler_used_for_MLP.joblib')
            self.tt.trainXfeat = scaler.transform(self.tt.trainXfeat)
            self.Data = [self.tt.trainFilenames, self.tt.trainXimage, self.tt.trainY,
                         [], [], [],
                         [], [], [],
                         self.tt.trainXfeat, [], []]
        else:
            print("Set the right data type")

        self.Filenames = [self.tt.trainFilenames]
        
        return

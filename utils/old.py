#
# Script to prepare the data
#
#

###########
# IMPORTS #
###########

import argparse
import numpy as np
import pathlib
import pickle
import sys

from joblib import dump
from sklearn.preprocessing import StandardScaler
from utils import create_data as cdata


###############
# TRAIN CLASS #
###############

def ArgsCheck(args):
    """ Consistency checks for command line arguments """

    if args.ttkind != 'image' and args.aug == True:
        print('User asked for data augmentation, but we set it to False, because we only do it for `image` models')
        args.aug = False

    if args.ttkind == 'image':
        args.compute_extrafeat = 'no'
        print(
            'User asked for computing extra features, but we set it to False, because we only do it for `mixed` '
            'models')

    return


class PrepareData:
    def __init__(self, initMode='default', verbose=True):
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
        self.SetParameters(mode=initMode)

        return

    def SetParameters(self, mode='default'):
        """ default, from args"""
        if mode == 'default':
            self.ReadArgs(string=None)
        elif mode == 'args':
            self.ReadArgs(string=sys.argv[1:])
        else:
            print('Unknown parameter mode', mode)
            raise NotImplementedError
        return

    def ReadArgs(self, string=None):
        if string is None:
            string = ""

        parser = argparse.ArgumentParser(description='Create Dataset')

        parser.add_argument('-datapaths', nargs='*',
                            default=['./data/1_zooplankton_0p5x/training/zooplankton_trainingset_2020.04.28/'],
                            help="Directories with the data.")
        parser.add_argument('-outpath', default='./out/', help="directory where you want the output saved")

        parser.add_argument('-aug', action='store_true',
                            help="Perform data augmentation. Augmentation parameters are hard-coded.")
        # Data
        parser.add_argument('-L', type=int, default=128, help="Images are resized to a square of LxL pixels")
        parser.add_argument('-testSplit', type=float, default=0.2, help="Fraction of examples in the test set")
        parser.add_argument('-class_select', nargs='*', default=None,
                            help='List of classes to be looked at (put the class names '
                                 'one by one, separated by spaces).'
                                 ' If None, all available classes are studied.')
        parser.add_argument('-classifier', choices=['binary', 'multi', 'versusall'], default='multi',
                            help='Choose between "binary","multi","versusall" classifier')
        parser.add_argument('-balance_weight', choices=['yes', 'no'], default='no',
                            help='Choose "yes" or "no" for balancing class weights for imbalance classes')
        parser.add_argument('-datakind', choices=['mixed', 'feat', 'image'], default=None,
                            help="Which data to load: features, images, or both")
        parser.add_argument('-ttkind', choices=['mixed', 'feat', 'image'], default=None,
                            help="Which data to use in the test and training sets: features, images, or both")
        parser.add_argument('-training_data', choices=['True', 'False'], default='True',
                            help="This is to cope with the different directory structures")

        # Preprocessing Images
        parser.add_argument('-resize_images', type=int, default=1,
                            help="Images are resized to a square of LxL pixels by keeping the initial image proportions if resize=1. If resize=2, then the proportions are not kept but resized to match the user defined dimension")

        parser.add_argument('-save_data', choices=['yes', 'no'], default=None,
                            help="Whether to save the data for later use or not")
        parser.add_argument('-compute_extrafeat', choices=['yes', 'no'], default=None,
                            help="Whether to compute extra features or not")
        parser.add_argument('-valid_set', choices=['yes', 'no'], default='no',
                            help="Select to have validation set. Choose from Yes or No")

        args = parser.parse_args(string)

        # Add a trailing / to the paths, just for safety
        for i, elem in enumerate(args.datapaths):
            args.datapaths[i] = elem + '/'

        args.outpath = args.outpath + '/'
        args.training_data = True if args.training_data == 'True' else False

        ArgsCheck(args)
        self.params = args

        if self.verbose:
            print(args)

        return

    def CreateOutDir(self):
        """ Create a unique output directory, and put inside it a file with the simulation parameters """
        pathlib.Path(self.params.outpath).mkdir(parents=True, exist_ok=True)
        self.WriteParams()
        return

    def WriteParams(self):
        """ Writes a txt file with the simulation parameters """
        self.fsummary = open(self.params.outpath + '/params.txt', 'w')
        print(self.params, file=self.fsummary);
        self.fsummary.flush()

        ''' Writes the same simulation parameters in binary '''
        np.save(self.params.outpath + '/params.npy', self.params)
        return

    def UpdateParams(self, **kwargs):
        """ Updates the parameters given in kwargs, and updates params.txt"""
        self.paramsDict = vars(self.params)
        if kwargs is not None:
            for key, value in kwargs.items():
                self.paramsDict[key] = value
        self.CreateOutDir()
        self.WriteParams()

        return

    def LoadData(self, datapaths=None, L=None, class_select=-1, classifier=None, compute_extrafeat=None,
                 resize_images=None, balance_weight=None, datakind=None, training_data=None):
        """
        Loads dataset using the function in the Cdata class.
        Acts differently in case it is the first time or not that the data is loaded

        The flag `training_data` is there because of how the taxonomists created the data directories. In the folders that I use for training there is an extra subfolder called `training_data`. This subfolder is absent, for example, in the validation directories.
        """

        # Default values
        if datapaths is None:    datapaths = self.params.datapaths
        if L is None:             L = self.params.L
        if class_select == -1:      class_select = self.params.class_select  # class_select==None has the explicit
        # meaning of selecting all the classes
        if classifier is None:      classifier = self.params.classifier
        if compute_extrafeat is None:      compute_extrafeat = self.params.compute_extrafeat
        if resize_images is None:      resize_images = self.params.resize_images
        if balance_weight is None:      balance_weight = self.params.balance_weight
        if datakind is None:      datakind = self.params.datakind
        if training_data is None: training_data = self.params.training_data

        # Initialize or Load Data Structure
        if self.data is None:
            self.data = cdata.Cdata(datapaths, L, class_select, classifier, compute_extrafeat, resize_images,
                                    balance_weight, datakind, training_data=training_data)
        else:
            self.data.Load(datapaths, L, class_select, classifier, compute_extrafeat, resize_images, balance_weight,
                           datakind, training_data=training_data)

        # Reset parameters
        self.params.datapaths = self.data.datapath
        self.params.L = self.data.L
        self.params.class_select = self.data.class_select
        self.params.datakind = self.data.kind
        self.params.classifier = self.data.classifier
        self.params.compute_extrafeat = self.data.compute_extrafeat
        self.params.balance_weight = self.data.balance_weight
        return

    def CreateTrainTestSets(self, ttkind=None, classifier=None, save_data=None, balance_weight=None, testSplit=None,
                            valid_set=None, compute_extrafeat=None, random_state=12345):
        """
        Creates train and test sets using the CtrainTestSet class
        """

        # Set default value for ttkind
        if ttkind is None:
            ttkind = self.params.ttkind
        else:
            self.params.ttkind = ttkind

        # Set default value for testSplit
        if testSplit is None:
            testSplit = self.params.testSplit
        else:
            self.params.testSplit = testSplit

        if valid_set is None:
            valid_set = self.params.valid_set
        else:
            self.params.valid_set = valid_set

        if classifier is None:
            classifier = self.params.classifier

        if save_data is None:
            save_data = self.params.save_data

        if balance_weight is None:
            balance_weight = self.params.balance_weight

        if compute_extrafeat is None:
            compute_extrafeat = self.params.compute_extrafeat

        self.tt = cdata.CTrainTestSet(self.data.X, self.data.y, self.data.filenames,
                                      ttkind=ttkind, classifier=classifier, balance_weight=balance_weight,
                                      testSplit=testSplit, valid_set=valid_set,
                                      compute_extrafeat=compute_extrafeat, random_state=random_state)
        self.params.ttkind = self.tt.ttkind

        # To save the data
        if self.params.ttkind == 'mixed':
            scaler = StandardScaler()
            scaler.fit(self.tt.trainXfeat)
            dump(scaler, self.params.outpath + '/Features_scaler_used_for_MLP.joblib')
            self.tt.trainXfeat = scaler.transform(self.tt.trainXfeat)
            self.tt.testXfeat = scaler.transform(self.tt.testXfeat)
            if self.params.valid_set == 'yes':
                self.tt.valXfeat = scaler.transform(self.tt.valXfeat)
                Data = [self.tt.trainFilenames, self.tt.trainXimage, self.tt.trainY,
                        self.tt.testFilenames, self.tt.testXimage, self.tt.testY,
                        self.tt.valFilenames, self.tt.valXimage, self.tt.valY,
                        self.tt.trainXfeat, self.tt.testXfeat, self.tt.valXfeat]
            elif self.params.valid_set == 'no':
                Data = [self.tt.trainFilenames, self.tt.trainXimage, self.tt.trainY,
                        self.tt.testFilenames, self.tt.testXimage, self.tt.testY,
                        self.tt.trainXfeat, self.tt.testXfeat]

        elif self.params.ttkind == 'feat':
            scaler = StandardScaler()
            scaler.fit(self.tt.trainX)
            dump(scaler, self.params.outpath + '/Features_scaler_used_for_MLP.joblib')
            self.tt.trainX = scaler.transform(self.tt.trainX)
            self.tt.testX = scaler.transform(self.tt.testX)

            if self.params.valid_set == 'yes':
                self.tt.valXfeat = scaler.transform(self.tt.valXfeat)
                Data = [self.tt.trainFilenames, self.tt.trainX, self.tt.trainY,
                        self.tt.testFilenames, self.tt.testX, self.tt.testY,
                        self.tt.valFilenames, self.tt.valX, self.tt.valY]
            elif self.params.valid_set == 'no':
                Data = [self.tt.trainFilenames, self.tt.trainX, self.tt.trainY,
                        self.tt.testFilenames, self.tt.testX, self.tt.testY]

        elif self.params.ttkind == 'image' and self.params.compute_extrafeat == 'no':
            if self.params.valid_set == 'yes':
                Data = [self.tt.trainFilenames, self.tt.trainX, self.tt.trainY,
                        self.tt.testFilenames, self.tt.testX, self.tt.testY,
                        self.tt.valFilenames, self.tt.valX, self.tt.valY]
            elif self.params.valid_set == 'no':
                Data = [self.tt.trainFilenames, self.tt.trainX, self.tt.trainY,
                        self.tt.testFilenames, self.tt.testX, self.tt.testY]

        elif self.params.ttkind == 'image' and self.params.compute_extrafeat == 'yes':
            scaler = StandardScaler()
            scaler.fit(self.tt.trainXfeat)
            dump(scaler, self.params.outpath + '/Aqua_Features_scaler_used_for_MLP.joblib')
            self.tt.trainXfeat = scaler.transform(self.tt.trainXfeat)
            self.tt.testXfeat = scaler.transform(self.tt.testXfeat)
            if self.params.valid_set == 'yes':
                self.tt.valXfeat = scaler.transform(self.tt.valXfeat)

                Data = [self.tt.trainFilenames, self.tt.trainXimage, self.tt.trainY,
                        self.tt.testFilenames, self.tt.testXimage, self.tt.testY,
                        self.tt.valFilenames, self.tt.valXimage, self.tt.valY,
                        self.tt.trainXfeat, self.tt.testXfeat, self.tt.valXfeat]
            elif self.params.valid_set == 'no':
                Data = [self.tt.trainFilenames, self.tt.trainXimage, self.tt.trainY,
                        self.tt.testFilenames, self.tt.testXimage, self.tt.testY,
                        self.tt.trainXfeat, self.tt.testXfeat]
        else:
            print("Set the right data type")

        if self.params.save_data == 'yes':
            with open(self.params.outpath + '/Data.pickle', 'wb') as a:
                pickle.dump(Data, a)

        if self.params.balance_weight == 'yes':
            with open(self.params.outpath + '/class_weights.pickle', 'wb') as cw:
                pickle.dump(self.tt.class_weights, cw)

                # To Save classes and filenames
        np.save(self.params.outpath + '/classes.npy', self.tt.lb.classes_)

        Filenames_for_Ensemble = [self.tt.trainFilenames, self.tt.testFilenames]
        with open(self.params.outpath + '/Filenames_for_Ensemble_training.pickle', 'wb') as b:
            pickle.dump(Filenames_for_Ensemble, b)

        return


if __name__ == '__main__':
    print('\nRunning', sys.argv[0], sys.argv[1:])
    sim = PrepareData(initMode='args')
    sim.LoadData()
    sim.CreateOutDir()
    sim.CreateTrainTestSets()

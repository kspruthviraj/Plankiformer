###########
# IMPORTS #
###########

import argparse
import pathlib
import sys

import numpy as np

from utils import for_plankton_test as fplankton_test
from utils import model_training as mt
from utils import prepare_data_for_testing as pdata_test
import main


class LoadInputParameters:
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
        self.test_path = None
        self.model_path = None
        self.outpath = None
        self.init_name = None
        self.classes = None
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

        parser.add_argument('-test_path', nargs='*', default=['./data/'], help="directory where you want to predict")
        parser.add_argument('-model_path', default='./out/trained_models/', help="directory where you want to predict")
        parser.add_argument('-outpath', default='./out/', help="directory where you want to predict")
        parser.add_argument('-init_name', default='Init_1',
                            help="directory name where you want the Best models to be saved")

        args = parser.parse_args(string)

        self.params = args

        if self.verbose:
            print(args)
        return

    def CreateOutDir(self):
        """ Create a unique output directory, and put inside it a file with the simulation parameters """
        pathlib.Path(self.params.outpath).mkdir(parents=True, exist_ok=True)
        return


if __name__ == '__main__':
    print('\nRunning', sys.argv[0], sys.argv[1:])

    # Loading Testing Input parameters
    inp_params = LoadInputParameters(initMode='args')
    print('model_path: {}'.format(inp_params.params.model_path))
    inp_params.CreateOutDir()
    print('Loaded testing input parameters')
    #
    # Loading Trained Input parameters
    simPred = main.LoadInputParameters(initMode='args')
    print('model_path:{}'.format(inp_params.params.model_path))
    simPred.params = np.load(inp_params.params.model_path + '/params.npy', allow_pickle=True).item()
    #

    print('Creating dataset using input parameters')
    prep_test_data = pdata_test.CreateDataset()
    prep_test_data.LoadData(inp_params, simPred)
    prep_test_data.CreateTrainTestSets(simPred)

    # For Plankton testing
    for_plankton_test = fplankton_test.CreateDataForPlankton()
    for_plankton_test.make_train_test_for_model(simPred, prep_test_data)
    for_plankton_test.create_data_loaders(simPred, inp_params)

    # Model Training
    model_training = mt.import_and_train_model()

    # Prediction
    model_training.load_model_and_run_prediction(simPred, inp_params, for_plankton_test)

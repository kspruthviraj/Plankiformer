import argparse
import numpy as np
import pathlib
import pickle
import sys

from joblib import dump
from sklearn.preprocessing import StandardScaler
from utils import create_data as cdata
from utils import prepare_train_test_data as pdata
from utils import for_plankton as fplankton


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


sim = pdata
sim.LoadData()
sim.CreateOutDir()
sim.CreateTrainTestSets()

fplankton.CreateDataForPlankton()

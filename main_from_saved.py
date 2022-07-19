###########
# IMPORTS #
###########

import argparse
import pathlib
import sys

import numpy as np

from utils import for_birds as birds
from utils import for_cifar10 as cifar10
from utils import for_dogs as dogs
from utils import for_plankton as fplankton
from utils import for_wildtrap as wildtrap
from utils import model_training as mt
from utils import model_training_cnn as mt_cnn
from utils import prepare_train_test_data as pdata


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
        parser.add_argument('-training_data', choices=['True', 'False'], default='False',
                            help="This is to cope with the different directory structures")

        # Preprocessing Images
        parser.add_argument('-resize_images', type=int, default=1,
                            help="Images are resized to a square of LxL pixels by keeping the initial image "
                                 "proportions if resize=1. If resize=2, then the proportions are not kept but resized "
                                 "to match the user defined dimension")

        parser.add_argument('-save_data', choices=['yes', 'no'], default=None,
                            help="Whether to save the data for later use or not")
        parser.add_argument('-saved_data', choices=['yes', 'no'], default=None,
                            help="Whether to use the saved data for training")
        parser.add_argument('-compute_extrafeat', choices=['yes', 'no'], default=None,
                            help="Whether to compute extra features or not")
        parser.add_argument('-valid_set', choices=['yes', 'no'], default='yes',
                            help="Select to have validation set. Choose from Yes or No")
        parser.add_argument('-test_set', choices=['yes', 'no'], default='yes',
                            help="Select to have validation set. Choose from Yes or No")

        # Choose dataset name
        parser.add_argument('-dataset_name', choices=['zoolake', 'zooscan', 'whoi', 'kaggle',
                                                      'eilat', 'rsmas', 'birds', 'dogs', 'beetle', 'wildtrap',
                                                      'cifar10'],
                            default='zoolake', help='Choose between different datasets "zoolake", "zooscan", "whoi", '
                                                    '"kaggle", "eilat", "rsmas", "birds", "dogs", "beetle", "wildtrap"')

        # For model training
        parser.add_argument('-architecture', choices=['cnn', 'deit'],
                            default='deit', help='Choose between different datasets "cnn", "deit"')

        parser.add_argument('-batch_size', type=int, default=16, help="Batch size for training")
        parser.add_argument('-image_size', type=int, default=224, help="Image size for training the model")
        parser.add_argument('-epochs', type=int, default=100, help="number of epochs for training the model")
        parser.add_argument('-gpu_id', type=int, default=0, help="select the gpu id ")
        parser.add_argument('-lr', type=float, default=1e-4, help="starting learning rate")
        parser.add_argument('-warmup', type=int, default=10, help="starting learning rate")
        parser.add_argument('-weight_decay', type=float, default=3e-2, help="weight decay")
        parser.add_argument('-clip_grad_norm', type=float, default=0, help="clip gradient norm")
        parser.add_argument('-disable_cos', choices=[True, False], default=True,
                            help="Disable cos. Choose from Yes or No")
        parser.add_argument('-run_early_stopping', choices=['yes', 'no'], default='no', )
        parser.add_argument('-run_lr_scheduler', choices=['yes', 'no'], default='no', )

        # Superclass or not
        parser.add_argument('-super_class', choices=['yes', 'no'], default='yes', )
        parser.add_argument('-finetune', type=int, default=0, help='Choose "0" or "1" or "2" for finetuning')
        parser.add_argument('-finetune_epochs', type=int, default=40,
                            help="Total number of epochs for the funetune training")
        parser.add_argument('-init_name', default='Init_01',
                            help="directory name where you want the Best models to be saved")

        # Related to predicting on unseen
        parser.add_argument('-test_path', nargs='*', default=['./data/'], help="directory of images where you want to "
                                                                               "predict")
        parser.add_argument('-main_param_path', default='./out/trained_models/', help="main directory where the "
                                                                                      "training parameters are saved")
        parser.add_argument('-test_outpath', default='./out/', help="directory where you want to save the predictions")
        parser.add_argument('-model_path', nargs='*',
                            default=['./out/trained_models/Init_0/',
                                     './out/trained_models/Init_1/'],
                            help='path of the saved models')
        parser.add_argument('-finetuned', type=int, default=2, help='Choose "0" or "1" or "2" for finetuning')
        parser.add_argument('-threshold', type=float, default=0.0, help="Threshold to set")

        # Related to ensembling
        parser.add_argument('-ensemble', type=int, default=0,
                            help="Set this to one if you want to ensemble multiple models else set it to zero")

        # # Train from previous saved models
        # parser.add_argument('-train_from_saved', choices=['yes', 'no'], default='no', )

        args = parser.parse_args(string)

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
        print(self.params, file=self.fsummary)
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


if __name__ == '__main__':
    print('\nRunning', sys.argv[0], sys.argv[1:])

    # Loading Input parameters
    train_params = LoadInputParameters(initMode='args')
    train_params.CreateOutDir()
    print('Loaded input parameters')
    #
    loaded_data = None

    # For Plankton
    loaded_data = fplankton.CreateDataForPlankton()
    loaded_data.make_train_test_for_model(train_params, None)
    loaded_data.create_data_loaders(train_params)

    if train_params.params.architecture == 'deit':
        # Model Training
        model_training = mt.import_and_train_model()
        # Run training
        model_training.train_and_save(train_params, loaded_data)
    elif train_params.params.architecture == 'cnn':
        # Model Training
        model_training = mt_cnn.import_and_train_model()
        # Run training
        model_training.train_and_save(train_params, loaded_data)



###########
# IMPORTS #
###########

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report


class LoadEnsembleParameters:
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

        parser.add_argument('-main_model_dir',
                            default='/local/kyathasr/Plankiformer/out/phyto_super_class/',
                            help="Main directory where the model is stored")
        parser.add_argument('-outpath', default='./out/Ensemble/', help="directory where you want the output saved")

        parser.add_argument('-finetune', type=int, default=0, help='Choose "0" or "1" or "2" for finetuning')
        parser.add_argument('-model_dirs', nargs='*',
                            default=['./data/'],
                            help="Directories with the model.")

        args = parser.parse_args(string)

        args.outpath = args.outpath + '/'

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

    def get_ensemble_performance(self):
        print(self.params.main_model_dir)
        # classes = np.load(self.params.main_model_dir + '/classes.npy')

        classes = ('airplane', 'automobile', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        DEIT_models_all = []
        DEIT_GTLabel = []
        DEIT_PredLabel = []
        DEIT_Prob = []
        DEIT_GTLabel_sorted = []
        DEIT_GTLabel_indices = []
        DEIT_PredLabel_sorted = []
        DEIT_Prob_sorted = []

        for model_path in self.params.model_dirs:
            DEIT_model = []
            if self.params.finetune == 0:
                DEIT_model = pd.read_pickle(model_path + '/GT_Pred_GTLabel_PredLabel_prob_model_original.pickle')

            elif self.params.finetune == 1:
                DEIT_model = pd.read_pickle(model_path + '/GT_Pred_GTLabel_PredLabel_prob_model_tuned.pickle')

            elif self.params.finetune == 2:
                DEIT_model = pd.read_pickle(model_path + '//GT_Pred_GTLabel_PredLabel_prob_model_finetuned.pickle')
            else:
                print(' Please Select correct finetuning parameters')

            DEIT_01_GTLabel = DEIT_model[2]
            DEIT_01_PredLabel = DEIT_model[3]
            DEIT_01_Prob = DEIT_model[4]

            DEIT_01_GTLabel_sorted = np.sort(DEIT_01_GTLabel)
            DEIT_01_GTLabel_indices = np.argsort(DEIT_01_GTLabel)
            DEIT_01_PredLabel_sorted = DEIT_01_PredLabel[DEIT_01_GTLabel_indices]
            DEIT_01_Prob_sorted = DEIT_01_Prob[DEIT_01_GTLabel_indices]

            DEIT_models_all.append(DEIT_model)
            DEIT_GTLabel.append(DEIT_01_GTLabel)
            DEIT_PredLabel.append(DEIT_01_PredLabel)
            DEIT_Prob.append(DEIT_01_Prob)
            DEIT_GTLabel_sorted.append(DEIT_01_GTLabel_sorted)
            DEIT_GTLabel_indices.append(DEIT_01_GTLabel_indices)
            DEIT_PredLabel_sorted.append(DEIT_01_PredLabel_sorted)
            DEIT_Prob_sorted.append(DEIT_01_Prob_sorted)

        Ens_DEIT = sum(DEIT_Prob)/len(DEIT_Prob)
        Ens_DEIT_prob_max = Ens_DEIT.argmax(axis=1)  # The class that the classifier would bet on
        Ens_DEIT_label = np.array([classes[Ens_DEIT_prob_max[i]] for i in range(len(Ens_DEIT_prob_max))], dtype=object)

        print('Accuracy:  {}'.format(round(accuracy_score(DEIT_GTLabel[0], Ens_DEIT_label), 3)))
        print('F1-score:  {}'.format(round(f1_score(DEIT_GTLabel[0], Ens_DEIT_label, average='macro'), 3)))
        print(classification_report(DEIT_GTLabel[0], Ens_DEIT_label, digits=2))

        accuracy_model = accuracy_score(DEIT_GTLabel[0], Ens_DEIT_label)
        clf_report = classification_report(DEIT_GTLabel[0], Ens_DEIT_label)
        f1 = f1_score(DEIT_GTLabel[0], Ens_DEIT_label, average='macro')

        f = open(self.params.outpath + 'Ensemble_test_report.txt', 'w')
        f.write('\n Accuracy\n\n{}\n\nF1 Score\n\n{}\n\nClassification Report\n\n{}\n'.format(accuracy_model, f1,
                                                                                              clf_report))
        f.close()


if __name__ == '__main__':
    print('\n Running Ensemble', sys.argv[0], sys.argv[1:])

    # Loading Input parameters
    train_params = LoadEnsembleParameters(initMode='args')
    train_params.CreateOutDir()
    train_params.get_ensemble_performance()

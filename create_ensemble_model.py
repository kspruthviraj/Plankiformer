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

        parser.add_argument('-main_model_dir', nargs='*',
                            default='/local/kyathasr/Plankiformer/out/phyto_super_class/',
                            help="Main directory where the model is stored")
        parser.add_argument('-outpath', default='./out/Ensemble/', help="directory where you want the output saved")

        parser.add_argument('-finetune', type=int, default=0, help='Choose "0" or "1" or "2" for finetuning')

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
        classes = np.load(self.params.main_model_dir + '/classes.npy')

        DEIT_01Path = self.params.main_model_dir + 'trained_models/' + 'Init_01/'
        DEIT_02Path = self.params.main_model_dir + 'trained_models/' + 'Init_02/'
        DEIT_03Path = self.params.main_model_dir + 'trained_models/' + 'Init_03/'

        if self.params.finetune == 0:
            DEIT_01 = pd.read_pickle(DEIT_01Path + '/GT_Pred_GTLabel_PredLabel_prob_model_original.pickle')
            DEIT_02 = pd.read_pickle(DEIT_02Path + '/GT_Pred_GTLabel_PredLabel_prob_model_original.pickle')
            DEIT_03 = pd.read_pickle(DEIT_03Path + '/GT_Pred_GTLabel_PredLabel_prob_model_original.pickle')
        elif self.params.finetune == 1:
            DEIT_01 = pd.read_pickle(DEIT_01Path + '/GT_Pred_GTLabel_PredLabel_prob_model_tuned.pickle')
            DEIT_02 = pd.read_pickle(DEIT_02Path + '/GT_Pred_GTLabel_PredLabel_prob_model_tuned.pickle')
            DEIT_03 = pd.read_pickle(DEIT_03Path + '/GT_Pred_GTLabel_PredLabel_prob_model_tuned.pickle')
        elif self.params.finetune == 2:
            DEIT_01 = pd.read_pickle(DEIT_01Path + '/GT_Pred_GTLabel_PredLabel_prob_model_finetuned.pickle')
            DEIT_02 = pd.read_pickle(DEIT_02Path + '/GT_Pred_GTLabel_PredLabel_prob_model_finetuned.pickle')
            DEIT_03 = pd.read_pickle(DEIT_03Path + '/GT_Pred_GTLabel_PredLabel_prob_model_finetuned.pickle')

        DEIT_01_GTLabel = DEIT_01[2]
        DEIT_01_PredLabel = DEIT_01[3]
        DEIT_01_Prob = DEIT_01[4]
        DEIT_01_GTLabel_01 = np.sort(DEIT_01_GTLabel)
        DEIT_01_GTLabel_indices = np.argsort(DEIT_01_GTLabel)
        DEIT_01_PredLabel_01 = DEIT_01_PredLabel[DEIT_01_GTLabel_indices]
        DEIT_01_Prob_01 = DEIT_01_Prob[DEIT_01_GTLabel_indices]

        DEIT_02_GTLabel = DEIT_02[2]
        DEIT_02_PredLabel = DEIT_02[3]
        DEIT_02_Prob = DEIT_02[4]
        DEIT_02_GTLabel_02 = np.sort(DEIT_02_GTLabel)
        DEIT_02_GTLabel_indices = np.argsort(DEIT_02_GTLabel)
        DEIT_02_PredLabel_02 = DEIT_02_PredLabel[DEIT_02_GTLabel_indices]
        DEIT_02_Prob_02 = DEIT_02_Prob[DEIT_02_GTLabel_indices]

        DEIT_03_GTLabel = DEIT_03[2]
        DEIT_03_PredLabel = DEIT_03[3]
        DEIT_03_Prob = DEIT_03[4]
        DEIT_03_GTLabel_03 = np.sort(DEIT_03_GTLabel)
        DEIT_03_GTLabel_indices = np.argsort(DEIT_03_GTLabel)
        DEIT_03_PredLabel_03 = DEIT_03_PredLabel[DEIT_03_GTLabel_indices]
        DEIT_03_Prob_03 = DEIT_03_Prob[DEIT_03_GTLabel_indices]

        Ens_DEIT_1 = np.add(DEIT_01_Prob_01, DEIT_02_Prob_02)
        Ens_DEIT = np.add(Ens_DEIT_1, DEIT_03_Prob_03)
        Ens_DEIT = Ens_DEIT / 3
        Ens_DEIT_prob_max = Ens_DEIT.argmax(axis=1)  # The class that the classifier would bet on
        Ens_DEIT_label = np.array([classes[Ens_DEIT_prob_max[i]] for i in range(len(Ens_DEIT_prob_max))], dtype=object)

        print('Accuracy:  {}'.format(round(accuracy_score(Ens_DEIT_label, DEIT_03_GTLabel_03), 3)))
        print('F1-score:  {}'.format(round(f1_score(Ens_DEIT_label, DEIT_03_GTLabel_03, average='macro'), 3)))
        print(classification_report(Ens_DEIT_label, DEIT_03_GTLabel_03, digits=2))

        accuracy_model = accuracy_score(Ens_DEIT_label, DEIT_03_GTLabel_03)
        clf_report = classification_report(Ens_DEIT_label, DEIT_03_GTLabel_03)
        f1 = f1_score(Ens_DEIT_label, DEIT_03_GTLabel_03, average='macro')

        f = open(self.params.outpath + 'Ensemble_test_report.txt', 'w')
        f.write('\n Accuracy\n\n{}\n\nF1 Score\n\n{}\n\nClassification Report\n\n{}\n'.format(accuracy_model, f1,
                                                                                              clf_report))
        f.close()


if __name__ == '__main__':
    print('\n Running Ensemble', sys.argv[0], sys.argv[1:])

    # Loading Input parameters
    inp_params = LoadEnsembleParameters(initMode='args')
    inp_params.CreateOutDir()
    inp_params.get_ensemble_performance()

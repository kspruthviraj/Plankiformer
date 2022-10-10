import os
import shutil
import numpy as np
import pickle

def MakeDatasets_txt(datapath):
    filenames_train = np.loadtxt(datapath + '/zoolake_train_test_val_separated/train_filenames.txt', dtype=str)
    filenames_val = np.loadtxt(datapath + '/zoolake_train_test_val_separated/val_filenames.txt', dtype=str)
    filenames_test = np.loadtxt(datapath + '/zoolake_train_test_val_separated/test_filenames.txt', dtype=str)

    list_classes = os.listdir(datapath + '/zooplankton_0p5x/')
    
    for iclass in list_classes:
        if not os.path.exists(datapath + '/0_train/' + iclass):
            os.makedirs(datapath + '/0_train/' + iclass)
        for img in filenames_train:
            if iclass in img:
                shutil.copy(datapath + img[16:], datapath + '/0_train/' + iclass)

    for iclass in list_classes:
        if not os.path.exists(datapath + '/0_val/' + iclass):
            os.makedirs(datapath + '/0_val/' + iclass)
        for img in filenames_val:
            if iclass in img:
                shutil.copy(datapath + img[16:], datapath + '/0_val/' + iclass)

    for iclass in list_classes:
        if not os.path.exists(datapath + '/0_test/' + iclass):
            os.makedirs(datapath + '/0_test/' + iclass)
        for img in filenames_test:
            if iclass in img:
                shutil.copy(datapath + img[16:], datapath + '/0_test/' + iclass)


def MakeDatasets_pickle(datapath, outpath, split_pickle_file):
    train_test_val = pickle.load(open(split_pickle_file, 'rb'))

    train_filenames = train_test_val[0]
    test_filenames = train_test_val[1]
    val_filenames = train_test_val[2]

    list_classes = os.listdir(datapath)

    for iclass in list_classes:
        if not os.path.exists(outpath + '/0_train/' + iclass):
            os.makedirs(outpath + '/0_train/' + iclass)
        for img in train_filenames:
            if iclass in img:
                shutil.copy(datapath + img[50:], outpath + '/0_train/' + iclass)

    for iclass in list_classes:
        if not os.path.exists(outpath + '/0_test/' + iclass):
            os.makedirs(outpath + '/0_test/' + iclass)
        for img in test_filenames:
            if iclass in img:
                shutil.copy(datapath + img[50:], outpath + '/0_test/' + iclass)

    for iclass in list_classes:
        if not os.path.exists(outpath + '/0_val/' + iclass):
            os.makedirs(outpath + '/0_val/' + iclass)
        for img in val_filenames:
            if iclass in img:
                shutil.copy(datapath + img[50:], outpath + '/0_val/' + iclass)


MakeDatasets_pickle(r'/home/EAWAG/chenchen/data/train_data/new/training_zooplankton_new_220823/', r'/home/EAWAG/chenchen/data/train_data/new/', r'/home/EAWAG/chenchen/data/train_data/new/Files_used_for_training_testing.pickle')
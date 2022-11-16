import os
import shutil
import numpy as np
import pickle

def SplitFrom_txt(datapath):
    filenames_train = np.loadtxt(datapath + '/zoolake_train_test_val_separated/train_filenames.txt', dtype=str)
    filenames_val = np.loadtxt(datapath + '/zoolake_train_test_val_separated/val_filenames.txt', dtype=str)
    filenames_test = np.loadtxt(datapath + '/zoolake_train_test_val_separated/test_filenames.txt', dtype=str)

    list_classes = os.listdir(datapath + '/zooplankton_0p5x/')
    
    for iclass in list_classes:
        if not os.path.exists(datapath + '/0_train/' + iclass):
            os.makedirs(datapath + '/0_train/' + iclass)
        for img in filenames_train:
            if iclass + '/' in img:
                shutil.copy(datapath + img[16:], datapath + '/0_train/' + iclass)

    for iclass in list_classes:
        if not os.path.exists(datapath + '/0_val/' + iclass):
            os.makedirs(datapath + '/0_val/' + iclass)
        for img in filenames_val:
            if iclass + '/' in img:
                shutil.copy(datapath + img[16:], datapath + '/0_val/' + iclass)

    for iclass in list_classes:
        if not os.path.exists(datapath + '/0_test/' + iclass):
            os.makedirs(datapath + '/0_test/' + iclass)
        for img in filenames_test:
            if iclass + '/' in img:
                shutil.copy(datapath + img[16:], datapath + '/0_test/' + iclass)


def SplitFrom_pickle(datapath, outpath, split_pickle_file):
    train_test_val = pickle.load(open(split_pickle_file, 'rb'))

    train_filenames = train_test_val[0]
    test_filenames = train_test_val[1]
    val_filenames = train_test_val[2]

    list_classes = os.listdir(datapath)

    for iclass in list_classes:
        if not os.path.exists(outpath + '/0_train/' + iclass):
            os.makedirs(outpath + '/0_train/' + iclass)
        for img in train_filenames:
            if iclass + '/' in img:
                if not os.path.exists(datapath + '/' + img[50:]):
                    continue
                shutil.copy(datapath + '/' + img[50:], outpath + '/0_train/' + iclass)

    for iclass in list_classes:
        if not os.path.exists(outpath + '/0_test/' + iclass):
            os.makedirs(outpath + '/0_test/' + iclass)
        for img in test_filenames:
            if iclass + '/' in img:
                if not os.path.exists(datapath + '/' + img[50:]):
                    continue
                shutil.copy(datapath + '/' + img[50:], outpath + '/0_test/' + iclass)

    for iclass in list_classes:
        if not os.path.exists(outpath + '/0_val/' + iclass):
            os.makedirs(outpath + '/0_val/' + iclass)
        for img in val_filenames:
            if iclass + '/' in img:
                if not os.path.exists(datapath + '/' + img[50:]):
                    continue
                shutil.copy(datapath + '/' + img[50:], outpath + '/0_val/' + iclass)


def SplitByTime(datapath, outpath):
    list_class = os.listdir(datapath) # list of class names

    list_image = []
    # list of all image names in the train dataset
    for iclass in list_class:
        for img in os.listdir(datapath + '/%s/' % iclass):
            list_image.append(img)

    for iclass in list_class:
        if not os.path.exists(outpath + '/before2021/' + iclass):
            os.makedirs(outpath + '/before2021/' + iclass)
        if not os.path.exists(outpath + '/after2021/' + iclass):
            os.makedirs(outpath + '/after2021/' + iclass)

        for img in list_image:
            if img == 'Thumbs.db':
                continue
            if not os.path.exists(datapath + '/' + iclass + '/' + img):
                continue
            if int(img[15:25]) < 1609455600:
                shutil.copy(datapath + '/' + iclass + '/' + img, outpath + '/before2021/' + iclass)
            else:
                shutil.copy(datapath + '/' + iclass + '/' + img, outpath + '/after2021/' + iclass)




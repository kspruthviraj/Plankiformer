import os
import shutil
import numpy as np

def LoadFile(datapath):
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


LoadFile(r'C:\Users\chenchen\thesis\data\train_data\old\Zoolake_Dataset')
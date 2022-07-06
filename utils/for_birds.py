from __future__ import print_function

import os
from os.path import join

import numpy as np
import pandas as pd
import scipy.io
import torch
import torch.utils.data as data
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url, list_dir
from torchvision.datasets.folder import default_loader


class CreateDataForBirds:
    def __init__(self):
        self.label_map = None
        self.class_names = None
        self.checkpoint_path = None
        self.classes = None
        self.test_dataloader = None
        self.train_dataloader = None
        self.val_dataloader = None
        self.X_val = None
        self.X_test = None
        self.X_train = None
        self.class_weights_tensor = None
        self.params = None
        return

    def make_train_test_for_birds(self, train_main):
        train_transform = T.Compose([T.Resize((224, 224)),
                                     T.RandomHorizontalFlip(),
                                     T.RandomVerticalFlip(),
                                     T.ToTensor()])

        test_transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])

        train_PATH = train_main.params.datapaths
        train_PATH = ' '.join(map(str, train_PATH))

        trainset = NABirds(root=train_PATH, train=True, transform=train_transform)
        test_set = NABirds(root=train_PATH, train=False, transform=test_transform)

        train_set, val_set = torch.utils.data.random_split(trainset, [int(np.round(0.8 * len(trainset), 0)),
                                                                      int(np.round(0.2 * len(trainset), 0))])

        self.train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=train_main.params.batch_size,
                                                            shuffle=True, num_workers=4, pin_memory=True)
        self.val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=train_main.params.batch_size,
                                                          shuffle=True, num_workers=4, pin_memory=True)
        self.test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=train_main.params.batch_size,
                                                           shuffle=False, num_workers=4, pin_memory=True)

        self.checkpoint_path = train_main.params.outpath + 'trained_models/' + train_main.params.init_name + '/'

        dataset_path = os.path.join(train_PATH, 'nabirds')
        image_class_labels = pd.read_csv(os.path.join(dataset_path, 'image_class_labels.txt'), sep=' ',
                                         names=['img_id', 'target'])
        label_set = list(set(image_class_labels['target']))
        class_names = load_class_names(dataset_path)

        classes = []
        for i in label_set:
            classes.append(list(class_names.values())[list(class_names.keys()).index(str(i))])

        self.classes = classes
        self.class_weights_tensor = torch.load(train_main.params.outpath + '/class_weights.pt')


class NABirds(Dataset):
    """`NABirds <https://dl.allaboutbirds.org/nabirds>`_ Dataset.
        Args:
            root (string): Root directory of the dataset.
            train (bool, optional): If True, creates dataset from training set, otherwise
               creates from test set.
            transform (callable, optional): A function/transform that  takes in an PIL image
               and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
               target and transforms it.
            download (bool, optional): If true, downloads the dataset from the internet and
               puts it in root directory. If dataset is already downloaded, it is not
               downloaded again.
    """
    base_folder = 'nabirds/images'

    def __init__(self, root, train=True, transform=None):
        dataset_path = os.path.join(root, 'nabirds')
        self.root = root
        self.loader = default_loader
        self.train = train
        self.transform = transform

        image_paths = pd.read_csv(os.path.join(dataset_path, 'images.txt'),
                                  sep=' ', names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(dataset_path, 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        # Since the raw labels are non-continuous, map them to new ones
        self.label_map = get_continuous_class_map(image_class_labels['target'])
        train_test_split = pd.read_csv(os.path.join(dataset_path, 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])
        data = image_paths.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')
        # Load in the train / test split
        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

        # Load in the class data
        self.class_names = load_class_names(dataset_path)
        self.class_hierarchy = load_hierarchy(dataset_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = self.label_map[sample.target]
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)
        return img, target


def get_continuous_class_map(class_labels):
    label_set = set(class_labels)
    return {k: i for i, k in enumerate(label_set)}


def load_class_names(dataset_path=''):
    names = {}

    with open(os.path.join(dataset_path, 'classes.txt')) as f:
        for line in f:
            pieces = line.strip().split()
            class_id = pieces[0]
            names[class_id] = ' '.join(pieces[1:])

    return names


def load_hierarchy(dataset_path=''):
    parents = {}

    with open(os.path.join(dataset_path, 'hierarchy.txt')) as f:
        for line in f:
            pieces = line.strip().split()
            child_id, parent_id = pieces
            parents[child_id] = parent_id

    return parents

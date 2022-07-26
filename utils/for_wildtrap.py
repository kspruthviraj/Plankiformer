import os
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from sklearn.utils import compute_class_weight
from torch.utils.data import Dataset
from torchvision import datasets
import torchvision.transforms as T
from torch.utils.data import Dataset


class CreateDataForWildtrap:
    def __init__(self):
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

    def make_train_test_for_wildtrap(self, train_main):
        train_PATH = train_main.params.datapaths
        test_PATH = train_main.params.test_path

        train_PATH = ' '.join(map(str, train_PATH))
        test_PATH = ' '.join(map(str, test_PATH))

        trainset = datasets.ImageFolder(train_PATH)
        testset = datasets.ImageFolder(test_PATH)

        class_weight_path = train_main.params.outpath + '/class_weights_tensor.pt'
        if os.path.exists(class_weight_path):
            self.class_weights_tensor = torch.load(train_main.params.outpath + '/class_weights_tensor.pt')
        else:
            class_train = []
            for i in range(len(trainset)):
                class_train.append(trainset[i][1])
            class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(class_train),
                                                 y=class_train)
            self.class_weights_tensor = torch.Tensor(class_weights)
            torch.save(self.class_weights_tensor, train_main.params.outpath + '/class_weights_tensor.pt')

        torch.save(trainset.classes, train_main.params.outpath + '/class_labels.pt')

        self.classes = trainset.classes

        train_transform = T.Compose([T.Resize((224, 224)), T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
                                     T.GaussianBlur(kernel_size=(3, 9), sigma=(0.1, 2)),
                                     T.RandomRotation(degrees=(0, 180)), T.ToTensor()])

        test_transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])

        train_set, val_set = torch.utils.data.random_split(trainset, [int(np.round(0.8 * len(trainset), 0)),
                                                                      int(np.round(0.2 * len(trainset), 0))])

        train_set = ApplyTransform(train_set, transform=train_transform)
        val_set = ApplyTransform(val_set, transform=train_transform)

        test_set = ApplyTransform(testset, transform=test_transform)

        self.train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=train_main.params.batch_size,
                                                            shuffle=True, num_workers=4, pin_memory=True)
        self.val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=train_main.params.batch_size,
                                                          shuffle=True, num_workers=4, pin_memory=True)
        self.test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=train_main.params.batch_size,
                                                           shuffle=False, num_workers=4, pin_memory=True)

        # classes_train = [label for _, label in trainset]
        # classes_val = [label for _, label in valset]
        # classes_all = classes_train + classes_val
        # print(len(Counter(classes_all)))
        # class_weights_all = compute_class_weight(class_weight='balanced',classes=np.unique(classes_all),y=classes_all)
        # self.class_weights_tensor = torch.Tensor(class_weights_all)
        # torch.save(self.class_weights_tensor, train_main.params.outpath + '/class_weights.pt')

        # self.class_weights_tensor = torch.load(train_main.params.outpath + '/class_weights.pt')
        self.checkpoint_path = train_main.params.outpath + 'trained_models/' + train_main.params.init_name + '/'
        Path(self.checkpoint_path).mkdir(parents=True, exist_ok=True)
        return


class ApplyTransform(Dataset):
    """
    Apply transformations to a Dataset

    Arguments:
        dataset (Dataset): A Dataset that returns (sample, target)
        transform (callable, optional): A function/transform to be applied on the sample
        target_transform (callable, optional): A function/transform to be applied on the target

    """

    def __init__(self, dataset, transform=None, target_transform=None):
        self.dataset = dataset
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, idx):
        sample, target = self.dataset[idx]
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def __len__(self):
        return len(self.dataset)

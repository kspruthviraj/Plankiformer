import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision import datasets


class CreateDataForCifar10:
    def __init__(self):
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

    def make_train_test_for_cifar(self, train_main):
        self.classes = ('plane', 'car', 'bird', 'cat',
                        'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        mean, std = (0.5,), (0.5,)
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean, std)
                                        ])

        trainset = datasets.CIFAR10('../data/CIFAR10/', download=True, train=True, transform=transform)

        train_set, val_set = torch.utils.data.random_split(trainset, [int(np.round(0.8 * len(trainset), 0)),
                                                                      int(np.round(0.2 * len(trainset), 0))])

        self.train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=train_main.params.batch_size,
                                                            shuffle=True)
        self.val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=train_main.params.batch_size,
                                                          shuffle=True)

        testset = datasets.CIFAR10('../data/CIFAR10/', download=True, train=False, transform=transform)
        self.test_dataloader = torch.utils.data.DataLoader(testset, batch_size=train_main.params.batch_size,
                                                           shuffle=False)

        return

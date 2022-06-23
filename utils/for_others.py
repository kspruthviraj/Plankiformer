###########
# IMPORTS #
###########
import shutil
from collections import Counter
import numpy as np
import pandas as pd
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as T
from sklearn.utils import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from pathlib import Path


# from torchvision.utils import make_grid
# from matplotlib import pyplot as plt


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


class CreateDataForPlankton:
    def __init__(self):
        self.val_dataloader = None
        self.test_dataloader = None
        self.train_dataloader = None
        self.checkpoint_path = None
        self.y_val = None
        self.y_test = None
        self.y_train = None
        self.testFilenames = None
        self.trainFilenames = None
        self.valFilenames = None
        self.X_val = None
        self.X_test = None
        self.X_train = None
        self.class_weights = None
        self.params = None
        return

    def make_train_test_for_model(self, train_main):
        train_path = train_main.params.datapaths + 'Train/'
        AA = pd.read_csv(train_main.params.datapaths + 'train0.txt', header=None)
        copy_data(AA, train_main.params.datapaths, train_path)
        test_path = train_main.params.datapaths + 'Test/'
        AA = pd.read_csv(train_main.params.datapaths + 'test0.txt', header=None)
        copy_data(AA, train_main.params.datapaths, test_path)

        train_set = datasets.ImageFolder(train_path)
        test_set = datasets.ImageFolder(test_path)
        # class_labels = train_set.classes

        torch.save(train_set.classes, train_main.params.outpath + '/class_labels.pt')

        train_set, val_set = torch.utils.data.random_split(train_set, [int(np.round(0.8 * len(train_set), 0)),
                                                                       int(np.round(0.2 * len(train_set), 0))])
        #
        # train_transform = T.Compose([T.Resize((224, 224)), T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
        #                              T.GaussianBlur(kernel_size=(3, 9), sigma=(0.1, 2)),
        #                              T.RandomRotation(degrees=(0, 180)),
        #                              T.ToTensor()])

        test_transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])

        train_set = ApplyTransform(train_set, transform=test_transform)
        val_set = ApplyTransform(val_set, transform=test_transform)
        test_set = ApplyTransform(test_set, transform=test_transform)

        self.train_dataloader = DataLoader(train_set, 64, shuffle=True, num_workers=4, pin_memory=True)
        self.val_dataloader = DataLoader(val_set, 64, shuffle=True, num_workers=4, pin_memory=True)
        self.test_dataloader = DataLoader(test_set, 64, shuffle=True, num_workers=4, pin_memory=True)

        # def show_images(data, nmax=64):
        #     fig, ax = plt.subplots(figsize=(8, 8))
        #     ax.set_xticks([])
        #     ax.set_yticks([])
        #     print(data[1])
        #     ax.imshow(make_grid((data[0].detach()[:nmax]), nrow=8).permute(1, 2, 0))
        #
        # def show_batch(dl, nmax=64):
        #     for images in dl:
        #         show_images(images, nmax)
        #         break
        #
        # show_batch(test_dataloader)

        classes = [label for _, label in train_set]
        classes_val = [label for _, label in val_set]
        classes_all = classes + classes_val
        print(len(Counter(classes_all)))
        class_weights_all = compute_class_weight(class_weight='balanced', classes=np.unique(classes_all), y=classes_all)
        class_weights_all = torch.Tensor(class_weights_all)
        torch.save(class_weights_all, train_main.params.datapaths + '/class_weights_all.pt')


def copy_data(AA, data_root, path_out):
    for i in range(len(AA)):
        tt = AA[0][i]
        x = tt.split('\\', 2)

        Data_PATH = data_root + 'Data/' + x[0] + '/' + x[1]
        copy_path = path_out + x[0] + '/'
        Path(copy_path).mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy(Data_PATH, copy_path)

        except Exception:
            pass

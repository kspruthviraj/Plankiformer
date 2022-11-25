import os
import random
from pathlib import Path

import numpy as np
import timm
import torch
from PIL import Image, ImageEnhance, ImageOps
from scipy import ndimage
from sklearn.utils import compute_class_weight
from timm.data.auto_augment import rand_augment_transform
from torch.utils.data import Dataset
from torchvision import datasets

torch.manual_seed(0)


# from auto_augment import AutoAugment, Cutout


class CreateDataForCifar10:
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

    def make_train_test_for_cifar(self, train_main, train_transform=None):
        self.classes = ('airplane', 'automobile', 'bird', 'cat',
                        'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        if train_main.params.architecture == 'cnn':
            model = timm.create_model('tf_efficientnet_b7', pretrained=True,
                                      num_classes=10)

        elif train_main.params.architecture == 'deit':
            model = timm.create_model('deit_base_distilled_patch16_224', pretrained=True,
                                      num_classes=10)

        # Load data config associated with the model to use in data augmentation pipeline
        data_config = timm.data.resolve_data_config({}, model=model, verbose=True)
        data_mean = data_config["mean"]
        data_std = data_config["std"]

        train_transform = timm.data.create_transform(
            input_size=224,
            is_training=True,
            mean=data_mean,
            std=data_std,
            auto_augment="rand-m9-mstd0.5", hflip=0.5, vflip=0.5, re_prob=0.3)

        # random_erase = RandomErasing(probability=0.5)
        # train_transform.extend(random_erase, AutoAugment(), Cutout())

        test_transform = timm.data.create_transform(input_size=224,
                                                    mean=data_mean,
                                                    std=data_std)

        # train_transform = [T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), AutoAugment(), Cutout()]
        # train_transform.extend([T.ToTensor(),
        #                         T.Normalize((0.4914, 0.4822, 0.4465),
        #                                     (0.2023, 0.1994, 0.2010)), ])
        # train_transform = T.Compose(train_transform)
        #
        # test_transform = T.Compose([
        #     T.ToTensor(),
        #     T.Normalize((0.5071, 0.4867, 0.4408),
        #                          (0.2675, 0.2565, 0.2761)),
        # ])
        #
        # trainset = datasets.CIFAR10(
        #     root='~/data',
        #     train=True,
        #     download=True,
        #     transform=train_transform)
        # test_set = datasets.CIFAR10(
        #     root='~/data',
        #     train=False,
        #     download=True,
        #     transform=test_transform)

        # train_transform = T.Compose([T.Resize((224, 224)), T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
        #                              T.GaussianBlur(kernel_size=(3, 9), sigma=(0.1, 2)),
        #                              T.RandomRotation(degrees=(0, 180)), T.ToTensor()])
        #
        # test_transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])

        trainset = datasets.CIFAR10('./data/CIFAR10/', download=True, train=True)
        test_set = datasets.CIFAR10('./data/CIFAR10/', download=True, train=False)

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

        train_set, val_set = torch.utils.data.random_split(trainset, [int(np.round(0.98 * len(trainset), 0)),
                                                                      int(np.round(0.02 * len(trainset), 0))])

        train_set = ApplyTransform(train_set, transform=train_transform)
        val_set = ApplyTransform(val_set, transform=train_transform)
        test_set = ApplyTransform(test_set, transform=test_transform)

        self.checkpoint_path = train_main.params.outpath + 'trained_models/' + train_main.params.init_name + '/'
        print('checkpoint_path: {}'.format(self.checkpoint_path))
        Path(self.checkpoint_path).mkdir(parents=True, exist_ok=True)

        self.train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=train_main.params.batch_size,
                                                            shuffle=False, num_workers=4, pin_memory=True)
        self.val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=train_main.params.batch_size,
                                                          shuffle=False, num_workers=4, pin_memory=True)
        self.test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=train_main.params.batch_size,
                                                           shuffle=False, num_workers=4, pin_memory=True)

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


class AutoAugment(object):
    def __init__(self):
        self.policies = [
            ['Invert', 0.1, 7, 'Contrast', 0.2, 6],
            ['Rotate', 0.7, 2, 'TranslateX', 0.3, 9],
            ['Sharpness', 0.8, 1, 'Sharpness', 0.9, 3],
            ['ShearY', 0.5, 8, 'TranslateY', 0.7, 9],
            ['AutoContrast', 0.5, 8, 'Equalize', 0.9, 2],
            ['ShearY', 0.2, 7, 'Posterize', 0.3, 7],
            ['Color', 0.4, 3, 'Brightness', 0.6, 7],
            ['Sharpness', 0.3, 9, 'Brightness', 0.7, 9],
            ['Equalize', 0.6, 5, 'Equalize', 0.5, 1],
            ['Contrast', 0.6, 7, 'Sharpness', 0.6, 5],
            ['Color', 0.7, 7, 'TranslateX', 0.5, 8],
            ['Equalize', 0.3, 7, 'AutoContrast', 0.4, 8],
            ['TranslateY', 0.4, 3, 'Sharpness', 0.2, 6],
            ['Brightness', 0.9, 6, 'Color', 0.2, 8],
            ['Solarize', 0.5, 2, 'Invert', 0.0, 3],
            ['Equalize', 0.2, 0, 'AutoContrast', 0.6, 0],
            ['Equalize', 0.2, 8, 'Equalize', 0.6, 4],
            ['Color', 0.9, 9, 'Equalize', 0.6, 6],
            ['AutoContrast', 0.8, 4, 'Solarize', 0.2, 8],
            ['Brightness', 0.1, 3, 'Color', 0.7, 0],
            ['Solarize', 0.4, 5, 'AutoContrast', 0.9, 3],
            ['TranslateY', 0.9, 9, 'TranslateY', 0.7, 9],
            ['AutoContrast', 0.9, 2, 'Solarize', 0.8, 3],
            ['Equalize', 0.8, 8, 'Invert', 0.1, 3],
            ['TranslateY', 0.7, 9, 'AutoContrast', 0.9, 1],
        ]

    def __call__(self, img):
        img = apply_policy(img, self.policies[random.randrange(len(self.policies))])
        return img


operations = {
    'ShearX': lambda img, magnitude: shear_x(img, magnitude),
    'ShearY': lambda img, magnitude: shear_y(img, magnitude),
    'TranslateX': lambda img, magnitude: translate_x(img, magnitude),
    'TranslateY': lambda img, magnitude: translate_y(img, magnitude),
    'Rotate': lambda img, magnitude: rotate(img, magnitude),
    'AutoContrast': lambda img, magnitude: auto_contrast(img, magnitude),
    'Invert': lambda img, magnitude: invert(img, magnitude),
    'Equalize': lambda img, magnitude: equalize(img, magnitude),
    'Solarize': lambda img, magnitude: solarize(img, magnitude),
    'Posterize': lambda img, magnitude: posterize(img, magnitude),
    'Contrast': lambda img, magnitude: contrast(img, magnitude),
    'Color': lambda img, magnitude: color(img, magnitude),
    'Brightness': lambda img, magnitude: brightness(img, magnitude),
    'Sharpness': lambda img, magnitude: sharpness(img, magnitude),
    'Cutout': lambda img, magnitude: cutout(img, magnitude),
}


def apply_policy(img, policy):
    if random.random() < policy[1]:
        img = operations[policy[0]](img, policy[2])
    if random.random() < policy[4]:
        img = operations[policy[3]](img, policy[5])

    return img


def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = offset_matrix @ matrix @ reset_matrix
    return transform_matrix


def shear_x(img, magnitude):
    img = np.array(img)
    magnitudes = np.linspace(-0.3, 0.3, 11)

    transform_matrix = np.array([[1, random.uniform(magnitudes[magnitude], magnitudes[magnitude + 1]), 0],
                                 [0, 1, 0],
                                 [0, 0, 1]])
    transform_matrix = transform_matrix_offset_center(transform_matrix, img.shape[0], img.shape[1])
    affine_matrix = transform_matrix[:2, :2]
    offset = transform_matrix[:2, 2]
    img = np.stack([ndimage.interpolation.affine_transform(
        img[:, :, c],
        affine_matrix,
        offset) for c in range(img.shape[2])], axis=2)
    img = Image.fromarray(img)
    return img


def shear_y(img, magnitude):
    img = np.array(img)
    magnitudes = np.linspace(-0.3, 0.3, 11)

    transform_matrix = np.array([[1, 0, 0],
                                 [random.uniform(magnitudes[magnitude], magnitudes[magnitude + 1]), 1, 0],
                                 [0, 0, 1]])
    transform_matrix = transform_matrix_offset_center(transform_matrix, img.shape[0], img.shape[1])
    affine_matrix = transform_matrix[:2, :2]
    offset = transform_matrix[:2, 2]
    img = np.stack([ndimage.interpolation.affine_transform(
        img[:, :, c],
        affine_matrix,
        offset) for c in range(img.shape[2])], axis=2)
    img = Image.fromarray(img)
    return img


def translate_x(img, magnitude):
    img = np.array(img)
    magnitudes = np.linspace(-150 / 331, 150 / 331, 11)

    transform_matrix = np.array([[1, 0, 0],
                                 [0, 1,
                                  img.shape[1] * random.uniform(magnitudes[magnitude], magnitudes[magnitude + 1])],
                                 [0, 0, 1]])
    transform_matrix = transform_matrix_offset_center(transform_matrix, img.shape[0], img.shape[1])
    affine_matrix = transform_matrix[:2, :2]
    offset = transform_matrix[:2, 2]
    img = np.stack([ndimage.interpolation.affine_transform(
        img[:, :, c],
        affine_matrix,
        offset) for c in range(img.shape[2])], axis=2)
    img = Image.fromarray(img)
    return img


def translate_y(img, magnitude):
    img = np.array(img)
    magnitudes = np.linspace(-150 / 331, 150 / 331, 11)

    transform_matrix = np.array(
        [[1, 0, img.shape[0] * random.uniform(magnitudes[magnitude], magnitudes[magnitude + 1])],
         [0, 1, 0],
         [0, 0, 1]])
    transform_matrix = transform_matrix_offset_center(transform_matrix, img.shape[0], img.shape[1])
    affine_matrix = transform_matrix[:2, :2]
    offset = transform_matrix[:2, 2]
    img = np.stack([ndimage.interpolation.affine_transform(
        img[:, :, c],
        affine_matrix,
        offset) for c in range(img.shape[2])], axis=2)
    img = Image.fromarray(img)
    return img


def rotate(img, magnitude):
    img = np.array(img)
    magnitudes = np.linspace(-30, 30, 11)
    theta = np.deg2rad(random.uniform(magnitudes[magnitude], magnitudes[magnitude + 1]))
    transform_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                 [np.sin(theta), np.cos(theta), 0],
                                 [0, 0, 1]])
    transform_matrix = transform_matrix_offset_center(transform_matrix, img.shape[0], img.shape[1])
    affine_matrix = transform_matrix[:2, :2]
    offset = transform_matrix[:2, 2]
    img = np.stack([ndimage.interpolation.affine_transform(
        img[:, :, c],
        affine_matrix,
        offset) for c in range(img.shape[2])], axis=2)
    img = Image.fromarray(img)
    return img


def auto_contrast(img, magnitude):
    img = ImageOps.autocontrast(img)
    return img


def invert(img, magnitude):
    img = ImageOps.invert(img)
    return img


def equalize(img, magnitude):
    img = ImageOps.equalize(img)
    return img


def solarize(img, magnitude):
    magnitudes = np.linspace(0, 256, 11)
    img = ImageOps.solarize(img, random.uniform(magnitudes[magnitude], magnitudes[magnitude + 1]))
    return img


def posterize(img, magnitude):
    magnitudes = np.linspace(4, 8, 11)
    img = ImageOps.posterize(img, int(round(random.uniform(magnitudes[magnitude], magnitudes[magnitude + 1]))))
    return img


def contrast(img, magnitude):
    magnitudes = np.linspace(0.1, 1.9, 11)
    img = ImageEnhance.Contrast(img).enhance(random.uniform(magnitudes[magnitude], magnitudes[magnitude + 1]))
    return img


def color(img, magnitude):
    magnitudes = np.linspace(0.1, 1.9, 11)
    img = ImageEnhance.Color(img).enhance(random.uniform(magnitudes[magnitude], magnitudes[magnitude + 1]))
    return img


def brightness(img, magnitude):
    magnitudes = np.linspace(0.1, 1.9, 11)
    img = ImageEnhance.Brightness(img).enhance(random.uniform(magnitudes[magnitude], magnitudes[magnitude + 1]))
    return img


def sharpness(img, magnitude):
    magnitudes = np.linspace(0.1, 1.9, 11)
    img = ImageEnhance.Sharpness(img).enhance(random.uniform(magnitudes[magnitude], magnitudes[magnitude + 1]))
    return img


def cutout(org_img, magnitude=None, img=None):
    img = np.array(img)

    magnitudes = np.linspace(0, 60 / 331, 11)

    img = np.copy(org_img)
    mask_val = img.mean()

    if magnitude is None:
        mask_size = 16
    else:
        mask_size = int(round(img.shape[0] * random.uniform(magnitudes[magnitude], magnitudes[magnitude + 1])))
    top = np.random.randint(0 - mask_size // 2, img.shape[0] - mask_size)
    left = np.random.randint(0 - mask_size // 2, img.shape[1] - mask_size)
    bottom = top + mask_size
    right = left + mask_size

    if top < 0:
        top = 0
    if left < 0:
        left = 0

    img[top:bottom, left:right, :].fill(mask_val)

    img = Image.fromarray(img)

    return img


class Cutout(object):
    def __init__(self, length=16):
        self.length = length

    def __call__(self, img):
        img = np.array(img)

        mask_val = img.mean()

        top = np.random.randint(0 - self.length // 2, img.shape[0] - self.length)
        left = np.random.randint(0 - self.length // 2, img.shape[1] - self.length)
        bottom = top + self.length
        right = left + self.length

        top = 0 if top < 0 else top
        left = 0 if left < 0 else top

        img[top:bottom, left:right, :] = mask_val

        img = Image.fromarray(img)

        return img

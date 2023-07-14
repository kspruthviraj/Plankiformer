###########
# IMPORTS #
###########
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
bcg_noise_components = np.load('/home/EAWAG/kyathasr/GAN_project/GAN_train_ver02/scripts/ver02/train_model/bcg_noise_components.npy')
pillow_noise_components = np.load('/home/EAWAG/kyathasr/GAN_project/GAN_train_ver02/scripts/ver02/train_model/pillow_noise_components.npy')

class CreateDataForPlankton:
    def __init__(self):
        self.Filenames_val = None
        self.Filenames_test = None
        self.Filenames_train = None
        self.classes_int = None
        self.classes = None
        self.Filenames = None
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
        self.class_weights_tensor = None
        self.params = None
        return

    def make_train_test_for_model(self, train_main, prep_data):
        if os.path.exists(train_main.params.outpath + '/Data.pickle'):
            Data = pd.read_pickle(train_main.params.outpath + '/Data.pickle')
            classes = np.load(train_main.params.outpath + '/classes.npy')
            self.class_weights_tensor = torch.load(train_main.params.outpath + '/class_weights_tensor.pt')
            self.Filenames = pd.read_pickle(train_main.params.outpath + 'Files_used_for_training_testing.pickle')

        # if train_main.params.saved_data == 'yes':
        #     Data = pd.read_pickle(train_main.params.outpath + '/Data.pickle')
        #     classes = np.load(train_main.params.outpath + '/classes.npy')
        #     self.class_weights_tensor = torch.load(train_main.params.outpath + '/class_weights_tensor.pt')
        else:
            self.class_weights_tensor = prep_data.class_weights_tensor
            self.Filenames = prep_data.Filenames
            classes = prep_data.classes
            Data = prep_data.Data

        if train_main.params.test_set == 'no':
            if train_main.params.ttkind == 'mixed':
                self.trainFilenames = Data[0]
                trainXimage = Data[1]
                trY = Data[2]
                trainXfeat = Data[9]
                trX = [trainXimage, trainXfeat]
            elif train_main.params.ttkind == 'image' and train_main.params.compute_extrafeat == 'yes':
                self.trainFilenames = Data[0]
                trainXimage = Data[1]
                trY = Data[2]
                trainXfeat = Data[9]
                trX = [trainXimage, trainXfeat]
            elif train_main.params.ttkind == 'feat':
                self.trainFilenames = Data[0]
                trY = Data[2]
                trainXfeat = Data[9]
                trX = [trainXfeat]
            else:
                self.trainFilenames = Data[0]
                trX = Data[1]
                trY = Data[2]

        elif train_main.params.test_set == 'yes':
            if train_main.params.valid_set == 'no':
                if train_main.params.ttkind == 'mixed':
                    self.trainFilenames = Data[0]
                    trainXimage = Data[1]
                    trY = Data[2]
                    self.testFilenames = Data[3]
                    testXimage = Data[4]
                    teY = Data[5]
                    trainXfeat = Data[9]
                    testXfeat = Data[10]
                    trX = [trainXimage, trainXfeat]
                    teX = [testXimage, testXfeat]
                elif train_main.params.ttkind == 'image' and train_main.params.compute_extrafeat == 'yes':
                    self.trainFilenames = Data[0]
                    trainXimage = Data[1]
                    trY = Data[2]
                    self.testFilenames = Data[3]
                    testXimage = Data[4]
                    teY = Data[5]
                    trainXfeat = Data[9]
                    testXfeat = Data[10]
                    trX = [trainXimage, trainXfeat]
                    teX = [testXimage, testXfeat]
                elif train_main.params.ttkind == 'feat':
                    self.trainFilenames = Data[0]
                    trainXfeat = Data[9]
                    trY = Data[2]
                    testXfeat = Data[10]
                    teY = Data[5]
                    trX = [trainXfeat]
                    teX = [testXfeat]
                else:
                    self.trainFilenames = Data[0]
                    trX = Data[1]
                    trY = Data[2]
                    self.testFilenames = Data[3]
                    teX = Data[4]
                    teY = Data[5]

            elif train_main.params.valid_set == 'yes':
                if train_main.params.ttkind == 'mixed':
                    self.trainFilenames = Data[0]
                    trainXimage = Data[1]
                    trY = Data[2]
                    self.testFilenames = Data[3]
                    testXimage = Data[4]
                    teY = Data[5]
                    self.valFilenames = Data[6]
                    valXimage = Data[7]
                    veY = Data[8]
                    trainXfeat = Data[9]
                    testXfeat = Data[10]
                    valXfeat = Data[11]
                    trX = [trainXimage, trainXfeat]
                    teX = [testXimage, testXfeat]
                    veX = [valXimage, valXfeat]
                elif train_main.params.ttkind == 'image' and train_main.params.compute_extrafeat == 'yes':
                    self.trainFilenames = Data[0]
                    trainXimage = Data[1]
                    trY = Data[2]
                    self.testFilenames = Data[3]
                    testXimage = Data[4]
                    teY = Data[5]
                    self.valFilenames = Data[6]
                    valXimage = Data[7]
                    veY = Data[8]
                    trainXfeat = Data[9]
                    testXfeat = Data[10]
                    valXfeat = Data[11]
                    trX = [trainXimage, trainXfeat]
                    teX = [testXimage, testXfeat]
                    veX = [valXimage, valXfeat]
                elif train_main.params.ttkind == 'feat':
                    self.trainFilenames = Data[0]
                    self.testFilenames = Data[3]
                    self.valFilenames = Data[6]
                    trainXfeat = Data[9]
                    testXfeat = Data[10]
                    valXfeat = Data[11]
                    trY = Data[2]
                    teY = Data[5]
                    veY = Data[8]
                    trX = [trainXfeat]
                    teX = [testXfeat]
                    veX = [valXfeat]
                else:
                    self.trainFilenames = Data[0]
                    trX = Data[1]
                    trY = Data[2]
                    self.testFilenames = Data[3]
                    teX = Data[4]
                    teY = Data[5]
                    self.valFilenames = Data[6]
                    veX = Data[7]
                    veY = Data[8]

        classes_int = np.unique(np.argmax(trY, axis=1))
        self.classes = classes
        self.classes_int = classes_int

        if train_main.params.test_set == 'no':
            y_train_max = trY.argmax(axis=1)  # The class that the classifier would bet on
            self.y_train = np.array([classes_int[y_train_max[i]] for i in range(len(y_train_max))], dtype=object)

            data_train = trX.astype(np.float64)
            data_train = 255 * data_train
            self.X_train = data_train.astype(np.uint8)

        elif train_main.params.test_set == 'yes' and train_main.params.valid_set == 'no':
            y_train_max = trY.argmax(axis=1)  # The class that the classifier would bet on
            self.y_train = np.array([classes_int[y_train_max[i]] for i in range(len(y_train_max))], dtype=object)

            y_test_max = teY.argmax(axis=1)  # The class that the classifier would bet on
            self.y_test = np.array([classes_int[y_test_max[i]] for i in range(len(y_test_max))], dtype=object)

            data_train = trX.astype(np.float64)
            data_train = 255 * data_train
            self.X_train = data_train.astype(np.uint8)

            data_test = teX.astype(np.float64)
            data_test = 255 * data_test
            self.X_test = data_test.astype(np.uint8)

        elif train_main.params.test_set == 'yes' and train_main.params.valid_set == 'yes':
            y_train_max = trY.argmax(axis=1)  # The class that the classifier would bet on
            self.y_train = np.array([classes_int[y_train_max[i]] for i in range(len(y_train_max))], dtype=object)

            y_test_max = teY.argmax(axis=1)  # The class that the classifier would bet on
            self.y_test = np.array([classes_int[y_test_max[i]] for i in range(len(y_test_max))], dtype=object)

            y_val_max = veY.argmax(axis=1)  # The class that the classifier would bet on
            self.y_val = np.array([classes_int[y_val_max[i]] for i in range(len(y_val_max))], dtype=object)

            data_train = trX.astype(np.float64)
            data_train = 255 * data_train
            self.X_train = data_train.astype(np.uint8)

            data_test = teX.astype(np.float64)
            data_test = 255 * data_test
            self.X_test = data_test.astype(np.uint8)

            data_val = veX.astype(np.float64)
            data_val = 255 * data_val
            self.X_val = data_val.astype(np.uint8)

        return

    def create_data_loaders(self, train_main):
        self.checkpoint_path = train_main.params.outpath + 'trained_models/' + train_main.params.init_name + '/'
        Path(self.checkpoint_path).mkdir(parents=True, exist_ok=True)

        if train_main.params.test_set == 'yes' and train_main.params.valid_set == 'yes':
            train_dataset = AugmentedDataset(X=self.X_train, y=self.y_train, aug_type=train_main.params.aug_type)
            self.train_dataloader = DataLoader(train_dataset, train_main.params.batch_size, shuffle=False, num_workers=4,
                                               pin_memory=True)

            test_dataset = CreateDataset(X=self.X_test, y=self.y_test)
            self.test_dataloader = DataLoader(test_dataset, train_main.params.batch_size, shuffle=False, num_workers=4,
                                              pin_memory=True)

            val_dataset = CreateDataset(X=self.X_val, y=self.y_val)
            self.val_dataloader = DataLoader(val_dataset, train_main.params.batch_size, shuffle=False, num_workers=4,
                                             pin_memory=True)
        elif train_main.params.test_set == 'no':
            train_dataset = AugmentedDataset(X=self.X_train, y=self.y_train, aug_type=train_main.params.aug_type)
            self.train_dataloader = DataLoader(train_dataset, train_main.params.batch_size, shuffle=False, num_workers=4,
                                               pin_memory=True)
        elif train_main.params.test_set == 'yes' and train_main.params.valid_set == 'no':
            train_dataset = AugmentedDataset(X=self.X_train, y=self.y_train, aug_type=train_main.params.aug_type)
            self.train_dataloader = DataLoader(train_dataset, train_main.params.batch_size, shuffle=False, num_workers=4,
                                               pin_memory=True)

            test_dataset = CreateDataset(X=self.X_test, y=self.y_test)
            self.test_dataloader = DataLoader(test_dataset, train_main.params.batch_size, shuffle=False, num_workers=4,
                                              pin_memory=True)

    def make_train_test_for_others(self, prep_data):
        self.class_weights_tensor = prep_data.class_weights_tensor
        self.Filenames = prep_data.Filenames
        classes = prep_data.classes
        Data = prep_data.Data

        # Data = pd.read_pickle(train_main.params.outpath + '/Data.pickle')
        # classes = np.load(train_main.params.outpath + '/classes.npy')

        self.trainFilenames = Data[0]
        trX = Data[1]
        trY = Data[2]
        self.testFilenames = Data[3]
        teX = Data[4]
        teY = Data[5]
        self.valFilenames = Data[6]
        veX = Data[7]
        veY = Data[8]

        classes_int = np.unique(np.argmax(trY, axis=1))
        self.classes = classes
        self.classes_int = classes_int

        y_train_max = trY.argmax(axis=1)  # The class that the classifier would bet on
        self.y_train = np.array([classes_int[y_train_max[i]] for i in range(len(y_train_max))], dtype=object)

        y_test_max = teY.argmax(axis=1)  # The class that the classifier would bet on
        self.y_test = np.array([classes_int[y_test_max[i]] for i in range(len(y_test_max))], dtype=object)

        y_val_max = veY.argmax(axis=1)  # The class that the classifier would bet on
        self.y_val = np.array([classes_int[y_val_max[i]] for i in range(len(y_val_max))], dtype=object)

        data_train = trX.astype(np.float64)
        data_train = 255 * data_train
        self.X_train = data_train.astype(np.uint8)

        data_test = teX.astype(np.float64)
        data_test = 255 * data_test
        self.X_test = data_test.astype(np.uint8)

        data_val = veX.astype(np.float64)
        data_val = 255 * data_val
        self.X_val = data_val.astype(np.uint8)

        return

    def create_data_loaders_for_others(self, train_main):
        self.checkpoint_path = train_main.params.outpath + 'trained_models/' + train_main.params.init_name + '/'
        Path(self.checkpoint_path).mkdir(parents=True, exist_ok=True)

        train_dataset = AugmentedDataset(X=self.X_train, y=self.y_train, aug_type=train_main.aug_type)
        self.train_dataloader = DataLoader(train_dataset, train_main.params.batch_size, shuffle=False, num_workers=4,
                                           pin_memory=True)

        test_dataset = CreateDataset(X=self.X_test, y=self.y_test)
        self.test_dataloader = DataLoader(test_dataset, train_main.params.batch_size, shuffle=False, num_workers=4,
                                          pin_memory=True)

        val_dataset = CreateDataset(X=self.X_val, y=self.y_val)
        self.val_dataloader = DataLoader(val_dataset, train_main.params.batch_size, shuffle=False, num_workers=4,
                                         pin_memory=True)


class AddPoissonNoise(object):
    def __init__(self, scale_factor_range=(1., 10.)):
        self.scale_factor_range = scale_factor_range

    def __call__(self, tensor):
        scale_factor = torch.FloatTensor(1).uniform_(*self.scale_factor_range).item()
        return (torch.poisson(tensor * scale_factor) / scale_factor).float()

    def __repr__(self):
        return self.__class__.__name__ + '(scale_factor_range={0})'.format(self.scale_factor_range)


class AddSaltPepperNoise(object):
    def __init__(self, salt_prob_range=(0.01, 0.05), pepper_prob_range=(0.01, 0.05)):
        self.salt_prob_range = salt_prob_range
        self.pepper_prob_range = pepper_prob_range

    def __call__(self, tensor):
        salt_prob = torch.FloatTensor(1).uniform_(*self.salt_prob_range).item()
        pepper_prob = torch.FloatTensor(1).uniform_(*self.pepper_prob_range).item()
        rand = torch.rand(tensor.shape)
        tensor[rand < salt_prob] = 1
        tensor[rand > 1 - pepper_prob] = 0
        return tensor.float()

    def __repr__(self):
        return self.__class__.__name__ + '(salt_prob_range={0}, pepper_prob_range={1})'.format(self.salt_prob_range, self.pepper_prob_range)


class AddSpeckleNoise(object):
    def __init__(self, std_range=(0.5, 1.5)):
        self.std_range = std_range

    def __call__(self, tensor):
        std = torch.FloatTensor(1).uniform_(*self.std_range).item()
        noise = torch.randn(tensor.size()) * std
        return (tensor + tensor * noise).float()

    def __repr__(self):
        return self.__class__.__name__ + '(std_range={0})'.format(self.std_range)

# class AddBackgroundNoise(object):
#     def __init__(self, bcg_noise_components, scale_range=(0.1, 1.0), sigma_range=(0.5, 1.5)):
#         self.bcg_noise_components = bcg_noise_components / np.linalg.norm(bcg_noise_components, axis=0, keepdims=True)
#         self.scale_range = scale_range
#         self.sigma_range = sigma_range

#     def __call__(self, tensor):
#         C, H, W = tensor.shape
#         scale = torch.FloatTensor(1).uniform_(*self.scale_range).item()
#         sigma = torch.FloatTensor(1).uniform_(*self.sigma_range).item()
#         noise_pca = np.sum([comp * np.random.normal(scale=scale) for comp in self.bcg_noise_components], axis=0)
#         resized_noise_pca = cv2.resize(noise_pca, (W, H), interpolation=cv2.INTER_LINEAR)
#         resized_noise_pca_3d = np.repeat(resized_noise_pca[:, :, np.newaxis], C, axis=2)
#         smoothed_noise_pca = gaussian_filter(resized_noise_pca_3d, sigma=sigma)
#         smoothed_noise_pca = torch.from_numpy(smoothed_noise_pca).permute(2, 0, 1)  # Make sure to adjust the dimensions
#         noisy_tensor = torch.clamp(tensor + smoothed_noise_pca, 0, 1)
#         return noisy_tensor

#     def __repr__(self):
#         return self.__class__.__name__ + '(scale_range={0}, sigma_range={1})'.format(self.scale_range, self.sigma_range)


# class AddPillowNoise(object):
#     def __init__(self, pillow_noise_components, std_range):
#         self.pillow_noise_components = pillow_noise_components
#         self.std_range = std_range

#     def __call__(self, img):
#         std = np.random.uniform(*self.std_range)
#         noise_pca = torch.stack([
#             torch.sum(torch.tensor([torch.tensor(comp, dtype=torch.float32).reshape(img.shape[1:]) * torch.randn(img.shape[1:]) * std for comp in comp_channel]), dim=0)
#             for comp_channel in self.pillow_noise_components
#         ])
#         noise_pca = noise_pca.numpy()
#         noisy_img = img + noise_pca
#         noisy_img_clipped = np.clip(noisy_img, 0, 255).astype(np.uint8)

#         return Image.fromarray(noisy_img_clipped)

#     def __repr__(self):
#         return self.__class__.__name__ + '(std_range={0})'.format(self.std_range)
    
    
class AddBackgroundNoise(object):
    def __init__(self, bcg_noise_components, scale_range=(0.1, 1.0), intensity_range=(0.5, 1.5)):
        self.bcg_noise_components = bcg_noise_components / np.linalg.norm(bcg_noise_components, axis=0, keepdims=True)
        self.scale_range = scale_range
        self.intensity_range = intensity_range

    def __call__(self, tensor):
        C, H, W = tensor.shape
        scale = torch.FloatTensor(1).uniform_(*self.scale_range).item()
        intensity = torch.FloatTensor(1).uniform_(*self.intensity_range).item()
        
        noise_pca = np.sum([comp * np.random.normal(scale=scale) for comp in self.bcg_noise_components], axis=0)
        
        resized_noise_pca = cv2.resize(noise_pca, (W, H), interpolation=cv2.INTER_LINEAR)
        resized_noise_pca_3d = np.repeat(resized_noise_pca[:, :, np.newaxis], C, axis=2)

        noise_tensor = torch.from_numpy(resized_noise_pca_3d).permute(2, 0, 1).float()  # convert to FloatTensor

        noisy_tensor = torch.clamp(tensor + noise_tensor, 0, 1)
        return noisy_tensor

    def __repr__(self):
        return self.__class__.__name__ + '(scale_range={0}, intensity_range={1})'.format(self.scale_range, self.intensity_range)


class AddPillowNoise(object):
    def __init__(self, pillow_noise_components, scale_range=(0.1, 1.0)):
        self.pillow_noise_components = pillow_noise_components / np.linalg.norm(pillow_noise_components, axis=0, keepdims=True)
        self.scale_range = scale_range

    def __call__(self, tensor):
        C, H, W = tensor.shape
        scale = torch.FloatTensor(1).uniform_(*self.scale_range).item()
        
        noise_pca = np.sum([comp * np.random.normal(scale=scale) for comp in self.pillow_noise_components], axis=0)
        
        resized_noise_pca = cv2.resize(noise_pca, (W, H), interpolation=cv2.INTER_LINEAR)
        # resized_noise_pca_3d = np.dstack([resized_noise_pca] * C) # repeat along third dimension
        
        # print("Before transpose:", resized_noise_pca_3d.shape)
        resized_noise_pca_3d = np.transpose(resized_noise_pca, (2, 0, 1))  # change from (H, W, C) to (C, H, W)
        # print("After transpose:", resized_noise_pca_3d.shape)

        noise_tensor = torch.from_numpy(resized_noise_pca_3d).float()  # convert to FloatTensor
        # print("Input tensor shape:", tensor.shape)
        # print("Noise tensor shape:", noise_tensor.shape)

        noisy_tensor = torch.clamp(tensor + noise_tensor, 0, 1)
        return noisy_tensor

    def __repr__(self):
        return self.__class__.__name__ + '(scale_range={0})'.format(self.scale_range)



class AugmentedDataset(Dataset):
    """Characterizes a dataset for PyTorch"""

    def __init__(self, X, y, aug_type):
        """Initialization"""
        self.X = X
        self.y = y
        self.aug_type = aug_type

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.X)

    def __getitem__(self, index):
        """Generates one sample of data"""
        # Select sample
        image = self.X[index]
        label = self.y[index]
        aug_type = self.aug_type
        if aug_type == 'high':
            X = self.transform4(image)
        elif aug_type == 'medium':
            X = self.transform4(image)
        else:
            X = self.transform4(image)
            # print(' I am USING TRANSFORM_3')
        y = label
        sample = [X, y]
        return sample

    transform4 = T.Compose([
        T.ToPILImage(),
        T.Resize(224),
        T.ToTensor(),  # Convert the image to tensor before adding noise
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        # AddPoissonNoise((1., 10.)),  
        # AddSaltPepperNoise((0.01, 0.1), (0.01, 0.1)),
        # AddSpeckleNoise((0.05, 0.2)),
        AddBackgroundNoise(bcg_noise_components, (0.01, 1.0), (0.01, 1.0)),
        AddPillowNoise(pillow_noise_components, (0.1, 1.0)),
        T.ColorJitter(brightness=0.1, hue=0.1),
        T.GaussianBlur(kernel_size=(1, 3), sigma=(0.1, 0.5)),
        T.RandomRotation(degrees=(0, 180)),
        T.RandomAffine(degrees=(30, 90), translate=(0.1, 0.3), scale=(0.5, 0.9))
    ])
    transform4_y = T.Compose([T.ToTensor()])

    
# class AugmentedDataset(Dataset):
#     """Characterizes a dataset for PyTorch"""

#     def __init__(self, X, y, aug_type):
#         """Initialization"""
#         self.X = X
#         self.y = y
#         self.aug_type = aug_type

#     def __len__(self):
#         """Denotes the total number of samples"""
#         return len(self.X)

#     def __getitem__(self, index):
#         """Generates one sample of data"""
#         # Select sample
#         image = self.X[index]
#         label = self.y[index]
#         aug_type = self.aug_type
#         if aug_type == 'high':
#             X = self.transform3(image)
#         elif aug_type == 'medium':
#             X = self.transform3(image)
#         else:
#             X = self.transform3(image)
#             # print(' I am USING TRANSFORM_3')
#         y = label
#         sample = [X, y]
#         return sample

#     # transform1 = T.Compose([
#     #         T.ToPILImage(),
#     #         T.Resize(224),
#     #         T.RandomHorizontalFlip(),
#     #         T.RandomVerticalFlip(),
#     #         # T.RandAugment(),
#     #         T.TrivialAugmentWide(),
#     #         # T.AugMix(),
#     #         # T.RandomResizedCrop(size=(224, 224)),
#     #         T.RandomErasing(),
#     #         T.Grayscale(),
#     #         T.RandomInvert(),
#     #         T.RandomAutocontrast(),
#     #         T.RandomEqualize(),
#     #         T.RandomAdjustSharpness(sharpness_factor=2),
#     #         T.ColorJitter(brightness=0.3, hue=0.3),
#     #         T.GaussianBlur(kernel_size=(1, 5), sigma=(0.1, 2)),
#     #         T.RandomPerspective(distortion_scale=0.8, p=0.1),
#     #         T.RandomRotation(degrees=(0, 180)),
#     #         T.RandomAffine(degrees=(30, 90), translate=(0.1, 0.3), scale=(0.5, 0.9)),
#     #         T.ToTensor()])
#     # transform1_y = T.Compose([T.ToTensor()])
#     #
#     # transform2 = T.Compose([
#     #         T.ToPILImage(),
#     #         T.Resize(224),
#     #         T.RandomHorizontalFlip(),
#     #         T.RandomVerticalFlip(),
#     #         # T.RandAugment(),
#     #         # T.TrivialAugmentWide(),
#     #         # T.AugMix(),
#     #         # T.RandomResizedCrop(size=(224, 224)),
#     #         # T.RandomErasing(),
#     #         # T.Grayscale(),
#     #         # T.RandomInvert(),
#     #         T.RandomAutocontrast(),
#     #         T.RandomEqualize(),
#     #         T.RandomAdjustSharpness(sharpness_factor=2),
#     #         T.ColorJitter(brightness=0.3, hue=0.3),
#     #         T.GaussianBlur(kernel_size=(1, 5), sigma=(0.1, 2)),
#     #         # T.RandomPerspective(distortion_scale=0.8, p=0.1),
#     #         T.RandomRotation(degrees=(0, 180)),
#     #         T.RandomAffine(degrees=(30, 90), translate=(0.1, 0.3), scale=(0.5, 0.9)),
#     #         T.ToTensor()])
#     # transform2_y = T.Compose([T.ToTensor()])

#     transform3 = T.Compose([
#             T.ToPILImage(),
#             T.Resize(224),
#             T.RandomHorizontalFlip(),
#             T.RandomVerticalFlip(),
#             # T.RandAugment(),
#             # T.TrivialAugmentWide(),
#             # T.AugMix(),
#             # T.RandomResizedCrop(size=(224, 224)),
#             # T.RandomErasing(),
#             # T.Grayscale(),
#             # T.RandomInvert(),
#             # T.RandomAutocontrast(),
#             # T.RandomEqualize(),
#             # T.RandomAdjustSharpness(sharpness_factor=2),
#             # T.ColorJitter(brightness=0.3, hue=0.3),
#             # T.GaussianBlur(kernel_size=(1, 5), sigma=(0.1, 2)),
#             # T.RandomPerspective(distortion_scale=0.8, p=0.1),
#             T.RandomRotation(degrees=(0, 180)),
#             T.RandomAffine(degrees=(30, 90), translate=(0.1, 0.3), scale=(0.5, 0.9)),
#             T.ToTensor()])
#     transform3_y = T.Compose([T.ToTensor()])


class CreateDataset(Dataset):
    """Characterizes a dataset for PyTorch"""

    def __init__(self, X, y):
        'Initialization'
        self.X = X
        self.y = y

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.X)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        image = self.X[index]
        label = self.y[index]
        X = self.transform(image)
        y = label
        #         y = self.transform_y(label)
        #         sample = {'image': X, 'label': label}
        sample = [X, y]
        return sample

    transform = T.Compose([
        T.ToPILImage(),
        T.Resize(224),
        T.ToTensor()])
    transform_y = T.Compose([T.ToTensor()])

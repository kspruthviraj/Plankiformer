import glob
import math
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
# import the necessary packages
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.class_weight import compute_class_weight  # Added by SK


def compute_extrafeat_function(df):
    dfExtraFeat = pd.DataFrame()
    dfFeatExtra1 = pd.DataFrame(columns=['width', 'height', 'w_rot', 'h_rot', 'angle_rot', 'aspect_ratio_2',
                                         'rect_area', 'contour_area', 'contour_perimeter', 'extent',
                                         'compactness', 'formfactor', 'hull_area', 'solidity_2', 'hull_perimeter',
                                         'ESD', 'Major_Axis', 'Minor_Axis', 'Angle', 'Eccentricity1', 'Eccentricity2',
                                         'Convexity', 'Roundness'])
    dfFeatExtra2 = pd.DataFrame(columns=['m00', 'm10', 'm01', 'm20', 'm11', 'm02', 'm30', 'm21', 'm12', 'm03',
                                         'mu20', 'mu11', 'mu02', 'mu30', 'mu21', 'mu12', 'mu03', 'nu20', 'nu11',
                                         'nu02', 'nu30', 'nu21', 'nu12', 'nu03'])

    for i in range(len(df)):
        image = cv2.imread(df.filename[i])
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.blur(gray, (2, 2))  # blur the image
        ret, thresh = cv2.threshold(blur, 35, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # Find the largest contour
        cnt = max(contours, key=cv2.contourArea)
        # Bounding rectangle
        x, y, width, height = cv2.boundingRect(cnt)
        # Rotated rectangle
        rot_rect = cv2.minAreaRect(cnt)
        rot_box = cv2.boxPoints(rot_rect)
        rot_box = np.int0(rot_box)
        w_rot = rot_rect[1][0]
        h_rot = rot_rect[1][1]
        angle_rot = rot_rect[2]
        # Find Image moment of largest contour
        M = cv2.moments(cnt)
        # Find centroid
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        # Find the Aspect ratio or elongation --It is the ratio of width to height of bounding rect of the object.
        aspect_ratio = float(width) / height
        # Rectangular area
        rect_area = width * height
        # Area of the contour
        contour_area = cv2.contourArea(cnt)
        # Perimeter of the contour
        contour_perimeter = cv2.arcLength(cnt, True)
        # Extent --Extent is the ratio of contour area to bounding rectangle area
        extent = float(contour_area) / rect_area
        # Compactness -- from MATLAB
        compactness = (np.square(contour_perimeter)) / (4 * np.pi * contour_area)
        # Form factor
        formfactor = (4 * np.pi * contour_area) / (np.square(contour_perimeter))
        # Convex hull points
        hull_2 = cv2.convexHull(cnt)
        # Convex Hull Area
        hull_area = cv2.contourArea(hull_2)
        # solidity --Solidity is the ratio of contour area to its convex hull area.
        solidity = float(contour_area) / hull_area
        # Hull perimeter
        hull_perimeter = cv2.arcLength(hull_2, True)
        # Equivalent circular Diameter-is the diameter of the circle whose area is same as the contour area.
        ESD = np.sqrt(4 * contour_area / np.pi)
        # Orientation, Major Axis, Minos axis -Orientation is the angle at which object is directed
        (x1, y1), (Major_Axis, Minor_Axis), angle = cv2.fitEllipse(cnt)
        # Eccentricity or ellipticity.
        Eccentricity1 = Minor_Axis / Major_Axis
        Mu02 = M['m02'] - (cy * M['m01'])
        Mu20 = M['m20'] - (cx * M['m10'])
        Mu11 = M['m11'] - (cx * M['m01'])
        Eccentricity2 = (np.square(Mu02 - Mu20)) + 4 * Mu11 / contour_area
        # Convexity
        Convexity = hull_perimeter / contour_perimeter
        # Roundness
        Roundness = (4 * np.pi * contour_area) / (np.square(hull_perimeter))

        dfFeatExtra1.loc[i] = [width, height, w_rot, h_rot, angle_rot, aspect_ratio, rect_area, contour_area,
                               contour_perimeter, extent, compactness, formfactor, hull_area, solidity,
                               hull_perimeter, ESD, Major_Axis, Minor_Axis, angle, Eccentricity1,
                               Eccentricity2, Convexity, Roundness]
        dfFeatExtra2.loc[i] = M

    dfExtraFeat = pd.concat([dfFeatExtra1, dfFeatExtra2], axis=1)

    return dfExtraFeat


def ResizeWithoutProportions(im, desired_size):
    new_im = im.resize((desired_size, desired_size), Image.LANCZOS)
    rescaled = 1
    return new_im, rescaled


def ResizeWithProportions(im, desired_size):
    """
    Take and image and resize it to a square of the desired size.
    0) If any dimension of the image is larger than the desired size, shrink until the image can fully fit in the desired size
    1) Add black paddings to create a square
    """

    old_size = im.size
    largest_dim = max(old_size)
    smallest_dim = min(old_size)

    # If the image dimensions are very different, reducing the larger one to `desired_size` can make the other
    # dimension too small. We impose that it be at least 4 pixels.
    if desired_size * smallest_dim / largest_dim < 4:
        print('Image size: ({},{})'.format(largest_dim, smallest_dim))
        print('Desired size: ({},{})'.format(desired_size, desired_size))
        raise ValueError(
            'Images are too extreme rectangles to be reduced to this size. Try increasing the desired image size.')

    rescaled = 0  # This flag tells us whether there was a rescaling of the image (besides the padding). We can use
    # it as feature for training.

    # 0) If any dimension of the image is larger than the desired size, shrink until the image can fully fit in the
    # desired size
    if max(im.size) > desired_size:
        ratio = float(desired_size) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])
        # print('new_size:',new_size)
        sys.stdout.flush()
        im = im.resize(new_size, Image.LANCZOS)
        rescaled = 1

    # 1) Add black paddings to create a square
    new_im = Image.new("RGB", (desired_size, desired_size), color=0)
    new_im.paste(im, ((desired_size - im.size[0]) // 2,
                      (desired_size - im.size[1]) // 2))

    return new_im, rescaled


def LoadImage(filename, L=None, resize_images=None, show=False):
    """ Loads one image, and rescales it to size L.
    The pixel values are between 0 and 255, instead of between 0 and 1, so they should be normalized outside of the function
    """

    image = Image.open(filename)
    # Set image's largest dimension to target size, and fill the rest with black pixels
    if resize_images == 0 or resize_images is None:
        rescaled = 0
    elif resize_images == 1:
        image, rescaled = ResizeWithProportions(image,
                                                L)  # width and height are assumed to be the same (assertion at the
        # beginning)
    elif resize_images == 2:
        image, rescaled = ResizeWithoutProportions(image,
                                                   L)  # width and height are assumed to be the same (assertion at
        # the beginning)
    npimage = np.array(image.copy(), dtype=np.float32)
    # 	npimage = cv2.cvtColor(npimage,cv2.COLOR_GRAY2RGB) ## FOR WHOI and KAGGLE dataset
    if show:
        image.show()
    image.close()

    return npimage, rescaled, filename


def LoadImages(datapaths, L, resize_images=None, training_data=True):
    """
    Uses the data in datapath to create a DataFrame with images only.
    This cannot be a particular case of the mixed loading, because the mixed depends on the files written in the features.tsv file, whereas here we fetch the images directly.

    Arguments:
    datapath 	 - the path where the data is stored. Inside datapath, we expect to find directories with the names of the classes
    L 			 - images are rescaled to a square of size LxL (maintaining proportions)
    class_select - a list of the classes to load. If None (default), loads all classes
    training_data- a boolean variable to decide the structure of the directories
    Output:
    df 			 - a dataframe with classname, npimage, rescaled.
    """

    df = pd.DataFrame()

    # The following condition is because the taxonomists used different directory structures
    names1 = '/training_data/*.jp*g' if training_data == True else '/*.jp*g'
    names2 = '/training_data/*.png' if training_data == True else '/*.png'
    names3 = '/training_data/*.ti*f' if training_data == True else '/*.ti*f'

    classImages = []
    for idp in range(len(datapaths)):
        classImages.extend(glob.glob(datapaths[idp] + '/' + names1) + glob.glob(
            datapaths[idp] + '/' + names2) + glob.glob(datapaths[idp] + '/' + names3))

    # Create an empty dataframe for this class
    dfClass = pd.DataFrame(columns=['filename', 'npimage'])
    print('test: ({})'.format(len(classImages)))
    for i, imageName in enumerate(classImages):
        npimage, rescaled, filename = LoadImage(imageName, L, resize_images)
        dfClass.loc[i] = [filename, npimage]
    df = pd.concat([df, dfClass], axis=0)

    df.npimage = df.npimage / 255.0

    df = df.sample(frac=1).reset_index(drop=True)
    return df


def LoadImageList(im_names, L, resize_images, show=False):
    """
    Function that loads a list of images given in im_names, and returns
    them in a numpy format that can be used by the classifier.
    """
    npimages = np.ndarray((len(im_names), L, L, 3))

    for i, im_name in enumerate(im_names):
        npimage, rescaled, filename = LoadImage(im_name, L, resize_images, show)
        npimages[i] = npimage
    return npimages / 255.0


class Cdata:

    def __init__(self, datapath, L=None, compute_extrafeat=None, resize_images=None):
        self.classifier = None
        self.class_select = None
        self.filenames = None
        self.Xfeat = None
        self.Ximage = None
        self.datapath = datapath
        self.L = L
        self.compute_extrafeat = compute_extrafeat
        self.resize_images = resize_images
        self.df = None
        self.y = None
        self.X = None
        self.Load(self.datapath, self.L, self.class_select, self.classifier, self.compute_extrafeat, self.resize_images)
        return

    def Load(self, datapaths, L, class_select, classifier, compute_extrafeat, resize_images):
        """
        Loads dataset
        For the moment, only mixed data. Later, also pure images or pure features.
        """
        self.L = L
        self.datapath = datapaths
        self.class_select = class_select
        self.classifier = classifier
        self.compute_extrafeat = compute_extrafeat
        self.resize_images = resize_images

        self.df = LoadImages(datapaths, L, resize_images)
        if compute_extrafeat == 'yes':
            dfExtraFeat = compute_extrafeat_function(self.df)
            self.df = pd.concat([self.df, dfExtraFeat], axis=1)

        self.CreateXy()  # Creates X and y, i.e. features and labels
        return

    def CreateXy(self):
        """
        Creates features and target
        - removing the evidently junk columns.
        - allowing to access images and features separately and confortably
        """

        self.filenames = self.df.filename
        self.X = self.df.drop(columns=['classname', 'url', 'filename', 'file_size', 'timestamp'], errors='ignore')
        # 		self.X = self.df.drop(columns=['classname','url','file_size','timestamp'], errors='ignore')

        self.Ximage = self.X.npimage
        self.Xfeat = self.X.drop(columns=['npimage'], errors='ignore')

        return


def unique_cols(df):
    """ Returns one value per column, stating whether all the values are the same"""
    a = df.to_numpy()  # df.values (pandas<0.24)
    return (a[0] == a[1:]).all(0)


def DropCols(X, cols):
    """
    Gets rid of the columns cols from the dataframe X.
    cols is a list with the columns names
    """
    return X.drop(columns=cols, errors='ignore')


def RemoveUselessCols(df):
    """ Removes columns with no information from dataframe """
    # Select all columns except image
    morecols = []
    cols = df.columns.tolist()

    if 'npimage' in cols:
        cols.remove('npimage')
        morecols = ['npimage']

    # Remove all columns with all equal values
    badcols = np.where(unique_cols(df[cols]) == True)[0].tolist()
    badcols.reverse()  # I reverse, because otherwise the indices get messed up when I use del

    for i in badcols:
        del cols[i]

    cols = morecols + cols

    return df[cols]


# def prep_data_for_others():
#     data = Cdata.Load(datapaths, L, class_select, compute_extrafeat, resize_images)
#

class CTrainTestSet:
    """
    A class for extracting train and test sets from the original dataset, and preprocessing them.
    """

    def __init__(self, X, y, filenames, ttkind='image', classifier=None, balance_weight=None, rescale=False,
                 testSplit=0.2, valid_set=None, test_set=None, compute_extrafeat=None, random_state=12345):
        """
        X and y are dataframes with features and labels
        """

        self.valXfeat = None
        self.testXfeat = None
        self.trainXfeat = None
        self.valXimage = None
        self.testXimage = None
        self.trainXimage = None
        self.testFilenames = None
        self.trainFilenames = None
        self.testY = None
        self.trainX = None
        self.valY = None
        self.trainY = None
        self.valX = None
        self.valFilenames = None
        self.testX = None
        self.class_weights_tensor = None
        self.lb = None
        self.ttkind = ttkind
        self.testSplit = testSplit
        self.valid_set = valid_set
        self.test_set = test_set
        self.random_state = random_state
        self.classifier = classifier
        self.balance_weight = balance_weight
        self.compute_extrafeat = compute_extrafeat

        # Take care of the labels
        self.filenames = filenames
        self.X = self.ImageNumpyFromMixedDataframe(X)

        return

    def ImageNumpyFromMixedDataframe(self, X=None):
        """ Returns a numpy array of the shape (nexamples, L, L, channels)"""
        if X is None:
            X = self.X

        # The column containing npimage
        im_col = [i for i, col in enumerate(X.columns) if col == 'npimage'][0]

        return np.array([X.to_numpy()[i, im_col] for i in range(len(X.index))])


if __name__ == '__main__':
    pass

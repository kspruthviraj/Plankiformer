import glob
import math
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
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
    0) If any dimension of the image is larger than the desired size, shrink until the image can fully fit in the
    desired size
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

    rescaled = 0  # This flag tells us whether there was a rescaling of the image (besides the padding).
    # We can use it as feature for training.

    # 0) If any dimension of the image is larger than the desired size, shrink until the image can fully fit
    # in the desired size
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


def ReduceClasses(datapaths, class_select, classifier):
    print('datapaths:', datapaths)
    # allClasses = [ name for name in os.listdir(datapaths) if os.path.isdir(os.path.join(datapaths, name)) ]
    print('datapaths:{}'.format(datapaths))
    allClasses = list(set([name for idata in range(len(datapaths)) for name in os.listdir(datapaths[idata]) if
                           os.path.isdir(os.path.join(datapaths[idata], name))]))
    print('classes from datapaths:', allClasses)

    if classifier == 'multi':
        if class_select is None:
            class_select = allClasses
        else:
            if not set(class_select).issubset(allClasses):
                print('Some of the classes input by the user are not present in the dataset.')
                print('class_select:', class_select)
                print('all  classes:', allClasses)
                raise ValueError
        return class_select
    elif classifier == 'binary':
        class_select = class_select
        return class_select
    elif classifier == 'versusall':
        class_select_binary = class_select
        class_select = allClasses

        return class_select, class_select_binary


# learning rate schedule
def step_decay(epoch):
    initial_lrate = 0.01
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lrate


def LoadMixed(datapaths, L, class_select=None, classifier=None, resize_images=None, alsoImages=True,
              training_data=True):
    '''
    Uses the data in datapath to create a DataFrame with images and features.
    For each class, we read a tsv file with the features. This file also contains the name of the corresponding image, which we fetch and resize.
    For each line in the tsv file, we then have all the features in the tsv, plus class name, image (as numpy array), and a binary variable stating whether the image was resized or not.
    Assumes a well-defined directory structure.

    Arguments:
    datapaths 	  - list with the paths where the data is stored. Inside each datapath, we expect to find directories with the names of the classes
    L 			  - images are rescaled to a square of size LxL (maintaining proportions)
    class_select  - a list of the classes to load. If None (default), loads all classes
    alsoImages    - flag that tells whether to only load features, or features+images
    training_data - flag for adding a subdirectory called training_data
    Output:
    df 		 	  - a dataframe with classname, npimage, rescaled, and all the columns contained in features.tsv
    '''
    training_data_dir = '/training_data/' if training_data == True else '/'

    df = pd.DataFrame()
    dfA = pd.DataFrame()
    dfB = pd.DataFrame()

    if classifier == 'binary':
        class_select = ReduceClasses(datapaths, class_select, classifier)  # Decide whether to use all available classes
        if len(class_select) == 2:
            # 			print('LENGTH of CLASS_SELECT: {}'.format(len(class_select)))
            # Loop for data loading
            for c in class_select:  # Loop over the classes
                # Read from tsv file, and create column with full path to the image
                dfFeat = pd.DataFrame()
                for idp in range(len(datapaths)):
                    try:  # It could happen that a class is contained in one datapath but not in the others
                        dftemp = pd.read_csv(datapaths[idp] + c + '/features.tsv', sep='\t')
                        dftemp['filename'] = [datapaths[idp] + c + training_data_dir + os.path.basename(dftemp.url[ii])
                                              for ii in range(len(dftemp))]
                        dfFeat = pd.concat([dfFeat, dftemp], axis=0, sort=True)
                    except:
                        pass
                print('class: {} ({})'.format(c, len(dfFeat)))
                # Each line in features.tsv should be associated with classname (and image, if the options say it's
                # true)
                for index, row in dfFeat.iterrows():
                    if alsoImages:
                        npimage, rescaled, filename = LoadImage(row.filename, L, resize_images)

                        dftemp = pd.DataFrame([[c, npimage, rescaled] + row.to_list()],
                                              columns=['classname', 'npimage', 'rescaled'] + dfFeat.columns.to_list())
                    else:  # alsoImages is False here
                        dftemp = pd.DataFrame([[c] + row.to_list()], columns=['classname'] + dfFeat.columns.to_list())

                    df = pd.concat([df, dftemp], axis=0, sort=True)
            # If images were loaded, scale the raw pixel intensities to the range [0, 1]
            if alsoImages:
                df.npimage = df.npimage / 255.0
        else:
            print('Check if the number of classes is equal to TWO (2)')
            raise ValueError

    elif classifier == 'multi':
        class_select = ReduceClasses(datapaths, class_select, classifier)  # Decide whether to use all available classes
        if len(class_select) > 2:
            # 			print('LENGTH of CLASS_SELECT: {}'.format(len(class_select)))
            # Loop for data loading
            for c in class_select:  # Loop over the classes
                # Read from tsv file, and create column with full path to the image
                dfFeat = pd.DataFrame()
                for idp in range(len(datapaths)):
                    try:  # It could happen that a class is contained in one datapath but not in the others
                        dftemp = pd.read_csv(datapaths[idp] + c + '/features.tsv', sep='\t')
                        dftemp['filename'] = [datapaths[idp] + c + training_data_dir + os.path.basename(dftemp.url[ii])
                                              for ii in range(len(dftemp))]
                        dfFeat = pd.concat([dfFeat, dftemp], axis=0, sort=True)
                    except:
                        pass
                print('class: {} ({})'.format(c, len(dfFeat)))
                # Each line in features.tsv should be associated with classname (and image, if the options say it's
                # true)
                for index, row in dfFeat.iterrows():
                    if alsoImages:
                        npimage, rescaled, filename = LoadImage(row.filename, L, resize_images)

                        dftemp = pd.DataFrame([[c, npimage, rescaled] + row.to_list()],
                                              columns=['classname', 'npimage', 'rescaled'] + dfFeat.columns.to_list())
                    else:  # alsoImages is False here
                        dftemp = pd.DataFrame([[c] + row.to_list()], columns=['classname'] + dfFeat.columns.to_list())
                    df = pd.concat([df, dftemp], axis=0, sort=True)
            # If images were loaded, scale the raw pixel intensities to the range [0, 1]
            if alsoImages:
                df.npimage = df.npimage / 255.0
        else:
            print('Check if the number of classes is greater than TWO (2)')
            raise ValueError

    elif classifier == 'versusall':
        class_select, class_select_binary = ReduceClasses(datapaths, class_select, classifier)
        # 			print('Names of binary class: {}'.format(class_select_binary))
        # 			print('total binary class: {}'.format(len(class_select_binary)))
        # Get names of images belonging to this class, from all the data paths
        class_select = [classes for classes in class_select if classes not in class_select_binary]
        negative_class_name = 'NOT_' + '_or_'.join(class_select_binary)
        if len(class_select_binary) > 1:
            positive_class_name = '_or_'.join(class_select_binary)
        else:
            positive_class_name = '_'.join(class_select_binary)
        # 			print('POSITIVE class: {}'.format(positive_class_name))
        # 			print('NEGATIVE class: {}'.format(negative_class_name))
        # Loop for data loading
        Total_positive_class = 0
        for c in class_select_binary:  # Loop over the classes
            # Read from tsv file, and create column with full path to the image
            dfFeatB = pd.DataFrame()
            for idp in range(len(datapaths)):
                try:  # It could happen that a class is contained in one datapath but not in the others
                    dftempB = pd.read_csv(datapaths[idp] + c + '/features.tsv', sep='\t')
                    dftempB['filename'] = [datapaths[idp] + c + training_data_dir + os.path.basename(dftempB.url[ii])
                                           for ii in range(len(dftempB))]
                    dfFeatB = pd.concat([dfFeatB, dftempB], axis=0, sort=True)
                except:
                    pass
            print('Individual POSITIVE class: {} ({})'.format(c, len(dfFeatB)))
            Total_positive_class = Total_positive_class + len(dfFeatB)
            dftempimageBs = []
            # Each line in features.tsv should be associated with classname (and image, if the options say it's true)
            for index, row in dfFeatB.iterrows():
                if alsoImages:
                    npimageB, rescaledB, filenameB = LoadImage(row.filename, L, resize_images)
                    dftempB = pd.DataFrame([[positive_class_name, npimageB, rescaledB] + row.to_list()],
                                           columns=['classname', 'npimage', 'rescaled'] + dfFeatB.columns.to_list())
                # 						image.close()
                else:  # alsoImages is False here
                    dftempB = pd.DataFrame([[positive_class_name] + row.to_list()],
                                           columns=['classname'] + dfFeatB.columns.to_list())
                dfB = pd.concat([dfB, dftempB], axis=0, sort=True)
        print('Total POSITIVE class: {} ({})'.format(positive_class_name, Total_positive_class))
        # 			If images were loaded, scale the raw pixel intensities to the range [0, 1]
        if alsoImages:
            dfB.npimage = dfB.npimage / 255.0
        concatenated_list = pd.DataFrame()

        # Loop for data loading
        for c in class_select:  # Loop over the classes
            # Read from tsv file, and create column with full path to the image
            dfFeatA = pd.DataFrame()
            for idp in range(len(datapaths)):
                try:  # It could happen that a class is contained in one datapath but not in the others
                    dftempA = pd.read_csv(datapaths[idp] + c + '/features.tsv', sep='\t')
                    dftempA['filename'] = [datapaths[idp] + c + training_data_dir + os.path.basename(dftempA.url[ii])
                                           for ii in range(len(dftempA))]
                    dfFeatA = pd.concat([dfFeatA, dftempA], axis=0, sort=True)
                    # 						print('original class: {} ({})'.format(c, len(dfFeatA)))

                    # 						dfFeatA = dfFeatA.sample(frac=0.5, replace=True, random_state=1)

                    # 						print('50% class: {} ({})'.format(c, len(dfFeatA)))
                    concatenated_list = pd.concat([concatenated_list, dfFeatA])
                # 						print('ConCatenated:({})'.format(len(concatenated_list)))
                except:
                    pass

        Only_dirName = []
        Only_dirName1 = list(concatenated_list.filename)
        for i in range(len(Only_dirName1)):
            Only_dirName.append(os.path.basename(Path(Only_dirName1[i]).parents[1]))
        indices = np.arange(len(Only_dirName))
        y_train, y_test, y_train_idx, y_test_idx = train_test_split(Only_dirName, indices,
                                                                    test_size=len(dfB) / len(Only_dirName) * 1.5,
                                                                    random_state=42, stratify=Only_dirName)
        Negative_class = concatenated_list.iloc[y_test_idx]

        # 			Negative_class = concatenated_list.sample(n=len(dfB), replace=True)
        print('Total NEGATIVE class: {} ({})'.format(negative_class_name, len(Negative_class)))
        # Each line in features.tsv should be associated with classname (and image, if the options say it's true)
        for index, row in Negative_class.iterrows():
            if alsoImages:
                npimageA, rescaledA, filenameA = LoadImage(row.filename, L, resize_images)
                dftempA = pd.DataFrame([[negative_class_name, npimageA, rescaledA] + row.to_list()],
                                       columns=['classname', 'npimage', 'rescaled'] + dfFeatA.columns.to_list())
            else:  # alsoImages is False here
                dftempA = pd.DataFrame([[negative_class_name] + row.to_list()],
                                       columns=['classname'] + dfFeatA.columns.to_list())
            dfA = pd.concat([dfA, dftempA], axis=0, sort=True)
        # If images were loaded, scale the raw pixel intensities to the range [0, 1]
        if alsoImages:
            dfA.npimage = dfA.npimage / 255.0
        df = pd.concat([dfB, dfA], axis=0)  # Concatenate Selected class and all other class
    df = df.sample(frac=1).reset_index(drop=True)
    return df


def LoadImage(filename, L=None, resize_images=None, show=False):
    ''' Loads one image, and rescales it to size L.
    The pixel values are between 0 and 255, instead of between 0 and 1, so they should be normalized outside of the function
    '''

    image = Image.open(filename)
    # Set image's largest dimension to target size, and fill the rest with black pixels
    if resize_images == 0 or resize_images is None:
        rescaled = 0
    elif resize_images == 1:
        image, rescaled = ResizeWithProportions(image,
                                                L)  # width and height are assumed to be the same (assertion at the beginning)
    elif resize_images == 2:
        image, rescaled = ResizeWithoutProportions(image,
                                                   L)  # width and height are assumed to be the same (assertion at the beginning)
    npimage = np.array(image.copy(), dtype=np.float32)
    # 	npimage = cv2.cvtColor(npimage,cv2.COLOR_GRAY2RGB) ## FOR WHOI and KAGGLE dataset
    if show:
        image.show()
    image.close()

    return npimage, rescaled, filename


def LoadImages(datapaths, L, class_select=None, classifier=None, resize_images=None, training_data=True):
    """
    Uses the data in datapath to create a DataFrame with images only.
    This cannot be a particular case of the mixed loading, because the mixed depends on the files written in the
    features.tsv file, whereas here we fetch the images directly.

    Arguments:
    datapath 	 - the path where the data is stored. Inside datapath, we expect to find directories with the names of
    the classes
    L 			 - images are rescaled to a square of size LxL (maintaining proportions)
    class_select - a list of the classes to load. If None (default), loads all classes
    training_data- a boolean variable to decide the structure of the directories
    Output:
    df 			 - a dataframe with classname, npimage, rescaled.
    """

    df = pd.DataFrame()
    dfA = pd.DataFrame()
    dfB = pd.DataFrame()

    # The following condition is because the taxonomists used different directory structures
    names1 = '/training_data/*.jp*g' if training_data == True else '/*.jp*g'
    names2 = '/training_data/*.png' if training_data == True else '/*.png'
    names3 = '/training_data/*.ti*f' if training_data == True else '/*.ti*f'
    # 	print(names2)
    if classifier == 'multi':
        class_select = ReduceClasses(datapaths, class_select, classifier)  # Decide whether to use all available classes
        if len(class_select) > 2:
            for c in class_select:
                # Get names of images belonging to this class, from all the data paths
                classImages = []
                for idp in range(len(datapaths)):
                    classImages.extend(glob.glob(datapaths[idp] + '/' + c + '/' + names1) + glob.glob(
                        datapaths[idp] + '/' + c + '/' + names2) + glob.glob(datapaths[idp] + '/' + c + '/' + names3))
                # Create an empty dataframe for this class
                dfClass = pd.DataFrame(columns=['filename', 'classname', 'npimage'])
                print('class: {} ({})'.format(c, len(classImages)))
                for i, imageName in enumerate(classImages):
                    npimage, rescaled, filename = LoadImage(imageName, L, resize_images)
                    dfClass.loc[i] = [filename, c, npimage]
                df = pd.concat([df, dfClass], axis=0)
            df.npimage = df.npimage / 255.0
        else:
            print('Check if the number of classes is greater than TWO (2)')
            raise ValueError

    if classifier == 'binary':
        class_select = ReduceClasses(datapaths, class_select, classifier)  # Decide whether to use all available classes
        if len(class_select) == 2:
            for c in class_select:
                # Get names of images belonging to this class, from all the data paths
                classImages = []
                for idp in range(len(datapaths)):
                    classImages.extend(glob.glob(datapaths[idp] + '/' + c + '/' + names1) + glob.glob(
                        datapaths[idp] + '/' + c + '/' + names2) + glob.glob(datapaths[idp] + '/' + c + '/' + names3))
                # Create an empty dataframe for this class
                dfClass = pd.DataFrame(columns=['filename', 'classname', 'npimage'])
                print('class: {} ({})'.format(c, len(classImages)))
                for i, imageName in enumerate(classImages):
                    npimage, rescaled, filename = LoadImage(imageName, L, resize_images)
                    dfClass.loc[i] = [filename, c, npimage]
                df = pd.concat([df, dfClass], axis=0)
            df.npimage = df.npimage / 255.0
        else:
            print('Check if the number of classes is equal to TWO (2)')
            raise ValueError

    elif classifier == 'versusall':
        class_select, class_select_binary = ReduceClasses(datapaths, class_select, classifier)
        class_select = [classes for classes in class_select if classes not in class_select_binary]

        negative_class_name = 'NOT_' + '_or_'.join(class_select_binary)
        if len(class_select_binary) > 1:
            positive_class_name = '_or_'.join(class_select_binary)
        else:
            positive_class_name = '_'.join(class_select_binary)
        concatenated_list = []
        Total_positive_class = 0
        for c in class_select_binary:
            # Get names of images belonging to this class, from all the data paths
            BinaryclassImages = []
            for bidp in range(len(datapaths)):
                BinaryclassImages.extend(glob.glob(datapaths[bidp] + '/' + c + '/' + names1) + glob.glob(
                    datapaths[bidp] + '/' + c + '/' + names2) + glob.glob(datapaths[bidp] + '/' + c + '/' + names3))
            # Create an empty dataframe for this class
            dfBClass = pd.DataFrame(columns=['filename', 'classname', 'npimage'])
            print('Individual POSITIVE class: {} ({})'.format(c, len(BinaryclassImages)))
            Total_positive_class = Total_positive_class + len(BinaryclassImages)
            for bi, BinaryimageName in enumerate(BinaryclassImages):
                npimageB, rescaled, filenameB = LoadImage(BinaryimageName, L, resize_images)
                dfBClass.loc[bi] = [filenameB, positive_class_name, npimageB]
            dfB = pd.concat([dfB, dfBClass], axis=0)
        print('Total POSITIVE class: {} ({})'.format(positive_class_name, Total_positive_class))
        dfB.npimage = dfB.npimage / 255.0

        for c in class_select:
            classImages = []
            for idp in range(len(datapaths)):
                classImages.extend(glob.glob(datapaths[idp] + '/' + c + '/' + names1) + glob.glob(
                    datapaths[idp] + '/' + c + '/' + names2) + glob.glob(datapaths[idp] + '/' + c + '/' + names3))

            # 				random_classImages = np.random.choice(classImages, int(len(classImages)*0.50))  ## if positive class has more image than negatives then random sampling of 50% doesnot work. Therefore, it is edited to the line below removing 0.5.
            # 				random_classImages = np.random.choice(classImages, int(len(classImages))) # removed to check stratified split

            #				print('50% data of selected class: {} ({})'.format(c, len(random_classImages)))
            # 				concatenated_list=np.concatenate([concatenated_list,random_classImages]) # removed to check stratified split
            concatenated_list = np.concatenate([concatenated_list, classImages])  # added to check stratified split

        # 				print('class: {} ({})'.format(c, len(classImages)))
        # 				print('Concatenated images ({})'.format( len(concatenated_list)))

        # added to check stratified split
        Only_dirName = []
        for i in range(len(concatenated_list)):
            Only_dirName.append(os.path.basename(Path(concatenated_list[i]).parents[1]))
        indices = np.arange(len(Only_dirName))
        y_train, y_test, y_train_idx, y_test_idx = train_test_split(Only_dirName, indices,
                                                                    test_size=len(dfB) / len(Only_dirName) * 1.5,
                                                                    random_state=42, stratify=Only_dirName)
        Negative_class = concatenated_list[y_test_idx]

        # removed to check stratified split
        # 			if len(concatenated_list)>len(dfB):
        # 				Negative_class = np.random.choice(concatenated_list, len(dfB))
        # 			else:
        # 				Negative_class = concatenated_list

        print('Total NEGATIVE class: {} ({})'.format(negative_class_name, len(Negative_class)))
        # Create an empty dataframe for this class
        dfAClass = pd.DataFrame(columns=['filename', 'classname', 'npimage'])
        for i, imageName in enumerate(Negative_class):
            npimageA, rescaled, filenameA = LoadImage(imageName, L, resize_images)
            dfAClass.loc[i] = [filenameA, negative_class_name, npimageA]
        dfA = pd.concat([dfA, dfAClass], axis=0)
        dfA.npimage = dfA.npimage / 255.0
        df = pd.concat([dfB, dfA], axis=0)  # Concatenate Selected class and all other class
    df = df.sample(frac=1).reset_index(drop=True)
    return df


def LoadImageList(im_names, L, resize_images, show=False):
    '''
    Function that loads a list of images given in im_names, and returns
    them in a numpy format that can be used by the classifier.
    '''
    npimages = np.ndarray((len(im_names), L, L, 3))

    for i, im_name in enumerate(im_names):
        npimage, rescaled, filename = LoadImage(im_name, L, resize_images, show)
        npimages[i] = npimage
    return npimages / 255.0


def LoadMixedData(test_features, L, resize_images, alsoImages, compute_extrafeat):
    # Read from tsv file, and create column with full path to the image
    dfFeat = pd.DataFrame()
    for idp in range(len(test_features)):
        dftemp = pd.read_csv(test_features[idp], sep='\t')
        pathname = str(Path(test_features[idp]).parents[0])
        dftemp['filename'] = [pathname + '/' + dftemp.url[ii] for ii in range(len(dftemp))]
        dfFeat = pd.concat([dfFeat, dftemp], axis=0, sort=True)

    testimages1 = dfFeat['filename']
    testimages = list(testimages1)
    print('There are {} images in total'.format(len(testimages)))
    print('There are {} feature files in total'.format(len(test_features)))

    df = pd.DataFrame()
    for index, row in dfFeat.iterrows():
        if alsoImages:
            npimage, rescaled, filename = LoadImage(row.filename, L, resize_images)

            dftemp = pd.DataFrame([[npimage, rescaled] + row.to_list()],
                                  columns=['npimage', 'rescaled'] + dfFeat.columns.to_list())
        else:  # alsoImages is False here
            dftemp = pd.DataFrame([row.to_list()], columns=dfFeat.columns.to_list())
        df = pd.concat([df, dftemp], axis=0, sort=True)

    df.npimage = df.npimage / 255.0
    df = df.sample(frac=1).reset_index(drop=True)

    if compute_extrafeat == 'yes':
        dfExtraFeat = compute_extrafeat_function(df)
        df = pd.concat([df, dfExtraFeat], axis=1)

    return df


class Cdata:

    def __init__(self, datapath, L=None, class_select=None, classifier=None, compute_extrafeat=None, resize_images=None,
                 balance_weight=None, kind='mixed', training_data=True):
        self.Xfeat = None
        self.Ximage = None
        self.filenames = None
        self.classes = None
        self.datapath = datapath
        if L is None and kind != 'feat':
            print('CData: image size needs to be set, unless kind is \'feat\'')
            raise ValueError
        self.L = L
        self.class_select = class_select
        self.classifier = classifier
        self.compute_extrafeat = compute_extrafeat
        self.resize_images = resize_images
        self.balance_weight = balance_weight
        self.kind = kind
        self.df = None
        self.y = None
        self.X = None
        self.Load(self.datapath, self.L, self.class_select, self.classifier, self.compute_extrafeat, self.resize_images,
                  self.balance_weight, self.kind, training_data=training_data)
        return

    def Load(self, datapaths, L, class_select, classifier, compute_extrafeat, resize_images, balance_weight,
             kind='mixed', training_data=True):
        """
        Loads dataset
        For the moment, only mixed data. Later, also pure images or pure features.
        """
        self.L = L
        self.datapath = datapaths
        self.class_select = class_select
        self.kind = kind
        self.classifier = classifier
        self.compute_extrafeat = compute_extrafeat
        self.resize_images = resize_images

        if kind == 'mixed':
            self.df = LoadMixed(datapaths, L, class_select, classifier, resize_images, alsoImages=True)
            if compute_extrafeat == 'yes':
                dfExtraFeat = compute_extrafeat_function(self.df)
                self.df = pd.concat([self.df, dfExtraFeat], axis=1)

        elif kind == 'feat':
            self.df = LoadMixed(datapaths, L, class_select, classifier, resize_images, alsoImages=False)
            if compute_extrafeat == 'yes':
                dfExtraFeat = compute_extrafeat_function(self.df)
                self.df = pd.concat([self.df, dfExtraFeat], axis=1)

        elif kind == 'image':
            self.df = LoadImages(datapaths, L, class_select, classifier, resize_images, training_data=training_data)
            if compute_extrafeat == 'yes':
                dfExtraFeat = compute_extrafeat_function(self.df)
                self.df = pd.concat([self.df, dfExtraFeat], axis=1)

        else:
            raise NotImplementedError('Only mixed, image or feat data-loading')

        # 		print(self.df['classname'].unique())
        self.classes = self.df['classname'].unique()
        print('The data path is : {}'.format(self.datapath))
        self.kind = kind  # Now the data kind is kind. In most cases, we had already kind=self.kind, but if the user
        # tested another kind, it must be changed
        self.Check()  # Some sanity checks on the dataset
        self.CreateXy()  # Creates X and y, i.e. features and labels
        return

    def Check(self):
        """ Basic checks on the dataset """

        # Number of different classes
        classes = self.classes
        classifier = self.classifier
        if classifier == 'multi':
            if len(classes) < 2:
                print('There are less than 2 classes ({})'.format(len(classes)))
                raise ValueError
        elif classifier == 'binary':
            if len(classes) > 2:
                print('There are more than 2 classes for binary classifier ({})'.format(len(classes)))
                raise ValueError

        # Columns potentially useful for classification
        ucols = self.df.drop(columns=['classname', 'url', 'filename', 'file_size', 'timestamp'],
                             errors='ignore').columns
        if len(ucols) < 1:
            print('Columns: {}'.format(self.df.columns))
            raise ValueError('The dataset has no useful columns.')

        # Check for NaNs
        if self.df.isnull().any().any():
            print('There are NaN values in the data.')
            print(self.df)
            raise ValueError

        # Check that the images have the expected size
        if 'npimage' in self.df.columns:
            if self.df.npimage[0].shape != (self.L, self.L, 3):
                print(
                    'Cdata Check(): Images were not reshaped correctly: {} instead of {}'.format(self.npimage[0].shape,
                                                                                                 (self.L, self.L, 3)))

        return

    def CreateXy(self):
        """
        Creates features and target
        - removing the evidently junk columns.
        - allowing to access images and features separately and confortably
        """

        self.y = self.df.classname
        self.filenames = self.df.filename
        self.X = self.df.drop(columns=['classname', 'url', 'filename', 'file_size', 'timestamp'], errors='ignore')
        # 		self.X = self.df.drop(columns=['classname','url','file_size','timestamp'], errors='ignore')

        self.Ximage = self.X.npimage if (self.kind != 'feat') else None
        self.Xfeat = self.X.drop(columns=['npimage'], errors='ignore') if (self.kind != 'image') else None

        return


class Cdata_with_y:

    def __init__(self, classpath, datapath, L=None, class_select=None, classifier=None, compute_extrafeat=None,
                 resize_images=None, balance_weight=None, kind='mixed', training_data=True):
        self.Xfeat = None
        self.Ximage = None
        self.filenames = None
        self.classes = None
        self.datapath = datapath
        self.classpath = classpath
        if L is None and kind != 'feat':
            print('CData: image size needs to be set, unless kind is \'feat\'')
            raise ValueError
        self.L = L
        self.class_select = class_select
        self.classifier = classifier
        self.compute_extrafeat = compute_extrafeat
        self.resize_images = resize_images
        self.balance_weight = balance_weight
        self.kind = kind
        self.df = None
        self.y = None
        self.X = None

        # self.Load(self.datapath, self.L, self.class_select, self.classifier, self.compute_extrafeat, self.resize_images,
        #           self.balance_weight, self.kind, training_data=training_data)
        #
        self.Load_with_y(self.classpath, self.datapath, self.L, self.class_select, self.classifier,
                         self.compute_extrafeat, self.resize_images,
                         self.balance_weight, self.kind, training_data=training_data)
        return

    def Load_with_y(self, classpaths, datapaths, L, class_select, classifier, compute_extrafeat, resize_images,
                    balance_weight, kind='mixed', training_data=True):
        """
        Loads dataset
        For the moment, only mixed data. Later, also pure images or pure features.
        """
        self.L = L
        self.datapath = datapaths
        self.class_select = class_select
        self.kind = kind
        self.classifier = classifier
        self.compute_extrafeat = compute_extrafeat
        self.resize_images = resize_images
        self.classpath = classpaths

        if kind == 'mixed':
            self.df = LoadMixed(datapaths, L, class_select, classifier, resize_images, alsoImages=True)
            if compute_extrafeat == 'yes':
                dfExtraFeat = compute_extrafeat_function(self.df)
                self.df = pd.concat([self.df, dfExtraFeat], axis=1)

        elif kind == 'feat':
            self.df = LoadMixed(datapaths, L, class_select, classifier, resize_images, alsoImages=False)
            if compute_extrafeat == 'yes':
                dfExtraFeat = compute_extrafeat_function(self.df)
                self.df = pd.concat([self.df, dfExtraFeat], axis=1)

        elif kind == 'image':
            self.df = LoadImages(datapaths, L, class_select, classifier, resize_images, training_data=training_data)
            if compute_extrafeat == 'yes':
                dfExtraFeat = compute_extrafeat_function(self.df)
                self.df = pd.concat([self.df, dfExtraFeat], axis=1)

        else:
            raise NotImplementedError('Only mixed, image or feat data-loading')

        print('The class path is : {}'.format(self.classpath))
        # self.classes = self.df['classname'].unique()
        self.classes = np.load(self.classpath + '/classes.npy')
        self.kind = kind  # Now the data kind is kind. In most cases, we had already kind=self.kind, but if the user
        # tested another kind, it must be changed
        self.Check_with_y()  # Some sanity checks on the dataset
        self.CreateXy_with_y()  # Creates X and y, i.e. features and labels
        return

    def Check_with_y(self):
        """ Basic checks on the dataset """

        # Number of different classes
        classes = self.classes
        classifier = self.classifier
        if classifier == 'multi':
            if len(classes) < 2:
                print('There are less than 2 classes ({})'.format(len(classes)))
                raise ValueError
        elif classifier == 'binary':
            if len(classes) > 2:
                print('There are more than 2 classes for binary classifier ({})'.format(len(classes)))
                raise ValueError

        # Columns potentially useful for classification
        ucols = self.df.drop(columns=['classname', 'url', 'filename', 'file_size', 'timestamp'],
                             errors='ignore').columns
        if len(ucols) < 1:
            print('Columns: {}'.format(self.df.columns))
            raise ValueError('The dataset has no useful columns.')

        # Check for NaNs
        if self.df.isnull().any().any():
            print('There are NaN values in the data.')
            print(self.df)
            raise ValueError

        # Check that the images have the expected size
        if 'npimage' in self.df.columns:
            if self.df.npimage[0].shape != (self.L, self.L, 3):
                print(
                    'Cdata Check(): Images were not reshaped correctly: {} instead of {}'.format(self.npimage[0].shape,
                                                                                                 (self.L, self.L, 3)))

        return

    def CreateXy_with_y(self):
        """
        Creates features and target
        - removing the evidently junk columns.
        - allowing to access images and features separately and confortably
        """

        self.y = self.df.classname
        self.filenames = self.df.filename
        self.X = self.df.drop(columns=['classname', 'url', 'filename', 'file_size', 'timestamp'], errors='ignore')
        # 		self.X = self.df.drop(columns=['classname','url','file_size','timestamp'], errors='ignore')

        self.Ximage = self.X.npimage if (self.kind != 'feat') else None
        self.Xfeat = self.X.drop(columns=['npimage'], errors='ignore') if (self.kind != 'image') else None

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


class CTrainTestSet:
    """
    A class for extracting train and test sets from the original dataset, and preprocessing them.
    """

    def __init__(self, X, y, filenames, ttkind='image', classifier=None, balance_weight=None, rescale=False,
                 testSplit=0.2, valid_set=None, test_set=None, compute_extrafeat=None, random_state=12345):
        """
        X and y are dataframes with features and labels
        """

        self.testX = None
        self.class_weights_tensor = None
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
        self.class_weights = None
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
        self.y = y
        self.VectorizeLabels()
        self.filenames = filenames

        # Now the features
        if ttkind == 'image' and compute_extrafeat == 'no':
            self.X = self.ImageNumpyFromMixedDataframe(X)
        elif ttkind == 'feat':
            X = DropCols(X, ['npimage', 'rescaled'])
            X = RemoveUselessCols(X)
            self.X = np.array([X.to_numpy()[i] for i in range(len(X.index))])
        else:
            # This checks if there are images, but it also implicitly checks if there are features. In fact,
            # if there are only images, X is a series and has no attribute columns (I am aware this should be coded
            # better).
            if 'npimage' not in X.columns:
                raise RuntimeError(
                    'Error: you asked for mixed Train-Test, but the dataset you gave me does not contain images.')
            self.X = RemoveUselessCols(X)  # Note that with ttkind=mixed, X stays a dataframe

        # Split train and test data
        self.Split(test_size=testSplit, valid_set=valid_set, test_set=test_set, random_state=random_state)

        # Rescale features
        if rescale is True:
            self.Rescale()
            self.rescale = True
        else:
            self.rescale = False

        return

    def VectorizeLabels(self):
        """
        Transform labels in one-hot encoded vectors
        This is where we will act if we decide to train with HYBRID LABELS
        """
        self.lb = LabelBinarizer()
        self.y = self.lb.fit_transform(self.y.tolist())
        if self.classifier == 'binary' or self.classifier == 'versusall':
            self.y = np.hstack((1 - self.y, self.y))

        return

    def UnvectorizeLabels(self, y):
        """ Recovers the original labels from the vectorized ones """

        return self.lb.inverse_transform(y) if self.classifier == 'multi' else self.lb.inverse_transform(y[:, 1])

    def ImageNumpyFromMixedDataframe(self, X=None):
        """ Returns a numpy array of the shape (nexamples, L, L, channels)"""
        if X is None:
            X = self.X

        # The column containing npimage
        im_col = [i for i, col in enumerate(X.columns) if col == 'npimage'][0]

        return np.array([X.to_numpy()[i, im_col] for i in range(len(X.index))])

    def Split(self, test_size=0.2, valid_set=None, test_set=None, random_state=12345):
        """
        Splits train and test datasets.
        Allows to put all the data in the test set by choosing test_size=1. This is useful for evaluation.
        Handles differently the mixed case, because in that case  X is a dataframe.
        """
        if test_set != 'no':
            if test_size < 1:
                if valid_set == 'no':
                    self.trainX, self.testX, self.trainY, self.testY, self.trainFilenames, self.testFilenames = \
                        train_test_split(self.X, self.y, self.filenames, test_size=test_size, random_state=random_state,
                                         shuffle=True, stratify=self.y)
                elif valid_set == 'yes':
                    train_ratio = 0.80
                    validation_ratio = 0.05
                    test_ratio = 0.15
                    self.trainX, test1X, self.trainY, test1Y, self.trainFilenames, test1Filenames = \
                        train_test_split(self.X, self.y, self.filenames, test_size=1 - train_ratio,
                                         random_state=random_state,
                                         shuffle=True, stratify=self.y)
                    self.valX, self.testX, self.valY, self.testY, self.valFilenames, self.testFilenames = \
                        train_test_split(test1X, test1Y, test1Filenames,
                                         test_size=test_ratio / (test_ratio + validation_ratio),
                                         random_state=random_state, shuffle=True, stratify=test1Y)

                y_integers = np.argmax(self.trainY, axis=1)
                if self.balance_weight == 'yes':
                    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_integers),
                                                         y=y_integers)
                else:
                    class_weights = compute_class_weight(class_weight=None, classes=np.unique(y_integers), y=y_integers)
                self.class_weights = dict(enumerate(class_weights))
                self.class_weights_tensor = torch.Tensor(class_weights)

            else:  # This allows us to pack everything into the test set
                self.trainX, self.testX, self.trainY, self.testY, self.trainFilenames, self.testFilenames = \
                    self.X, None, self.y, None, self.filenames, None

        elif test_set == 'no':
            self.trainX, self.trainY, self.trainFilenames = self.X, self.y, self.filenames
            y_integers = np.argmax(self.trainY, axis=1)

            if self.balance_weight == 'yes':
                class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_integers),
                                                     y=y_integers)
            else:
                class_weights = compute_class_weight(class_weight=None, classes=np.unique(y_integers), y=y_integers)
            self.class_weights = dict(enumerate(class_weights))
            self.class_weights_tensor = torch.Tensor(class_weights)

        if self.ttkind == 'mixed':
            # Images
            if test_set != 'no':
                if self.trainX is not None:
                    self.trainXimage = self.ImageNumpyFromMixedDataframe(self.trainX)
                self.testXimage = self.ImageNumpyFromMixedDataframe(self.testX)
                if valid_set == 'yes':
                    self.valXimage = self.ImageNumpyFromMixedDataframe(self.valX)
            else:
                self.trainXimage = self.ImageNumpyFromMixedDataframe(self.trainX)

            # Features
            if test_set != 'no':
                if self.trainX is not None:
                    Xf = DropCols(self.trainX, ['npimage', 'rescaled'])
                    self.trainXfeat = np.array([Xf.to_numpy()[i] for i in range(len(Xf.index))])
                Xf = DropCols(self.testX, ['npimage', 'rescaled'])
                self.testXfeat = np.array([Xf.to_numpy()[i] for i in range(len(Xf.index))])
                if valid_set == 'yes':
                    Xf = DropCols(self.valX, ['npimage', 'rescaled'])
                    self.valXfeat = np.array([Xf.to_numpy()[i] for i in range(len(Xf.index))])
            else:
                Xf = DropCols(self.trainX, ['npimage', 'rescaled'])
                self.trainXfeat = np.array([Xf.to_numpy()[i] for i in range(len(Xf.index))])

        elif self.ttkind == 'image' and self.compute_extrafeat == 'yes':
            # Images
            if test_set != 'no':
                if self.trainX is not None:
                    self.trainXimage = self.ImageNumpyFromMixedDataframe(self.trainX)
                self.testXimage = self.ImageNumpyFromMixedDataframe(self.testX)
                if valid_set == 'yes':
                    self.valXimage = self.ImageNumpyFromMixedDataframe(self.valX)
            else:
                self.trainXimage = self.ImageNumpyFromMixedDataframe(self.trainX)

            # Features
            if test_set != 'no':
                if self.trainX is not None:
                    Xf = DropCols(self.trainX, ['npimage', 'rescaled'])
                    self.trainXfeat = np.array([Xf.to_numpy()[i] for i in range(len(Xf.index))])
                Xf = DropCols(self.testX, ['npimage', 'rescaled'])
                self.testXfeat = np.array([Xf.to_numpy()[i] for i in range(len(Xf.index))])
                if valid_set == 'yes':
                    Xf = DropCols(self.valX, ['npimage', 'rescaled'])
                    self.valXfeat = np.array([Xf.to_numpy()[i] for i in range(len(Xf.index))])
            else:
                Xf = DropCols(self.trainX, ['npimage', 'rescaled'])
                self.trainXfeat = np.array([Xf.to_numpy()[i] for i in range(len(Xf.index))])
        return

    def Rescale(self):
        if self.ttkind == 'mixed':
            self.RescaleMixed()
        elif self.ttkind == 'feat':
            self.RescaleFeat()
        elif self.ttkind == 'image' and self.compute_extrafeat == 'yes':
            self.RescaleMixed()
        elif self.ttkind == 'image' and self.compute_extrafeat == 'no':
            pass  # We don't rescale the image
        else:
            raise NotImplementedError('CTrainTestSet: ttkind must be feat, image or mixed')
        return

    def RescaleMixed(self):
        """
        Rescales all columns except npimage to have mean zero and unit standard deviation

        To avoid data leakage, the rescaling factors are chosen from the training set
        """

        if self.trainX is None:
            print(
                'No rescaling is performed because the training set is empty, but the truth is that '
                'in this case we should have rescaling parameters coming from elsewhere')
            return

        cols = self.trainX.columns.tolist()

        if 'npimage' in cols:
            cols.remove('npimage')

        # Set to zero mean and unit standard deviation
        x = self.trainX[cols].to_numpy()
        mu = x.mean(axis=0)
        sigma = np.std(x, axis=0, ddof=0)

        # Training set
        self.trainX[cols] -= mu  # Set mean to zero
        self.trainX[cols] /= sigma  # Set standard dev to one
        # Test set
        self.testX[cols] -= mu  # Set mean to zero
        self.testX[cols] /= sigma  # Set standard dev to one

        # These checks are only valid for the training set
        assert (np.all(np.isclose(self.trainX[cols].mean(), 0, atol=1e-5)))  # Check that mean is zero
        assert (
            np.all(np.isclose(np.std(self.trainX[cols], axis=0, ddof=0), 1, atol=1e-5)))  # Check that std dev is unity

        return

    def RescaleFeat(self):
        """
        Rescales all columns

        To avoid data leakage, the rescaling factors are chosen from the training set
        """

        # Set to zero mean and unit standard deviation
        mu = self.trainX.mean(axis=0)
        sigma = np.std(self.trainX, axis=0, ddof=0)

        # Training set
        self.trainX -= mu  # Set mean to zero
        self.trainX /= sigma  # Set standard dev to one
        # Test set
        self.testX -= mu  # Set mean to zero
        self.testX /= sigma  # Set standard dev to one

        # These checks are only valid for the training set
        assert (np.all(np.isclose(self.trainX.mean(), 0, atol=1e-5)))  # Check that mean is zero
        assert (np.all(np.isclose(np.std(self.trainX, axis=0, ddof=0), 1, atol=1e-5)))  # Check that std dev is unity

        return

    def SelectCols(self, X, cols):
        """
        Keeps only the columns cols from the dataframe X.
        cols is a list with the columns names
        """

        if isinstance(X, pd.DataFrame):  # Make sure it is not a series
            if set(cols).issubset(set(X.columns)):  # Check that columns we want to select exist
                return X[cols]
            else:
                print('self.X.columns: {}'.format(self.X.columns))
                print('requested cols: {}'.format(cols))
                raise IndexError('You are trying to select columns that are not present in the dataframe')
        else:
            assert (len(cols) == 1)  # If it's a series there should be only one column
            assert (self.X.name == cols[0])  # And that column should coincide with the series name
            return

    def MergeLabels(self):
        """ Merges labels to create aggregated classes """
        raise NotImplementedError


class CTestSet_with_y:
    """
    A class for extracting train and test sets from the original dataset, and preprocessing them.
    """

    def __init__(self, X, y, filenames, ttkind='image', classifier=None, balance_weight=None, rescale=False,
                 testSplit=0.2, valid_set=None, test_set=None, compute_extrafeat=None, random_state=12345):
        """
        X and y are dataframes with features and labels
        """

        self.testX = None
        self.class_weights_tensor = None
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
        self.class_weights = None
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
        self.y = y
        self.filenames = filenames

        # Now the features
        if ttkind == 'image' and compute_extrafeat == 'no':
            self.X = self.ImageNumpyFromMixedDataframe(X)
        elif ttkind == 'feat':
            X = DropCols(X, ['npimage', 'rescaled'])
            X = RemoveUselessCols(X)
            self.X = np.array([X.to_numpy()[i] for i in range(len(X.index))])
        else:
            # This checks if there are images, but it also implicitly checks if there are features. In fact,
            # if there are only images, X is a series and has no attribute columns (I am aware this should be coded
            # better).
            if 'npimage' not in X.columns:
                raise RuntimeError(
                    'Error: you asked for mixed Train-Test, but the dataset you gave me does not contain images.')
            self.X = RemoveUselessCols(X)  # Note that with ttkind=mixed, X stays a dataframe

        # Split train and test data
        self.Split(test_size=testSplit, valid_set=valid_set, test_set=test_set, random_state=random_state)

        # Rescale features
        if rescale is True:
            self.Rescale()
            self.rescale = True
        else:
            self.rescale = False

        return

    def ImageNumpyFromMixedDataframe(self, X=None):
        """ Returns a numpy array of the shape (nexamples, L, L, channels)"""
        if X is None:
            X = self.X

        # The column containing npimage
        im_col = [i for i, col in enumerate(X.columns) if col == 'npimage'][0]

        return np.array([X.to_numpy()[i, im_col] for i in range(len(X.index))])

    def Split(self, test_size=0.2, valid_set=None, test_set=None, random_state=12345):
        """
        Splits train and test datasets.
        Allows to put all the data in the test set by choosing test_size=1. This is useful for evaluation.
        Handles differently the mixed case, because in that case  X is a dataframe.
        """
        if test_set != 'no':
            if test_size < 1:
                if valid_set == 'no':
                    self.trainX, self.testX, self.trainY, self.testY, self.trainFilenames, self.testFilenames = \
                        train_test_split(self.X, self.y, self.filenames, test_size=test_size, random_state=random_state,
                                         shuffle=True, stratify=self.y)
                elif valid_set == 'yes':
                    train_ratio = 0.80
                    validation_ratio = 0.05
                    test_ratio = 0.15
                    self.trainX, test1X, self.trainY, test1Y, self.trainFilenames, test1Filenames = \
                        train_test_split(self.X, self.y, self.filenames, test_size=1 - train_ratio,
                                         random_state=random_state,
                                         shuffle=True, stratify=self.y)
                    self.valX, self.testX, self.valY, self.testY, self.valFilenames, self.testFilenames = \
                        train_test_split(test1X, test1Y, test1Filenames,
                                         test_size=test_ratio / (test_ratio + validation_ratio),
                                         random_state=random_state, shuffle=True, stratify=test1Y)

                y_integers = np.argmax(self.trainY, axis=1)
                if self.balance_weight == 'yes':
                    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_integers),
                                                         y=y_integers)
                else:
                    class_weights = compute_class_weight(class_weight=None, classes=np.unique(y_integers), y=y_integers)
                self.class_weights = dict(enumerate(class_weights))
                self.class_weights_tensor = torch.Tensor(class_weights)

            else:  # This allows us to pack everything into the test set
                self.trainX, self.testX, self.trainY, self.testY, self.trainFilenames, self.testFilenames = \
                    self.X, None, self.y, None, self.filenames, None

        elif test_set == 'no':
            self.trainX, self.trainY, self.trainFilenames = self.X, self.y, self.filenames

        if self.ttkind == 'mixed':
            # Images
            if test_set != 'no':
                if self.trainX is not None:
                    self.trainXimage = self.ImageNumpyFromMixedDataframe(self.trainX)
                self.testXimage = self.ImageNumpyFromMixedDataframe(self.testX)
                if valid_set == 'yes':
                    self.valXimage = self.ImageNumpyFromMixedDataframe(self.valX)
            else:
                self.trainXimage = self.ImageNumpyFromMixedDataframe(self.trainX)

            # Features
            if test_set != 'no':
                if self.trainX is not None:
                    Xf = DropCols(self.trainX, ['npimage', 'rescaled'])
                    self.trainXfeat = np.array([Xf.to_numpy()[i] for i in range(len(Xf.index))])
                Xf = DropCols(self.testX, ['npimage', 'rescaled'])
                self.testXfeat = np.array([Xf.to_numpy()[i] for i in range(len(Xf.index))])
                if valid_set == 'yes':
                    Xf = DropCols(self.valX, ['npimage', 'rescaled'])
                    self.valXfeat = np.array([Xf.to_numpy()[i] for i in range(len(Xf.index))])
            else:
                Xf = DropCols(self.trainX, ['npimage', 'rescaled'])
                self.trainXfeat = np.array([Xf.to_numpy()[i] for i in range(len(Xf.index))])

        elif self.ttkind == 'image' and self.compute_extrafeat == 'yes':
            # Images
            if test_set != 'no':
                if self.trainX is not None:
                    self.trainXimage = self.ImageNumpyFromMixedDataframe(self.trainX)
                self.testXimage = self.ImageNumpyFromMixedDataframe(self.testX)
                if valid_set == 'yes':
                    self.valXimage = self.ImageNumpyFromMixedDataframe(self.valX)
            else:
                self.trainXimage = self.ImageNumpyFromMixedDataframe(self.trainX)

            # Features
            if test_set != 'no':
                if self.trainX is not None:
                    Xf = DropCols(self.trainX, ['npimage', 'rescaled'])
                    self.trainXfeat = np.array([Xf.to_numpy()[i] for i in range(len(Xf.index))])
                Xf = DropCols(self.testX, ['npimage', 'rescaled'])
                self.testXfeat = np.array([Xf.to_numpy()[i] for i in range(len(Xf.index))])
                if valid_set == 'yes':
                    Xf = DropCols(self.valX, ['npimage', 'rescaled'])
                    self.valXfeat = np.array([Xf.to_numpy()[i] for i in range(len(Xf.index))])
            else:
                Xf = DropCols(self.trainX, ['npimage', 'rescaled'])
                self.trainXfeat = np.array([Xf.to_numpy()[i] for i in range(len(Xf.index))])
        return

    def Rescale(self):
        if self.ttkind == 'mixed':
            self.RescaleMixed()
        elif self.ttkind == 'feat':
            self.RescaleFeat()
        elif self.ttkind == 'image' and self.compute_extrafeat == 'yes':
            self.RescaleMixed()
        elif self.ttkind == 'image' and self.compute_extrafeat == 'no':
            pass  # We don't rescale the image
        else:
            raise NotImplementedError('CTrainTestSet: ttkind must be feat, image or mixed')
        return

    def RescaleMixed(self):
        """
        Rescales all columns except npimage to have mean zero and unit standard deviation

        To avoid data leakage, the rescaling factors are chosen from the training set
        """

        if self.trainX is None:
            print(
                'No rescaling is performed because the training set is empty, but the truth is that '
                'in this case we should have rescaling parameters coming from elsewhere')
            return

        cols = self.trainX.columns.tolist()

        if 'npimage' in cols:
            cols.remove('npimage')

        # Set to zero mean and unit standard deviation
        x = self.trainX[cols].to_numpy()
        mu = x.mean(axis=0)
        sigma = np.std(x, axis=0, ddof=0)

        # Training set
        self.trainX[cols] -= mu  # Set mean to zero
        self.trainX[cols] /= sigma  # Set standard dev to one
        # Test set
        self.testX[cols] -= mu  # Set mean to zero
        self.testX[cols] /= sigma  # Set standard dev to one

        # These checks are only valid for the training set
        assert (np.all(np.isclose(self.trainX[cols].mean(), 0, atol=1e-5)))  # Check that mean is zero
        assert (
            np.all(np.isclose(np.std(self.trainX[cols], axis=0, ddof=0), 1, atol=1e-5)))  # Check that std dev is unity

        return

    def RescaleFeat(self):
        """
        Rescales all columns

        To avoid data leakage, the rescaling factors are chosen from the training set
        """

        # Set to zero mean and unit standard deviation
        mu = self.trainX.mean(axis=0)
        sigma = np.std(self.trainX, axis=0, ddof=0)

        # Training set
        self.trainX -= mu  # Set mean to zero
        self.trainX /= sigma  # Set standard dev to one
        # Test set
        self.testX -= mu  # Set mean to zero
        self.testX /= sigma  # Set standard dev to one

        # These checks are only valid for the training set
        assert (np.all(np.isclose(self.trainX.mean(), 0, atol=1e-5)))  # Check that mean is zero
        assert (np.all(np.isclose(np.std(self.trainX, axis=0, ddof=0), 1, atol=1e-5)))  # Check that std dev is unity

        return

    def SelectCols(self, X, cols):
        """
        Keeps only the columns cols from the dataframe X.
        cols is a list with the columns names
        """

        if isinstance(X, pd.DataFrame):  # Make sure it is not a series
            if set(cols).issubset(set(X.columns)):  # Check that columns we want to select exist
                return X[cols]
            else:
                print('self.X.columns: {}'.format(self.X.columns))
                print('requested cols: {}'.format(cols))
                raise IndexError('You are trying to select columns that are not present in the dataframe')
        else:
            assert (len(cols) == 1)  # If it's a series there should be only one column
            assert (self.X.name == cols[0])  # And that column should coincide with the series name
            return

    def MergeLabels(self):
        """ Merges labels to create aggregated classes """
        raise NotImplementedError


if __name__ == '__main__':
    pass

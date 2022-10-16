import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2



def PlotFeatureDistribution(datapaths, outpath, selected_features, n_bins, datapath_labels):

    print('-----------------Now plotting feature distribution for each class and each selected feature.-----------------')

    n_data = len(datapaths) # number of datapaths

    list_class_rep = ['aphanizomenon', 'asplanchna', 'asterionella', 'bosmina', 'brachionus', 'ceratium',
                     'chaoborus', 'collotheca', 'conochilus', 'copepod_skins', 'cyclops', 'daphnia', 'daphnia_skins', 
                     'diaphanosoma', 'diatom_chain', 'dinobryon', 'dirt', 'eudiaptomus', 'filament', 
                     'fish', 'fragilaria', 'hydra', 'kellicottia', 'keratella_cochlearis', 'keratella_quadrata', 
                     'leptodora', 'maybe_cyano', 'nauplius', 'paradileptus', 'polyarthra', 'rotifers', 
                     'synchaeta', 'trichocerca', 'unknown', 'unknown_plankton', 'uroglena']
    
    # find the repetitive classes in selected datasets
    for idatapath in datapaths:
        list_class = os.listdir(idatapath)
        list_class_rep = list(set(list_class) & set(list_class_rep))
        list.sort(list_class_rep)
    # print('Repetitive classes of two datasets: {}'.format(list_class_rep))
    
    # plot feature distribution

    all_features = [([0] * len(datapaths)) for i in range(len(list_class_rep))]
    for i, iclass in enumerate(list_class_rep):
        
        for j, idatapath in enumerate(datapaths):
            class_datapath = idatapath + iclass + '/' # directory of each class with classname
            df_all_feat = ConcatAllFeatures(class_datapath)
            all_features[i][j] = df_all_feat

        for ifeature in selected_features:
            ax = plt.subplot(1, 1, 1)
            ax.set_xlabel(ifeature + ' (normalized)')
            ax.set_ylabel('Density')

            min_feature = []
            max_feature = []
            features = []

            for j, idatapath in enumerate(datapaths):
                feature = all_features[i][j][ifeature]
                min_feature.append(min(feature))
                max_feature.append(max(feature))
                features.append(feature)

            min_bin = np.min(min_feature) # find global minimum value of feature in all datasets
            max_bin = np.max(max_feature) # find global maximum value of feature in all datasets

            normalized_feature = np.divide((np.array(features, dtype=object) - min_bin), (max_bin - min_bin)) # normalization of feature values

            histogram = plt.hist(normalized_feature, histtype='stepfilled', bins=n_bins, range=(0, 1), density=True, alpha=0.5, label=datapath_labels)
            density_1 = histogram[0][0]
            density_2 = histogram[0][1]

            HD = HellingerDistance(density_1, density_2) # compute the Hellinger distance of feature between 2 datasets
            
            plt.title('Hellinger distance = %.3f' % HD)
            plt.legend()
            plt.tight_layout()
                
            outpath_feature = outpath + ifeature + '/'
            try:
                os.mkdir(outpath_feature)
            except FileExistsError:
                pass
            plt.savefig(outpath_feature + ifeature + '_' + iclass + '.png')
            plt.close()
            ax.clear()



# def PlotFeatureDistribution(datapaths, outpath, selected_features, n_bins, datapath_labels):

#     print('-----------------Now plotting feature distribution for each class and each selected feature.-----------------')

#     n_data = len(datapaths) # number of datapaths

#     list_class_rep = ['aphanizomenon', 'asplanchna', 'asterionella', 'bosmina', 'brachionus', 'ceratium',
#                      'chaoborus', 'collotheca', 'conochilus', 'copepod_skins', 'cyclops', 'daphnia', 'daphnia_skins', 
#                      'diaphanosoma', 'diatom_chain', 'dinobryon', 'dirt', 'eudiaptomus', 'filament', 
#                      'fish', 'fragilaria', 'hydra', 'kellicottia', 'keratella_cochlearis', 'keratella_quadrata', 
#                      'leptodora', 'maybe_cyano', 'nauplius', 'paradileptus', 'polyarthra', 'rotifers', 
#                      'synchaeta', 'trichocerca', 'unknown', 'unknown_plankton', 'uroglena']
    
#     # find the repetitive classes in selected datasets
#     for idatapath in datapaths:
#         list_class = os.listdir(idatapath)
#         list_class_rep = list(set(list_class) & set(list_class_rep))
#         list.sort(list_class_rep)
#     # print('Repetitive classes of two datasets: {}'.format(list_class_rep))
    
#     # plot feature distribution
#     for iclass in list_class_rep:
#         for ifeature in selected_features:
#             ax = plt.subplot(1, 1, 1)
#             ax.set_xlabel(ifeature + ' (normalized)')
#             ax.set_ylabel('Density')

#             min_feature = []
#             max_feature = []
#             feature = []

#             for idatapath in datapaths:
#                 class_datapath = idatapath + iclass + '/' # directory of each class with classname
#                 df_all_feat = ConcatAllFeatures(class_datapath)

#                 min_feature.append(min(df_all_feat[ifeature]))
#                 max_feature.append(max(df_all_feat[ifeature]))
#                 feature.append(df_all_feat[ifeature])

#             min_bin = np.min(min_feature) # find global minimum value of feature in all datasets
#             max_bin = np.max(max_feature) # find global maximum value of feature in all datasets

#             normalized_feature = np.divide((np.array(feature, dtype=object) - min_bin), (max_bin - min_bin)) # normalization of feature values

#             histogram = plt.hist(normalized_feature, histtype='stepfilled', bins=n_bins, range=(0, 1), density=True, alpha=0.5, label=datapath_labels)
#             density_1 = histogram[0][0]
#             density_2 = histogram[0][1]

#             HD = HellingerDistance(density_1, density_2) # compute the Hellinger distance of feature between 2 datasets
            
#             plt.title('Hellinger distance = %.3f' % HD)
#             plt.legend()
#             plt.tight_layout()
                
#             outpath_feature = outpath + ifeature + '/'
#             try:
#                 os.mkdir(outpath_feature)
#             except FileExistsError:
#                 pass
#             plt.savefig(outpath_feature + ifeature + '_' + iclass + '.png')
#             plt.close()
#             ax.clear()



def PlotFeatureHDversusBin(datapaths, outpath, selected_features):

    print('-----------------Now plotting Hellinger distances v.s. numbers of bin.-----------------')

    list_class_rep = ['aphanizomenon', 'asplanchna', 'asterionella', 'bosmina', 'brachionus', 'ceratium',
                     'chaoborus', 'collotheca', 'conochilus', 'copepod_skins', 'cyclops', 'daphnia', 'daphnia_skins', 
                     'diaphanosoma', 'diatom_chain', 'dinobryon', 'dirt', 'eudiaptomus', 'filament', 
                     'fish', 'fragilaria', 'hydra', 'kellicottia', 'keratella_cochlearis', 'keratella_quadrata', 
                     'leptodora', 'maybe_cyano', 'nauplius', 'paradileptus', 'polyarthra', 'rotifers', 
                     'synchaeta', 'trichocerca', 'unknown', 'unknown_plankton', 'uroglena']

    # find the repetitive classes in selected datasets
    for idatapath in datapaths:
        list_class = os.listdir(idatapath)
        list_class_rep = list(set(list_class) & set(list_class_rep))
        list.sort(list_class_rep)
    # print('Repetitive classes of two datasets: {}'.format(list_class_rep))

    list_n_bins = [5, 10, 20, 50, 100, 125, 150, 175, 200, 300, 400, 500]
    for ifeature in selected_features:
        ax = plt.subplot(1, 1, 1)
        plt.figure(figsize=(10, 10))
        
        for iclass in list_class_rep:
            list_HD = []
            for in_bins in list_n_bins:
                min_feature = []
                max_feature = []
                feature = []

                for idatapath in datapaths:
                    class_datapath = idatapath + iclass + '/' # directory of each class with classname
                    df_all_feat = ConcatAllFeatures(class_datapath)

                    min_feature.append(min(df_all_feat[ifeature]))
                    max_feature.append(max(df_all_feat[ifeature]))
                    feature.append(df_all_feat[ifeature])

                min_bin = min(min_feature) # find global minimum value of feature in all datasets
                max_bin = max(max_feature) # find global maximum value of feature in all datasets

                normalized_feature = np.divide((np.array(feature, dtype=object) - min_bin), (max_bin - min_bin)) # normalization of feature values

                histogram_1 = np.histogram(normalized_feature[0], bins=in_bins, range=(0, 1), density=True)
                histogram_2 = np.histogram(normalized_feature[1], bins=in_bins, range=(0, 1), density=True)
                density_1 = histogram_1[0]
                density_2 = histogram_2[0]

                HD = HellingerDistance(density_1, density_2) # compute the Hellinger distance of feature between 2 datasets
                list_HD.append(HD)
            
            plt.plot(list_n_bins, list_HD, label=iclass)
            
        ax.set_xlabel('Number of bins')
        ax.set_ylabel('Hellinger Distance')
        ax.set_title(ifeature)
        plt.legend(bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
        plt.tight_layout()
        
        outpath_feature = outpath + ifeature + '/'
        try:
            os.mkdir(outpath_feature)
        except FileExistsError:
            pass
        plt.savefig(outpath_feature + ifeature + '_HD.png' )
        plt.close()
        ax.clear()



def GlobalHD_feature(datapaths, outpath, n_bins):

    print('-----------------Now computing global Hellinger distances on feature.-----------------')

    list_class_rep = ['aphanizomenon', 'asplanchna', 'asterionella', 'bosmina', 'brachionus', 'ceratium',
                     'chaoborus', 'collotheca', 'conochilus', 'copepod_skins', 'cyclops', 'daphnia', 'daphnia_skins', 
                     'diaphanosoma', 'diatom_chain', 'dinobryon', 'dirt', 'eudiaptomus', 'filament', 
                     'fish', 'fragilaria', 'hydra', 'kellicottia', 'keratella_cochlearis', 'keratella_quadrata', 
                     'leptodora', 'maybe_cyano', 'nauplius', 'paradileptus', 'polyarthra', 'rotifers', 
                     'synchaeta', 'trichocerca', 'unknown', 'unknown_plankton', 'uroglena']

    # find the repetitive classes in selected datasets
    for idatapath in datapaths:
        list_class = os.listdir(idatapath)
        list_class_rep = list(set(list_class) & set(list_class_rep))
        list.sort(list_class_rep)
    # print('Repetitive classes of two datasets: {}'.format(list_class_rep))

    list_features = ['width', 'height', 'w_rot', 'h_rot', 'angle_rot', 'aspect_ratio_2',
                    'rect_area', 'contour_area', 'contour_perimeter', 'extent',
                    'compactness', 'formfactor', 'hull_area', 'solidity_2', 'hull_perimeter',
                    'ESD', 'Major_Axis', 'Minor_Axis', 'Angle', 'Eccentricity1', 'Eccentricity2',
                    'Convexity', 'Roundness', 'm00', 'm10', 'm01', 'm20', 'm11', 'm02', 'm30', 'm21', 'm12', 'm03',
                    'mu20', 'mu11', 'mu02', 'mu30', 'mu21', 'mu12', 'mu03', 'nu20', 'nu11',
                    'nu02', 'nu30', 'nu21', 'nu12', 'nu03']

    list_global_HD = []
    all_features = [([0] * len(datapaths)) for i in range(len(list_class_rep))]

    for i, iclass in enumerate(list_class_rep):
        list_HD = []

        for j, idatapath in enumerate(datapaths):
            class_datapath = idatapath + iclass + '/'
            df_all_feat = ConcatAllFeatures(class_datapath)
            all_features[i][j] = df_all_feat

        for ifeature in list_features:
            min_feature = []
            max_feature = []
            features = []

            for j, idatapath in enumerate(datapaths):
                feature = all_features[i][j][ifeature]
                min_feature.append(min(feature))
                max_feature.append(max(feature))
                features.append(feature)

            min_bin = min(min_feature) # find global minimum value of feature in all datasets
            max_bin = max(max_feature) # find global maximum value of feature in all datasets

            normalized_feature = np.divide((np.array(features, dtype=object) - min_bin), (max_bin - min_bin)) # normalization of feature values

            histogram_1 = np.histogram(normalized_feature[0], bins=n_bins, range=(0, 1), density=True)
            histogram_2 = np.histogram(normalized_feature[1], bins=n_bins, range=(0, 1), density=True)
            density_1 = histogram_1[0]
            density_2 = histogram_2[0]

            HD = HellingerDistance(density_1, density_2) # compute the Hellinger distance of feature between 2 datasets
            list_HD.append(HD)

        global_HD_each_class = np.average(list_HD)
        list_global_HD.append(global_HD_each_class)
        
        with open(outpath + 'Global_HD_feature.txt', 'a') as f:
            # f.write('{}: {}\n'.format(iclass, global_HD_each_class))
            f.write('%-20s%-20f\n' % (iclass, global_HD_each_class))

    global_HD = np.average(list_global_HD)
    with open(outpath + 'Global_HD_feature.txt', 'a') as f:
        f.write(f'\n Global Hellinger Distance: {global_HD}')



# def GlobalHD_feature(datapaths, outpath, n_bins):

#     print('-----------------Now computing global Hellinger distances on feature.-----------------')

#     list_class_rep = ['aphanizomenon', 'asplanchna', 'asterionella', 'bosmina', 'brachionus', 'ceratium',
#                      'chaoborus', 'collotheca', 'conochilus', 'copepod_skins', 'cyclops', 'daphnia', 'daphnia_skins', 
#                      'diaphanosoma', 'diatom_chain', 'dinobryon', 'dirt', 'eudiaptomus', 'filament', 
#                      'fish', 'fragilaria', 'hydra', 'kellicottia', 'keratella_cochlearis', 'keratella_quadrata', 
#                      'leptodora', 'maybe_cyano', 'nauplius', 'paradileptus', 'polyarthra', 'rotifers', 
#                      'synchaeta', 'trichocerca', 'unknown', 'unknown_plankton', 'uroglena']

#     # find the repetitive classes in selected datasets
#     for idatapath in datapaths:
#         list_class = os.listdir(idatapath)
#         list_class_rep = list(set(list_class) & set(list_class_rep))
#         list.sort(list_class_rep)
#     # print('Repetitive classes of two datasets: {}'.format(list_class_rep))

#     list_features = ['width', 'height', 'w_rot', 'h_rot', 'angle_rot', 'aspect_ratio_2',
#                     'rect_area', 'contour_area', 'contour_perimeter', 'extent',
#                     'compactness', 'formfactor', 'hull_area', 'solidity_2', 'hull_perimeter',
#                     'ESD', 'Major_Axis', 'Minor_Axis', 'Angle', 'Eccentricity1', 'Eccentricity2',
#                     'Convexity', 'Roundness', 'm00', 'm10', 'm01', 'm20', 'm11', 'm02', 'm30', 'm21', 'm12', 'm03',
#                     'mu20', 'mu11', 'mu02', 'mu30', 'mu21', 'mu12', 'mu03', 'nu20', 'nu11',
#                     'nu02', 'nu30', 'nu21', 'nu12', 'nu03']

#     list_global_HD = []
#     for iclass in list_class_rep:
#         list_HD = []

#         for ifeature in list_features:
#             min_feature = []
#             max_feature = []
#             feature = []

#             for idatapath in datapaths:
#                 class_datapath = idatapath + iclass + '/' # directory of each class with classname
#                 df_all_feat = ConcatAllFeatures(class_datapath)

#                 min_feature.append(min(df_all_feat[ifeature]))
#                 max_feature.append(max(df_all_feat[ifeature]))
#                 feature.append(df_all_feat[ifeature])

#             min_bin = min(min_feature) # find global minimum value of feature in all datasets
#             max_bin = max(max_feature) # find global maximum value of feature in all datasets

#             normalized_feature = np.divide((np.array(feature, dtype=object) - min_bin), (max_bin - min_bin)) # normalization of feature values

#             histogram_1 = np.histogram(normalized_feature[0], bins=n_bins, range=(0, 1), density=True)
#             histogram_2 = np.histogram(normalized_feature[1], bins=n_bins, range=(0, 1), density=True)
#             density_1 = histogram_1[0]
#             density_2 = histogram_2[0]

#             HD = HellingerDistance(density_1, density_2) # compute the Hellinger distance of feature between 2 datasets
#             list_HD.append(HD)

#         global_HD_each_class = np.average(list_HD)
#         list_global_HD.append(global_HD_each_class)
        
#         with open(outpath + 'Global_HD_feature.txt', 'a') as f:
#             f.write('{}: {}\n'.format(iclass, global_HD_each_class))

#     global_HD = np.average(list_global_HD)
#     with open(outpath + 'Global_HD_feature.txt', 'a') as f:
#         f.write(f'\n Global Hellinger Distance: {global_HD}')



def LoadExtraFeatures(class_image_datapath):
    df_extra_feat = pd.DataFrame()
    dfFeatExtra1 = pd.DataFrame(columns=['width', 'height', 'w_rot', 'h_rot', 'angle_rot', 'aspect_ratio_2',
                                            'rect_area', 'contour_area', 'contour_perimeter', 'extent',
                                            'compactness', 'formfactor', 'hull_area', 'solidity_2', 'hull_perimeter',
                                            'ESD', 'Major_Axis', 'Minor_Axis', 'Angle', 'Eccentricity1', 'Eccentricity2',
                                            'Convexity', 'Roundness'])
    dfFeatExtra2 = pd.DataFrame(columns=['m00', 'm10', 'm01', 'm20', 'm11', 'm02', 'm30', 'm21', 'm12', 'm03',
                                            'mu20', 'mu11', 'mu02', 'mu30', 'mu21', 'mu12', 'mu03', 'nu20', 'nu11',
                                            'nu02', 'nu30', 'nu21', 'nu12', 'nu03'])

    list_image = os.listdir(class_image_datapath)
    for img in list_image:
        if img == 'Thumbs.db':
            continue
        
        image = cv2.imread(class_image_datapath + img)
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

        dfFeatExtra1.loc[img] = [width, height, w_rot, h_rot, angle_rot, aspect_ratio, rect_area, contour_area,
                                contour_perimeter, extent, compactness, formfactor, hull_area, solidity,
                                hull_perimeter, ESD, Major_Axis, Minor_Axis, angle, Eccentricity1,
                                Eccentricity2, Convexity, Roundness]
        dfFeatExtra2.loc[img] = M

    df_extra_feat = pd.concat([dfFeatExtra1, dfFeatExtra2], axis=1) # dataframe of extra features
    df_extra_feat = df_extra_feat.sort_index() # sort the images by index (filename)

    return df_extra_feat



def ConcatAllFeatures(class_datapath):
    if os.path.exists(class_datapath + 'training_data/'):
        class_image_datapath = class_datapath + 'training_data/' # folder with images inside
        df_feat = pd.read_csv(class_datapath + 'features.tsv', sep='\t') # dataframe of original features 

        # sort the dataframe of original features by image name
        for i in range(df_feat.shape[0]):
            df_feat.loc[i, 'url'] = df_feat.loc[i, 'url'][13:]
        df_feat = df_feat.sort_values(by='url')
        df_feat = df_feat.reset_index(drop=True)
        # df_feat.to_csv(class_datapath + 'features_sorted.tsv') # save sorted original features
        
        # load extra features from image
        df_extra_feat = LoadExtraFeatures(class_image_datapath)
        df_extra_feat = df_extra_feat.reset_index(drop=True)
        # df_extra_feat.to_csv(class_datapath + 'extra_features.tsv') # save extra features

        # original_features = df_feat.columns.to_list()
        # extra_features = df_extra_feat.columns.to_list()
        # all_features = original_features + extra_features
        # df_all_feat = pd.DataFrame(columns=all_features)
        
        df_all_feat = pd.concat([df_feat, df_extra_feat], axis=1) # concatenate orginal and extra features
    
    else:
        class_image_datapath = class_datapath
        df_extra_feat = LoadExtraFeatures(class_image_datapath)
        df_extra_feat = df_extra_feat.reset_index(drop=True)
        df_all_feat = df_extra_feat

    return df_all_feat



def HellingerDistance(p, q):
    p = np.array(p)
    q = np.array(q)
    p = np.divide(p, len(p))
    q = np.divide(q, len(q))
    HD = np.linalg.norm(np.sqrt(p) - np.sqrt(q)) / np.sqrt(2)
    return HD
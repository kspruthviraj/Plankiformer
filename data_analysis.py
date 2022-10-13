import os
import time
import argparse

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Plot some figures about data distribution')
parser.add_argument('-datapaths', nargs='*', help='path of the dataset')
parser.add_argument('-datapath_labels', nargs='*', help='name of the dataset')
parser.add_argument('-train_datapath', help='path of train dataset')
parser.add_argument('-outpath', help='path of the output')
parser.add_argument('-selected_features', nargs='*', help='select the features that you want to analyse')
parser.add_argument('-n_bins', type=int, help='number of bins in the feature distribution plot')
args = parser.parse_args()



def PlotSamplingDate(train_datapath, outpath):

    print('-----------------Now plotting sampling date of training set.-----------------')

    list_class = os.listdir(train_datapath) # list of class names

    list_image = []
    # list of all image names in the train dataset
    for iclass in list_class:
        for img in os.listdir(train_datapath + '/%s/' % iclass):
            list_image.append(img)

    list_time = []
    list_date = []
    for img in list_image:
        if img == 'Thumbs.db':
            continue

        timestamp = int(img[15:25])
        localtime = time.localtime(timestamp)        
        t = time.strftime('%Y-%m-%d %H:%M:%S', localtime)
        date = time.strftime('%Y-%m-%d', localtime)
        list_time.append(t)
        list_date.append(date)
    
    list.sort(list_date)

    df = pd.DataFrame({'date': pd.to_datetime(np.unique(list_date)), 'count': pd.value_counts(list_date).sort_index()})
    date_range = pd.date_range(start=df['date'].min(), end=df['date'].max(), freq='D')

    df_full = df.set_index('date').reindex(date_range).fillna(0).rename_axis('date').reset_index() # create a full time range and fill the NA with 0

    ax = plt.subplot(1, 1, 1)
    plt.figure(figsize=(20, 5))
    plt.xlabel('Date')
    plt.ylabel('Image')
    plt.title('Image sampling date in train dataset')

    plt.plot(df_full['date'], df_full['count'])
    plt.yscale('log')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(outpath + 'image_date_train.png')
    plt.close()
    ax.clear()



def PlotSamplingDateEachClass(train_datapath, outpath):

    print('-----------------Now plotting sampling date for each class.-----------------')

    list_class = os.listdir(train_datapath) # list of class names

    ax = plt.subplot(1, 1, 1)
    plt.figure(figsize=(20, 8))
    plt.xlabel('Date')
    plt.ylabel('Image')
    plt.title('Image sampling date for each class')
    plt.yscale('log')
    plt.grid(axis='y')

    for iclass in list_class:
        list_image_class = os.listdir(train_datapath + '/%s/' % iclass)

        list_time_class = []
        list_date_class = []
        for img in list_image_class:
            if img == 'Thumbs.db':
                continue

            timestamp = int(img[15:25])
            localtime = time.localtime(timestamp)        
            t = time.strftime('%Y-%m-%d %H:%M:%S', localtime)
            date = time.strftime('%Y-%m-%d', localtime)
            list_time_class.append(t)
            list_date_class.append(date)

        list.sort(list_date_class)

        df = pd.DataFrame({'date': pd.to_datetime(np.unique(list_date_class)), 'count': pd.value_counts(list_date_class).sort_index()})
        date_range = pd.date_range(start=df['date'].min(), end=df['date'].max(), freq='D')

        df_full = df.set_index('date').reindex(date_range).fillna(0).rename_axis('date').reset_index() # create a full time range and fill the NA with 0

        plt.plot(df_full['date'], df_full['count'], label=iclass)

    plt.legend(bbox_to_anchor=(1, 1), ncol=1)
    plt.tight_layout()
    plt.savefig(outpath + 'image_date_each_class.png')
    plt.close()
    ax.clear()

    

def PlotAbundanceSep(datapaths, outpath):
    '''plot the abundance of datasets seperately'''

    print('-----------------Now plotting abundance distributions of each dataset seperately.-----------------')

    ith = 0
    for idatapath in datapaths:
        ith = ith + 1 # this is the i-th set
        n_class = len(os.listdir(idatapath)) # count the number of classes in dataset
        list_class = os.listdir(idatapath) # list of class names

        list_n_image_class = []
        # list of the numbers of images in each class
        for iclass in list_class:
            if os.path.exists(idatapath + '/%s/training_data/' % iclass):
                n_image_class = len(os.listdir(idatapath + '/%s/training_data/' % iclass))
            else:
                n_image_class = len(os.listdir(idatapath + '/%s/' % iclass))

            list_n_image_class.append(n_image_class)

        dict_class = dict(zip(list_class, list_n_image_class))
        sorted_dict_class = sorted(dict_class.items(), key=lambda x: x[1], reverse=True) # sorted dictionary {class name: number of images}
        
        # plot abundance
        ax = plt.subplot(1, 1, 1)
        ax.set_xlabel('Class')
        ax.set_ylabel('Abundance')

        for i in range(len(sorted_dict_class)):
            plt.bar(sorted_dict_class[i][0], sorted_dict_class[i][1], log=True)
            plt.xticks(rotation=90)

        plt.tight_layout()
        plt.savefig(outpath + 'abundance_set%s.png' % ith)
        plt.close()
        ax.clear()



def PlotAbundance(datapaths, outpath, datapath_labels):
    '''plot the abundance of two datasets together'''

    print('-----------------Now plotting abundance distributions of each dataset together.-----------------')

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
    # print('Repetitive classes of two datasets: {}'.format(list_class_rep))

    list_n_image_class_combined = []
    for idatapath in datapaths:
        list_n_image_class = []
        # list of the numbers of images in each class
        for iclass in list_class_rep:
            if os.path.exists(idatapath + '/%s/training_data/' % iclass):
                n_image_class = len(os.listdir(idatapath + '/%s/training_data/' % iclass))
            else:
                n_image_class = len(os.listdir(idatapath + '/%s/' % iclass))

            list_n_image_class.append(n_image_class)

        list_n_image_class_combined.append(list_n_image_class)


    df_abundance = pd.DataFrame({'class': list_class_rep, 'dataset_1': list_n_image_class_combined[0], 'dataset_2': list_n_image_class_combined[1]})
    df_abundance['ratio'] = df_abundance['dataset_2'] / df_abundance['dataset_1']
    df_abundance_sorted = df_abundance.sort_values(by='ratio', ascending=False, ignore_index=True)

    fig = plt.figure(figsize=(11, 8))
    ax = plt.subplot(1, 1, 1)
    ax.set_xlabel('Class')
    ax.set_ylabel('Abundance')

    x = np.arange(0, len(list_class_rep) * 2, 2)
    width = 0.5
    x1 = x - width / 2
    x2 = x + width / 2

    y1 = df_abundance_sorted['dataset_1']
    y2 = df_abundance_sorted['dataset_2']

    plt.bar(x1, y1, width=0.5, label=datapath_labels[0], log=True)
    plt.bar(x2, y2, width=0.5, label=datapath_labels[1], log=True)
    plt.xticks(x, df_abundance_sorted['class'], rotation=90)

    ax_2 = ax.twinx()
    ax_2.set_ylabel('Ratio')
    ax_2.plot(x, df_abundance_sorted['ratio'], label='ratio', color='green', marker='.')

    fig.legend(loc=1, bbox_to_anchor=(1, 1), bbox_transform=ax.transAxes)
    plt.tight_layout()
    plt.savefig(outpath + 'abundance.png')
    plt.close()
    ax.clear()



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
    # print('Repetitive classes of two datasets: {}'.format(list_class_rep))
    
    # plot feature distribution
    for iclass in list_class_rep:
        for ifeature in selected_features:
            ax = plt.subplot(1, 1, 1)
            ax.set_xlabel(ifeature + ' (normalized)')
            ax.set_ylabel('Density')

            min_feature = []
            max_feature = []
            feature = []

            for idatapath in datapaths:
                class_datapath = idatapath + iclass + '/' # directory of each class with classname
                df_all_feat = ConcatAllFeatures(class_datapath)

                min_feature.append(min(df_all_feat[ifeature]))
                max_feature.append(max(df_all_feat[ifeature]))
                feature.append(df_all_feat[ifeature])

            min_bin = np.min(min_feature) # find global minimum value of feature in all datasets
            max_bin = np.max(max_feature) # find global maximum value of feature in all datasets

            normalized_feature = np.divide((np.array(feature, dtype=object) - min_bin), (max_bin - min_bin)) # normalization of feature values

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



def PlotHDversusBin(datapaths, outpath, selected_features):

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
    # print('Repetitive classes of two datasets: {}'.format(list_class_rep))

    list_n_bins = [5, 10, 20, 50, 100, 120, 150, 200]
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



# def GlobalHD(datapaths):
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
#     print('Repetitive classes of two datasets: {}'.format(list_class_rep))



def HellingerDistance(p, q):
    p = np.array(p)
    q = np.array(q)
    p = np.divide(p, len(p))
    q = np.divide(q, len(q))
    HD = np.linalg.norm(np.sqrt(p) - np.sqrt(q)) / np.sqrt(2)
    return HD


    
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



if __name__ == '__main__':
    # PlotSamplingDate(args.train_datapath, args.outpath)
    PlotSamplingDateEachClass(args.train_datapath, args.outpath)
    # PlotAbundance(args.datapaths, args.outpath, args.datapath_labels)
    # PlotFeatureDistribution(args.datapaths, args.outpath, args.selected_features, args.n_bins, args.datapath_labels)
    # PlotHDversusBin(args.datapaths, args.outpath, args.selected_features)
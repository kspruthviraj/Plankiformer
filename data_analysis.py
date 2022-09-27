import os
import argparse

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Plot some figures about data distribution')
parser.add_argument('-datapaths', nargs='*', help='path of the dataset')
parser.add_argument('-outpath', help='path of the output')
parser.add_argument('-selected_features', nargs='*', help='select the features that you want to analyse')
parser.add_argument('-n_bins', type=int, help='number of bins in the feature distribution plot')
args = parser.parse_args()


def PlotAbundance(datapaths, outpath):
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
        ax.clear()
        
    

def PlotFeatureDistribution(datapaths, outpath, selected_features, n_bins):
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
    print(list_class_rep)
    
    # plot feature distribution
    for iclass in list_class_rep:
        for ifeature in selected_features:
            ax = plt.subplot(1, 1, 1)
            ax.set_xlabel(ifeature)
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

            min_bin = min(min_feature) # find global minimum value of feature in all datasets
            max_bin = max(max_feature) # find global maximum value of feature in all datasets
            
            histogram = plt.hist(feature, histtype='stepfilled', bins=n_bins, range=(min_bin, max_bin), density=True, alpha=0.5)
            
            density_1 = histogram[0][0]
            density_2 = histogram[0][1]

            HD = HellingerDistance(density_1, density_2) # compute the Hellinger distance of feature between 2 datasets
            
            plt.title('Hellinger distance = %.3f' % HD)
            plt.tight_layout()
                
            outpath_feature = outpath + ifeature + '/'
            try:
                os.mkdir(outpath_feature)
            except FileExistsError:
                pass
            plt.savefig(outpath_feature + ifeature + '_' + iclass + '.png')
            ax.clear()



def HellingerDistance(p, q):
    p = np.array(p)
    q = np.array(q)
    HD = 1 / np.sqrt(2) * np.linalg.norm(np.sqrt(p) - np.sqrt(q))
    return HD


    
def LoadExtraFeatures(class_image_datapath, list_image):
    df_extra_feat = pd.DataFrame()
    dfFeatExtra1 = pd.DataFrame(columns=['width', 'height', 'w_rot', 'h_rot', 'angle_rot', 'aspect_ratio_2',
                                            'rect_area', 'contour_area', 'contour_perimeter', 'extent',
                                            'compactness', 'formfactor', 'hull_area', 'solidity_2', 'hull_perimeter',
                                            'ESD', 'Major_Axis', 'Minor_Axis', 'Angle', 'Eccentricity1', 'Eccentricity2',
                                            'Convexity', 'Roundness'])
    dfFeatExtra2 = pd.DataFrame(columns=['m00', 'm10', 'm01', 'm20', 'm11', 'm02', 'm30', 'm21', 'm12', 'm03',
                                            'mu20', 'mu11', 'mu02', 'mu30', 'mu21', 'mu12', 'mu03', 'nu20', 'nu11',
                                            'nu02', 'nu30', 'nu21', 'nu12', 'nu03'])

    for img in list_image:
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
        list_image = os.listdir(class_image_datapath) # list of image names
        df_extra_feat = LoadExtraFeatures(class_image_datapath, list_image)
        df_extra_feat = df_extra_feat.reset_index(drop=True)
        # df_extra_feat.to_csv(class_datapath + 'extra_features.tsv') # save extra features

        # original_features = df_feat.columns.to_list()
        # extra_features = df_extra_feat.columns.to_list()
        # all_features = original_features + extra_features
        # df_all_feat = pd.DataFrame(columns=all_features)
        
        df_all_feat = pd.concat([df_feat, df_extra_feat], axis=1) # concatenate orginal and extra features
    
    else:
        class_image_datapath = class_datapath
        list_image = os.listdir(class_image_datapath) # list of image names
        df_extra_feat = LoadExtraFeatures(class_image_datapath, list_image)
        df_extra_feat = df_extra_feat.reset_index(drop=True)
        df_all_feat = df_extra_feat

    return df_all_feat



if __name__ == '__main__':
    PlotAbundance(args.datapaths, args.outpath)
    PlotFeatureDistribution(args.datapaths, args.outpath, args.selected_features, args.n_bins)





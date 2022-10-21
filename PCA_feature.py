import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from utils_analysis import feature as f


def ConcactAllClasses(datapath):

    '''Concatenate features of all images (all classes) in a dataset.'''

    list_class = os.listdir(datapath)
    df_all_feat = pd.DataFrame()

    for iclass in list_class:
        class_datapath = datapath + iclass + '/'
        df_class_feat = f.ConcatAllFeatures(class_datapath)
        df_class_feat['class'] = iclass
        df_all_feat = pd.concat([df_all_feat, df_class_feat], ignore_index=True)
    
    return df_all_feat


def Standardize(dataframe):

    '''Standardize an input pandas dataframe.'''

    data = dataframe.iloc[:, :-1].values
    data = StandardScaler().fit_transform(data)
    cols = dataframe.columns[:-1]
    cols = [i + '_standardized' for i in cols]
    df_standardized = pd.DataFrame(data, columns=cols)
    df_standardized['class'] = dataframe['class']

    return df_standardized


def PrincipalComponentAnalysis(dataframe, n_components):

    '''Principal component analysis on a dataframe.'''

    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(dataframe.iloc[:, :-1].values)
    df_pca = pd.DataFrame(data=principal_components, columns=['principal_component_{}'.format(i+1) for i in range(n_components)])
    df_pca['class'] = dataframe['class']

    return pca, df_pca


def PCA_train_val_test(dataframe, pca):

    '''Implement PCA on in-distribution datasets.'''

    principal_components = pca.fit_transform(dataframe.iloc[:, :-1].values)
    df_pca_split = pd.DataFrame(data=principal_components, columns=['principal_component_{}'.format(i+1) for i in range(np.shape(principal_components)[1])])
    df_pca_split['class'] = dataframe['class']

    return df_pca_split


def PCA_OOD(dataframe_OOD, pca):

    '''Implement PCA on out-of-distribution datasets.'''

    principal_components = pca.fit_transform(dataframe_OOD.iloc[:, :-1].values)
    df_pca_OOD = pd.DataFrame(data=principal_components, columns=['principal_component_{}'.format(i+1) for i in range(np.shape(principal_components)[1])])
    df_pca_OOD['class'] = dataframe_OOD['class']

    return df_pca_OOD


parser = argparse.ArgumentParser(description='Principal component analysis on datasets')
parser.add_argument('-Zoolake2_datapath', help='path of the Zoolake2 dataset')
parser.add_argument('-in_distribution_datapaths', nargs='*', help='paths of the in-domain datasets, in an order of: train_val_test')
parser.add_argument('-OOD_datapaths', nargs='*', help='paths of the out-of-distribution datasets')
parser.add_argument('-outpath', help='path for saving output csv')
parser.add_argument('-n_components', type=int, help='number of principal components')
args = parser.parse_args()


if __name__ == '__main__':

    df = ConcactAllClasses(args.Zoolake2_datapath)
    df_standardized = Standardize(df)
    pca, df_pca = PrincipalComponentAnalysis(df_standardized, n_components=args.n_components)
    df_pca.to_csv(args.outpath + 'PCA_Zoolake2_feature.csv')

    np.savetxt(args.outpath + 'PCA_explained_variance_ratio_feature.txt', pca.explained_variance_ratio_)

    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of components')
    plt.ylabel('Explained variance ratio')
    plt.grid()
    plt.tight_layout()
    plt.savefig(args.outpath + 'PCA_explained_variance_ratio_feature.png')

    
    df_train = ConcactAllClasses(args.in_distribution_datapaths[0])
    df_val = ConcactAllClasses(args.in_distribution_datapaths[1])
    df_test = ConcactAllClasses(args.in_distribution_datapaths[2])
    df_train_standardized = Standardize(df_train)
    df_val_standardized = Standardize(df_val)
    df_test_standardized = Standardize(df_test)
    df_pca_train = PCA_train_val_test(df_train_standardized, pca)
    df_pca_val = PCA_train_val_test(df_val_standardized, pca)
    df_pca_test = PCA_train_val_test(df_test_standardized, pca)
    df_pca_train.to_csv(args.outpath + 'PCA_train_feature.csv')
    df_pca_val.to_csv(args.outpath + 'PCA_val_feature.csv')
    df_pca_test.to_csv(args.outpath + 'PCA_test_feature.csv')

    for i in range(len(args.OOD_datapaths)):
        df_OOD = ConcactAllClasses(args.OOD_datapaths[i])
        df_OOD_standardized = Standardize(df_OOD)
        df_pca_OOD = PCA_OOD(df_OOD_standardized, pca)
        df_pca_OOD.to_csv(args.outpath + 'PCA_OOD{}_feature.csv'.format(i + 1))

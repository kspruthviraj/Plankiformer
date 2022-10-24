import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def PlotPixelDistribution_PCA(PCA_files_pixel, outpath, selected_components_pixel, n_bins_pixel, data_labels, image_threshold):

    print('-----------------Now plotting PCA pixel distribution for each class and each selected component.-----------------')
    
    df_pca_1 = pd.read_csv(PCA_files_pixel[0], sep=',', index_col=0)
    df_pca_2 = pd.read_csv(PCA_files_pixel[1], sep=',', index_col=0)

    list_class_1 = np.unique(df_pca_1['class'])
    list_class_2 = np.unique(df_pca_2['class'])
    list_class_rep = list(set(list_class_1) & set(list_class_2))
    list.sort(list_class_rep)

    for iclass in list_class_rep:
        df_pca_1_class = df_pca_1[df_pca_1['class']==iclass].drop(['class'], axis=1).reset_index()
        df_pca_2_class = df_pca_2[df_pca_2['class']==iclass].drop(['class'], axis=1).reset_index()

        if not (df_pca_1_class.shape[0] >= image_threshold and df_pca_2_class.shape[0] >= image_threshold):
            continue

        for ipc in selected_components_pixel:
            ax = plt.subplot(1, 1, 1)
            ax.set_xlabel(ipc + ' (normalized)')
            ax.set_ylabel('Density')

            component_1 = df_pca_1_class[ipc]
            component_2 = df_pca_2_class[ipc]
            components = [component_1, component_2]

            min_bin = np.min([min(component_1), min(component_2)])
            max_bin = np.max([max(component_1), max(component_2)])

            normalized_component = np.divide((np.array(components, dtype=object) - min_bin), (max_bin - min_bin))

            histogram = plt.hist(normalized_component, histtype='stepfilled', bins=n_bins_pixel, range=(0, 1), density=True, alpha=0.5, label=data_labels)
            density_1 = histogram[0][0]
            density_2 = histogram[0][1]

            HD = HellingerDistance(density_1, density_2)
            
            plt.title('Hellinger distance = %.3f' % HD)
            plt.legend()
            plt.tight_layout()

            outpath_component = outpath + ipc + '/'
            try:
                os.mkdir(outpath_component)
            except FileExistsError:
                pass
            plt.savefig(outpath_component + ipc + '_' + iclass + '.png')
            plt.close()
            ax.clear()


def PlotGlobalHDversusBin_pixel_PCA(PCA_files_pixel, outpath, explained_variance_ratio_pixel, image_threshold):

    print('-----------------Now plotting global Hellinger distances of PCA pixel v.s. numbers of bin.-----------------')

    df_pca_1 = pd.read_csv(PCA_files_pixel[0], sep=',', index_col=0)
    df_pca_2 = pd.read_csv(PCA_files_pixel[1], sep=',', index_col=0)

    # list_class_1 = np.unique(df_pca_1['class'])
    # list_class_2 = np.unique(df_pca_2['class'])
    # list_class_rep = list(set(list_class_1) & set(list_class_2))
    # list.sort(list_class_rep)

    class_1 = df_pca_1['class'].to_list()
    class_2 = df_pca_2['class'].to_list()
    df_count_1 = pd.DataFrame(np.unique(class_1, return_counts=True)).transpose()
    df_count_2 = pd.DataFrame(np.unique(class_2, return_counts=True)).transpose()
    list_class_1 = df_count_1[df_count_1.iloc[:, 1]>=image_threshold].iloc[:, 0].to_list()
    list_class_2 = df_count_2[df_count_2.iloc[:, 1]>=image_threshold].iloc[:, 0].to_list()
    list_class_rep = list(set(list_class_1) & set(list_class_2))
    list.sort(list_class_rep)

    list_principal_components = df_pca_1.columns.to_list()[:-1]
    list_explained_variance_ratio = np.loadtxt(explained_variance_ratio_pixel)
    list_n_bins = [5, 10, 20, 50, 100, 125, 150, 175, 200, 300, 400, 500, 750, 1000]

    df_global_HD = pd.DataFrame(columns=[in_bins for in_bins in list_n_bins], index=[iclass for iclass in list_class_rep])
    for in_bins in list_n_bins:
        list_global_HD = []
        for iclass in list_class_rep:
            df_pca_1_class = df_pca_1[df_pca_1['class']==iclass].drop(['class'], axis=1).reset_index()
            df_pca_2_class = df_pca_2[df_pca_2['class']==iclass].drop(['class'], axis=1).reset_index()

            list_HD = []
            for ipc in list_principal_components:
                component_1 = df_pca_1_class[ipc]
                component_2 = df_pca_2_class[ipc]
                components = [component_1, component_2]

                min_bin = np.min([min(component_1), min(component_2)])
                max_bin = np.max([max(component_1), max(component_2)])

                normalized_component = np.divide((np.array(components, dtype=object) - min_bin), (max_bin - min_bin))

                histogram_1 = np.histogram(normalized_component[0], bins=in_bins, range=(0, 1), density=True)
                histogram_2 = np.histogram(normalized_component[1], bins=in_bins, range=(0, 1), density=True)
                density_1 = histogram_1[0]
                density_2 = histogram_2[0]

                HD = HellingerDistance(density_1, density_2)
                list_HD.append(HD)

            # global_HD_each_class = np.average(list_HD, weights=list_explained_variance_ratio)
            global_HD_each_class = np.average(list_HD)
            list_global_HD.append(global_HD_each_class)

        df_global_HD[in_bins] = list_global_HD

    df_global_HD = df_global_HD.transpose()

    ax = plt.subplot(1, 1, 1)
    ax.set_xlabel('Number of bins')
    ax.set_ylabel('Hellinger Distance')
    plt.figure(figsize=(10, 10))
    
    for iclass in list_class_rep:
        plt.plot(list_n_bins, df_global_HD[iclass], label=iclass)

    plt.legend(bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
    plt.tight_layout()

    try:
        os.mkdir(outpath)
    except FileExistsError:
        pass
    plt.savefig(outpath + 'HD_bin_pixel_PCA.png')
    plt.close()
    ax.clear()
    

def GlobalHD_pixel(PCA_files_pixel, outpath, n_bins_pixel, explained_variance_ratio_pixel, image_threshold):

    print('-----------------Now computing global Hellinger distances on PCA pixel.-----------------')

    df_pca_1 = pd.read_csv(PCA_files_pixel[0], sep=',', index_col=0)
    df_pca_2 = pd.read_csv(PCA_files_pixel[1], sep=',', index_col=0)

    list_class_1 = np.unique(df_pca_1['class'])
    list_class_2 = np.unique(df_pca_2['class'])
    list_class_rep = list(set(list_class_1) & set(list_class_2))
    list.sort(list_class_rep)

    list_principal_components = df_pca_1.columns.to_list()[:-1]
    list_explained_variance_ratio = np.loadtxt(explained_variance_ratio_pixel)

    list_global_HD = []
    for iclass in list_class_rep:
        df_pca_1_class = df_pca_1[df_pca_1['class']==iclass].drop(['class'], axis=1).reset_index()
        df_pca_2_class = df_pca_2[df_pca_2['class']==iclass].drop(['class'], axis=1).reset_index()

        if not (df_pca_1_class.shape[0] >= image_threshold and df_pca_2_class.shape[0] >= image_threshold):
            continue
        
        list_HD = []
        for ipc in list_principal_components:
            component_1 = df_pca_1_class[ipc]
            component_2 = df_pca_2_class[ipc]
            components = [component_1, component_2]

            min_bin = np.min([min(component_1), min(component_2)])
            max_bin = np.max([max(component_1), max(component_2)])

            normalized_component = np.divide((np.array(components, dtype=object) - min_bin), (max_bin - min_bin))

            histogram_1 = np.histogram(normalized_component[0], bins=n_bins_pixel, range=(0, 1), density=True)
            histogram_2 = np.histogram(normalized_component[1], bins=n_bins_pixel, range=(0, 1), density=True)
            density_1 = histogram_1[0]
            density_2 = histogram_2[0]

            HD = HellingerDistance(density_1, density_2)
            list_HD.append(HD)

        # global_HD_each_class = np.average(list_HD, weights=list_explained_variance_ratio)
        global_HD_each_class = np.average(list_HD)
        list_global_HD.append(global_HD_each_class)

        with open(outpath + 'Global_HD_pixel_PCA.txt', 'a') as f:
            f.write('%-20s%-20f\n' % (iclass, global_HD_each_class))

    global_HD = np.average(list_global_HD)
    with open(outpath + 'Global_HD_pixel_PCA.txt', 'a') as f:
        f.write(f'\n Global Hellinger Distance: {global_HD}\n')


def HellingerDistance(p, q):
    p = np.array(p)
    q = np.array(q)
    p = np.divide(p, len(p))
    q = np.divide(q, len(q))
    HD = np.linalg.norm(np.sqrt(p) - np.sqrt(q)) / np.sqrt(2)
    return HD
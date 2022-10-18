import os
from cv2 import log

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def PlotAbundanceSep(datapaths, outpath, datapath_labels):
    '''plot the abundance of datasets seperately'''

    print('-----------------Now plotting abundance distributions of each dataset seperately.-----------------')

    for j, idatapath in enumerate(datapaths):
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
        plt.savefig(outpath + 'abundance_%s.png' % datapath_labels[j])
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
        list.sort(list_class_rep)
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

    total_image_1 = np.sum(list_n_image_class_combined[0])
    total_image_2 = np.sum(list_n_image_class_combined[1])
    df_abundance = pd.DataFrame({'class': list_class_rep, 'dataset_1': np.divide(list_n_image_class_combined[0], total_image_1), 'dataset_2': np.divide(list_n_image_class_combined[1], total_image_2)})
    # df_abundance = pd.DataFrame({'class': list_class_rep, 'dataset_1': list_n_image_class_combined[0], 'dataset_2': list_n_image_class_combined[1]})
    df_abundance['ratio'] = df_abundance['dataset_2'] / df_abundance['dataset_1']
    df_abundance_sorted = df_abundance.sort_values(by='dataset_1', ascending=False, ignore_index=True)

    fig = plt.figure(figsize=(11, 8))
    ax = plt.subplot(1, 1, 1)
    ax.set_xlabel('Class')
    ax.set_ylabel('Abundance (percentage)')
    # ax.set_ylabel('Abundance')

    x = np.arange(0, len(list_class_rep) * 2, 2)
    width = 0.5
    x1 = x - width / 2
    x2 = x + width / 2

    y1 = df_abundance_sorted['dataset_1']
    y2 = df_abundance_sorted['dataset_2']

    plt.bar(x1, y1, width=0.5, label=datapath_labels[0])
    plt.bar(x2, y2, width=0.5, label=datapath_labels[1])
    # plt.bar(x1, y1, width=0.5, label=datapath_labels[0], log=True)
    # plt.bar(x2, y2, width=0.5, label=datapath_labels[1], log=True)
    plt.xticks(x, df_abundance_sorted['class'], rotation=90)

    ax_2 = ax.twinx()
    ax_2.set_ylabel('Ratio of two datasets')
    ax_2.plot(x, df_abundance_sorted['ratio'], label='ratio', color='green', marker='.')

    fig.legend(loc=1, bbox_to_anchor=(1, 1), bbox_transform=ax.transAxes)
    plt.tight_layout()
    plt.savefig(outpath + 'abundance.png')
    plt.close()
    ax.clear()
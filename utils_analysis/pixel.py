import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2


def GlobalHD_pixel(datapaths, outpath, n_bins, resized_length=64):
    
    print('-----------------Now computing global Hellinger distances on pixel.-----------------')

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

    list_global_HD = []
    all_pixels = [([0] * len(datapaths)) for i in range(len(list_class_rep))]

    for i, iclass in enumerate(list_class_rep):
        list_HD = []
   
        for j, idatapath in enumerate(datapaths):
            class_datapath = idatapath + iclass + '/'
            df_pixels = LoadPixels(class_datapath, resized_length)
            all_pixels[i][j] = df_pixels

        for ipixel in range(np.square(resized_length)*3):
            pixels = []

            for j, idatapath in enumerate(datapaths):
                pixel = all_pixels[i][j][ipixel]
                pixels.append(pixel)

            pixels = np.divide(np.array(pixels, dtype=object), 255)

            histogram_1 = np.histogram(pixels[0], bins=n_bins, range=(0, 1), density=True)
            histogram_2 = np.histogram(pixels[1], bins=n_bins, range=(0, 1), density=True)
            density_1 = histogram_1[0]
            density_2 = histogram_2[0]

            HD = HellingerDistance(density_1, density_2)
            list_HD.append(HD)

        global_HD_each_class = np.average(list_HD)
        list_global_HD.append(global_HD_each_class)

        with open(outpath + 'Global_HD_pixel.txt', 'a') as f:
            f.write('{}: {}\n'.format(iclass, global_HD_each_class))

    global_HD = np.average(list_global_HD)
    with open(outpath + 'Global_HD_pixel.txt', 'a') as f:
        f.write(f'\n Global Hellinger Distance: {global_HD}')



# def GlobalHD_pixel(datapaths, outpath, n_bins, resized_length=64):
    
#     print('-----------------Now computing global Hellinger distances on pixel.-----------------')

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

#     list_global_HD = []
#     for iclass in list_class_rep:
#         list_HD = []

#         for ipixel in range(np.square(resized_length)*3):
#             pixels = []

#             for idatapath in datapaths:
#                 class_datapath = idatapath + iclass + '/'
#                 df_pixels = LoadPixels(class_datapath, resized_length)
                
#                 pixels.append(df_pixels[ipixel])

#             pixels = np.divide(np.array(pixels, dtype=object), 255)

#             histogram_1 = np.histogram(pixels[0], bins=n_bins, range=(0, 1), density=True)
#             histogram_2 = np.histogram(pixels[1], bins=n_bins, range=(0, 1), density=True)
#             density_1 = histogram_1[0]
#             density_2 = histogram_2[0]

#             HD = HellingerDistance(density_1, density_2)
#             list_HD.append(HD)

#         global_HD_each_class = np.average(list_HD)
#         list_global_HD.append(global_HD_each_class)

#         with open(outpath + 'Global_HD_pixel.txt', 'a') as f:
#             f.write('{}: {}\n'.format(iclass, global_HD_each_class))

#     global_HD = np.average(list_global_HD)
#     with open(outpath + 'Global_HD_pixel.txt', 'a') as f:
#         f.write(f'\n Global Hellinger Distance: {global_HD}')



def LoadPixels(class_image_datapath, resized_length):
    df_pixels = pd.DataFrame()

    list_image = os.listdir(class_image_datapath)
    for img in list_image:
        if img == 'Thumbs.db':
            continue
        
        image = cv2.imread(class_image_datapath + '/' + img)
        image = cv2.resize(image, (resized_length, resized_length), interpolation=cv2.INTER_LANCZOS4)
        pixels = pd.Series(image.flatten())
        
        df_pixels = pd.concat([df_pixels, pixels], axis=1, ignore_index=True)

    df_pixels = df_pixels.transpose()

    return df_pixels



def HellingerDistance(p, q):
    p = np.array(p)
    q = np.array(q)
    p = np.divide(p, len(p))
    q = np.divide(q, len(q))
    HD = np.linalg.norm(np.sqrt(p) - np.sqrt(q)) / np.sqrt(2)
    return HD
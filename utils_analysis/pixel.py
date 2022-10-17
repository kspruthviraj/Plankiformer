import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2



def PlotPixelDistribution(datapaths, outpath, selected_pixels, n_bins_pixel, datapath_labels, resized_length=64):

    print('-----------------Now plotting pixel distribution for each class and each selected feature.-----------------')

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
    
    all_pixels = [([0] * len(datapaths)) for i in range(len(list_class_rep))]
    for i, iclass in enumerate(list_class_rep):

        for j, idatapath in enumerate(datapaths):
            class_datapath = idatapath + iclass + '/' # directory of each class with classname
            df_pixels = LoadPixels(class_datapath, resized_length)
            all_pixels[i][j] = df_pixels

        for ipixel in selected_pixels:
            ax = plt.subplot(1, 1, 1)
            ax.set_xlabel(ipixel + ' (normalized)')
            ax.set_ylabel('Density')

            pixels = []

            for j, idatapath in enumerate(datapaths):
                pixel = all_pixels[i][j][int(ipixel)]
                pixels.append(pixel)

            pixels = np.divide(np.array(pixels, dtype=object), 255)

            histogram = plt.hist(pixels, histtype='stepfilled', bins=n_bins_pixel, range=(0, 1), density=True, alpha=0.5, label=datapath_labels)
            density_1 = histogram[0][0]
            density_2 = histogram[0][1]

            HD = HellingerDistance(density_1, density_2)

            plt.title('Hellinger distance = %.3f' % HD)
            plt.legend()
            plt.tight_layout()

            outpath_pixel = outpath + ipixel + '/'
            try:
                os.mkdir(outpath_pixel)
            except FileExistsError:
                pass
            plt.savefig(outpath_pixel + ipixel + '_' + iclass + '.png')
            plt.close()
            ax.clear()



def PlotPixelHDversusBin(datapaths, outpath, selected_pixels, resized_length=64):

    print('-----------------Now plotting Hellinger distances of pixel v.s. numbers of bin.-----------------')

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
    all_pixels = [([0] * len(datapaths)) for i in range(len(list_class_rep))]

    for i, iclass in enumerate(list_class_rep):

        for j, idatapath in enumerate(datapaths):
            class_datapath = idatapath + iclass + '/'
            df_pixels = LoadPixels(class_datapath, resized_length)
            all_pixels[i][j] = df_pixels

    for ipixel in selected_pixels:
        ax = plt.subplot(1, 1, 1)
        plt.figure(figsize=(10, 10))

        for i, iclass in enumerate(list_class_rep):
            list_HD = []
            for in_bins in list_n_bins:
                pixels = []

                for j, idatapath in enumerate(datapaths):
                    pixel = all_pixels[i][j][int(ipixel)]

                    pixels.append(pixel)

                pixels = np.divide(np.array(pixels, dtype=object), 255)

                histogram_1 = np.histogram(pixels[0], bins=in_bins, range=(0, 1), density=True)
                histogram_2 = np.histogram(pixels[1], bins=in_bins, range=(0, 1), density=True)
                density_1 = histogram_1[0]
                density_2 = histogram_2[0]

                HD = HellingerDistance(density_1, density_2) # compute the Hellinger distance of feature between 2 datasets
                list_HD.append(HD)

            plt.plot(list_n_bins, list_HD, label=iclass)

        ax.set_xlabel('Number of bins')
        ax.set_ylabel('Hellinger Distance')
        ax.set_title(ipixel)
        plt.legend(bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
        plt.tight_layout()

        outpath_pixel = outpath + ipixel + '/'
        try:
            os.mkdir(outpath_pixel)
        except FileExistsError:
            pass
        plt.savefig(outpath_pixel + ipixel + '_HD.png' )
        plt.close()
        ax.clear()



def PlotGlobalHDversusBin_pixel(datapaths, outpath, resized_length=64):

    print('-----------------Now plotting global Hellinger distances of pixel v.s. numbers of bin.-----------------')

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

    
    list_n_bins = [5, 10, 20, 50, 100, 125, 150, 175, 200, 300, 400, 500, 750, 1000]
    all_pixels = [([0] * len(datapaths)) for i in range(len(list_class_rep))]

    df_global_HD = pd.DataFrame(columns=[in_bins for in_bins in list_n_bins], index=[iclass for iclass in list_class_rep])
    for in_bins in list_n_bins:
        list_global_HD = []
        for i, iclass in enumerate(list_class_rep):
            list_HD = []

            for j, idatapath in enumerate(datapaths):
                class_datapath = idatapath + iclass + '/'
                df_pixels = LoadPixels(class_datapath, resized_length)
                all_pixels[i][j] = df_pixels

            for ipixel in range(np.square(resized_length)*3):
                pixels = []

                for j, idatapath in enumerate(datapaths):
                    pixel = all_pixels[i][j][int(ipixel)]
                    pixels.append(pixel)

                pixels = np.divide(np.array(pixels, dtype=object), 255)

                histogram_1 = np.histogram(pixels[0], bins=in_bins, range=(0, 1), density=True)
                histogram_2 = np.histogram(pixels[1], bins=in_bins, range=(0, 1), density=True)
                density_1 = histogram_1[0]
                density_2 = histogram_2[0]

                HD = HellingerDistance(density_1, density_2) # compute the Hellinger distance of feature between 2 datasets
                list_HD.append(HD)

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

    outpath_pixel = outpath
    try:
        os.mkdir(outpath_pixel)
    except FileExistsError:
        pass
    plt.savefig(outpath_pixel + 'HD_bin_pixel.png' )
    plt.close()
    ax.clear()



def GlobalHD_pixel(datapaths, outpath, n_bins_pixel, resized_length=64):
    
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
                pixel = all_pixels[i][j][int(ipixel)]
                pixels.append(pixel)

            pixels = np.divide(np.array(pixels, dtype=object), 255)

            histogram_1 = np.histogram(pixels[0], bins=n_bins_pixel, range=(0, 1), density=True)
            histogram_2 = np.histogram(pixels[1], bins=n_bins_pixel, range=(0, 1), density=True)
            density_1 = histogram_1[0]
            density_2 = histogram_2[0]

            HD = HellingerDistance(density_1, density_2)
            list_HD.append(HD)

        global_HD_each_class = np.average(list_HD)
        list_global_HD.append(global_HD_each_class)

        with open(outpath + 'Global_HD_pixel.txt', 'a') as f:
            # f.write('{}: {}\n'.format(iclass, global_HD_each_class))
            f.write('%-20s%-20f\n' % (iclass, global_HD_each_class))

    global_HD = np.average(list_global_HD)
    with open(outpath + 'Global_HD_pixel.txt', 'a') as f:
        f.write(f'\n Global Hellinger Distance: {global_HD}')



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
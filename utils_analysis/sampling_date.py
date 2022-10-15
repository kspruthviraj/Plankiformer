import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def global_population_bias(population_counts, outpath):

    '''Sum the population bias of all dataset up.'''
    
    classes = []
    for i in population_counts:
        df_count = pd.read_excel(i, index_col=0)
        class_name = df_count.index
        classes = list(set(np.unique(classes)).union(set(np.unique(class_name))))
    classes.sort()

    df_population = pd.DataFrame(index=classes, columns=['Predict', 'Ground_truth', 'Bias'])
    for i in population_counts:
        df_count = pd.read_csv(i, sep='\t', index_col=0)
        class_name = df_count.index
        for j in class_name:
            predict_count = df_count.loc[j, 'Predict']
            GT_count = df_count.loc[j, 'Ground_truth']
            Bias = df_count.loc[j, 'Bias']

            df_population.loc[j, 'Predict'] += predict_count
            df_population.loc[j, 'Ground_truth'] += GT_count
            df_population.loc[j, 'Bias'] += Bias

    df_population.to_csv(outpath + 'Global_population.txt', sep='\t', index=True, header=True)

    GT_percentage = df_population['Ground_truth'] / np.sum(df_population['Ground_truth'])
    pred_percentage = df_population['Predict'] / np.sum(df_population['Predict'])
    
    plt.figure()
    plt.scatter(GT_percentage, pred_percentage, label=df_population.index)
    plt.xlabel('Real population')
    plt.ylabel('Predicted population')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath + 'Global population.png')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def global_population_bias(population_counts, outpath):

    '''Sum the population bias of all dataset up.'''

    classes = []
    for i in population_counts:
        df_count = pd.read_excel(i, index_col=0)
        class_name = df_count.index.tolist()
        classes = list(set(np.unique(classes)).union(set(np.unique(class_name))))
    classes.sort()

    df_population = pd.DataFrame(0, index=classes, columns=['Predict', 'Ground_truth', 'Bias'])
    for i in population_counts:
        df_count = pd.read_excel(i, index_col=0)
        class_name = df_count.index.tolist()
        for j in class_name:
            predict_count = df_count.loc[j, 'Predict']
            GT_count = df_count.loc[j, 'Ground_truth']
            Bias = df_count.loc[j, 'Bias']

            df_population.loc[j, 'Predict'] += predict_count
            df_population.loc[j, 'Ground_truth'] += GT_count
            df_population.loc[j, 'Bias'] += Bias

    df_population.to_excel(outpath + 'Global_population.xlsx', index=True, header=True)

    GT_percentage = df_population['Ground_truth'] / np.sum(df_population['Ground_truth'])
    pred_percentage = df_population['Predict'] / np.sum(df_population['Predict'])

    plt.figure(figsize=(10, 10))
    plt.plot((0, 1), (0, 1), ls=':')
    plt.xlabel('Real population')
    plt.ylabel('Predicted population')

    for j in df_population.index.tolist():
        plt.scatter(GT_percentage[j], pred_percentage[j], label=j)

    plt.xlim([0, 1.05 * max(max(GT_percentage), max(pred_percentage))])
    plt.ylim([0, 1.05 * max(max(GT_percentage), max(pred_percentage))])
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath + 'Global population.png')
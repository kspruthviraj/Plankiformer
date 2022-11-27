import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


parser = argparse.ArgumentParser(description='Generate an overview of test performance')
parser.add_argument('-model_names', nargs='*', help='list of model names')
# parser.add_argument('-dataset_names', nargs='*', help='list of test dataset names')
parser.add_argument('-model_performance_paths', nargs='*', help='list of performance paths for each model')
parser.add_argument('-outpath', help='path for saving the overview')
args = parser.parse_args()


def plot_performance_overview(model_name, test_dataset, accuracy, f1_score, outpath):
    plt.figure(figsize=(12, 7))
    plt.subplot(1, 2, 1)
    plt.xticks(np.arange(len(test_dataset)), labels=test_dataset, rotation=45, rotation_mode='anchor', ha='right')
    plt.yticks(np.arange(len(model_name)), labels=model_name)
    plt.title('Accuracy')
    for i in range(len(model_name)):
        for j in range(len(test_dataset)):
            text = plt.text(j, i, format(accuracy[i, j], '.3f'), ha='center', va='center', color='black')
    plt.imshow(accuracy, cmap='RdYlGn')
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.xticks(np.arange(len(test_dataset)), labels=test_dataset, rotation=45, rotation_mode='anchor', ha='right')
    plt.yticks(np.arange(len(model_name)), labels=model_name)
    plt.title('F1-score')
    for i in range(len(model_name)):
        for j in range(len(test_dataset)):
            text = plt.text(j, i, format(f1_score[i, j], '.3f'), ha='center', va='center', color='black')
    plt.imshow(f1_score, cmap='RdYlGn')
    plt.colorbar()

    plt.tight_layout()
    plt.savefig(outpath + 'test_performance_overview.png')
    plt.close()


def read_test_report(test_report_file):
    test_report = pd.read_csv(test_report_file)
    accuracy_value = format(float(test_report.iloc[0].item()), '.3f')
    f1_value = format(float(test_report.iloc[2].item()), '.3f')

    return accuracy_value, f1_value


def performance_matrix(model_performance_paths):
    n_model = len(model_performance_paths)
    n_dataset = len(os.listdir(model_performance_paths[0]))
    test_dataset = os.listdir(model_performance_paths[0])
    test_dataset.sort()

    accuracy = np.zeros([n_model, n_dataset])
    f1_score = np.zeros([n_model, n_dataset])

    for i, imodel_path in enumerate(model_performance_paths):
        dataset_names = os.listdir(imodel_path)
        dataset_names.sort()
        for j, idataset in enumerate(dataset_names):
            test_report_path = imodel_path + '/' + idataset + '/'

            if 'Single_test_report_finetuned.txt' in os.listdir(test_report_path):
                test_report_file = test_report_path + 'Single_test_report_finetuned.txt'
            elif 'Ensemble_test_report_geo_mean_finetuned.txt' in os.listdir(test_report_path):
                test_report_file = test_report_path + 'Ensemble_test_report_geo_mean_finetuned.txt'

            # if 'Single_test_report_finetuned.txt' in os.listdir(test_report_path):
            #     test_report_file = test_report_path + 'Single_test_report_finetuned.txt'
            # elif 'Ensemble_test_report_rm_unknown_geo_mean_finetuned.txt' in os.listdir(test_report_path):
            #     test_report_file = test_report_path + 'Ensemble_test_report_rm_unknown_geo_mean_finetuned.txt'

            # if 'Single_test_report_tuned.txt' in os.listdir(test_report_path):
            #     test_report_file = test_report_path + 'Single_test_report_tuned.txt'
            # elif 'Ensemble_test_report_geo_mean_tuned.txt' in os.listdir(test_report_path):
            #     test_report_file = test_report_path + 'Ensemble_test_report_geo_mean_tuned.txt'

            accuracy_value, f1_value = read_test_report(test_report_file)
            accuracy[i, j], f1_score[i, j] = accuracy_value, f1_value
    
    return accuracy, f1_score, test_dataset


def plot_performance_curve(model_name, test_dataset, accuracy, f1_score, outpath):
    plt.figure(figsize=(12, 7))
    plt.subplot(1, 2, 1)
    plt.xticks(np.arange(len(test_dataset)), labels=test_dataset, rotation=45, rotation_mode='anchor', ha='right')
    plt.title('Accuracy')
    for i, j in zip(accuracy, model_name):
        plt.plot(i, label=j)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.xticks(np.arange(len(test_dataset)), labels=test_dataset, rotation=45, rotation_mode='anchor', ha='right')
    plt.title('F1-score')
    for i, j in zip(f1_score, model_name):
        plt.plot(i, label=j)
    plt.legend()

    plt.tight_layout()
    plt.savefig(outpath + 'test_performance_curves.png')
    plt.close()


if __name__ == '__main__':
    accuracy, f1_score, test_dataset = performance_matrix(args.model_performance_paths)
    model_name = args.model_names
    outpath = args.outpath
    plot_performance_overview(model_name, test_dataset, accuracy, f1_score, outpath)
    plot_performance_curve(model_name, test_dataset, accuracy, f1_score, outpath)


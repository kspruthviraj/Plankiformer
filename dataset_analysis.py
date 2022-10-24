import argparse

from utils_analysis import feature
from utils_analysis import pixel
from utils_analysis import abundance
from utils_analysis import sampling_date as sdate
from utils_analysis import feature_components as fc
from utils_analysis import pixel_components as pc


parser = argparse.ArgumentParser(description='Plot some figures about data distribution')
parser.add_argument('-datapaths', nargs='*', help='path of the dataset')
parser.add_argument('-datapath_labels', nargs='*', help='name of the dataset')
parser.add_argument('-train_datapath', help='path of train dataset')
parser.add_argument('-outpath', help='path of the output')
parser.add_argument('-selected_features', nargs='*', help='select the features that you want to analyse')
parser.add_argument('-selected_pixels', nargs='*', help='select the pixels that you want to analyse')
parser.add_argument('-n_bins_feature', type=int, help='number of bins in the feature distribution plot')
parser.add_argument('-n_bins_pixel', type=int, help='number of bins in the pixel distribution plot')
parser.add_argument('-resized_length', type=int, help='length of resized image')

parser.add_argument('-PCA_files', nargs='*', help='principal components file of all datasets')
parser.add_argument('-selected_components_feature', nargs='*', help='select the feature components that you want to analyse')
parser.add_argument('-selected_components_pixel', nargs='*', help='select the pixel components that you want to analyse')
parser.add_argument('-data_labels', nargs='*', help='name of the dataset')
parser.add_argument('-explained_variance_ratio_feature', help='explained variance ratio file of feature components')
parser.add_argument('-explained_variance_ratio_pixel', help='explained variance ratio file of pixel components')
parser.add_argument('-PCA', choices=['yes', 'no'], default='yes', help='apply PCA or not')
parser.add_argument('-feature_or_pixel', help='analysis on features or pixels')
args = parser.parse_args()


if __name__ == '__main__':

    sdate.PlotSamplingDate(args.train_datapath, args.outpath)
    sdate.PlotSamplingDateEachClass(args.train_datapath, args.outpath)
    abundance.PlotAbundance(args.datapaths, args.outpath, args.datapath_labels)
    abundance.PlotAbundanceSep(args.datapaths, args.outpath, args.datapath_labels)

    if args.PCA == 'no':
        feature.PlotFeatureDistribution(args.datapaths, args.outpath, args.selected_features, args.n_bins_feature, args.datapath_labels)
        # feature.PlotFeatureHDversusBin(args.datapaths, args.outpath, args.selected_features)
        feature.PlotGlobalHDversusBin_feature(args.datapaths, args.outpath)
        feature.GlobalHD_feature(args.datapaths, args.outpath, args.n_bins_feature)
        pixel.PlotPixelDistribution(args.datapaths, args.outpath, args.selected_pixels, args.n_bins_pixel, args.datapath_labels, args.resized_length)
        # pixel.PlotPixelHDversusBin(args.datapaths, args.outpath, args.selected_pixels, args.resized_length)
        pixel.PlotGlobalHDversusBin_pixel(args.datapaths, args.outpath, args.resized_length)
        pixel.GlobalHD_pixel(args.datapaths, args.outpath, args.n_bins_pixel, args.resized_length)

    if args.PCA == 'yes':
        if args.feature_or_pixel == 'feature':
            fc.PlotFeatureDistribution_PCA(args.PCA_files, args.outpath, args.selected_components_feature, args.n_bins_feature, args.data_labels)
            fc.PlotGlobalHDversusBin_feature_PCA(args.PCA_files, args.outpath, args.explained_variance_ratio_feature)
            fc.GlobalHD_feature(args.PCA_files, args.outpath, args.n_bins_feature, args.explained_variance_ratio_feature)
        if args.feature_or_pixel == 'pixel':
            pc.PlotPixelDistribution_PCA(args.PCA_files, args.outpath, args.selected_components_pixel, args.n_bins_pixel, args.data_labels)
            pc.PlotGlobalHDversusBin_pixel_PCA(args.PCA_files, args.outpath, args.explained_variance_ratio_pixel)
            pc.GlobalHD_pixel(args.PCA_files, args.outpath, args.n_bins_pixel, args.explained_variance_ratio_pixel)
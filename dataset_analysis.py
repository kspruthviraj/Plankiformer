import argparse

from utils_analysis import feature
from utils_analysis import pixel
from utils_analysis import abundance
from utils_analysis import sampling_date as sdate


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
args = parser.parse_args()


if __name__ == '__main__':
    sdate.PlotSamplingDate(args.train_datapath, args.outpath)
    # sdate.PlotSamplingDateEachClass(args.train_datapath, args.outpath)
    abundance.PlotAbundance(args.datapaths, args.outpath, args.datapath_labels)
    abundance.PlotAbundanceSep(args.datapaths, args.outpath, args.datapath_labels)
    feature.PlotFeatureDistribution(args.datapaths, args.outpath, args.selected_features, args.n_bins_feature, args.datapath_labels)
    # feature.PlotFeatureHDversusBin(args.datapaths, args.outpath, args.selected_features)
    # feature.PlotGlobalHDversusBin_feature(args.datapaths, args.outpath)
    feature.GlobalHD_feature(args.datapaths, args.outpath, args.n_bins_feature)
    pixel.PlotPixelDistribution(args.datapaths, args.outpath, args.selected_pixels, args.n_bins_pixel, args.datapath_labels, args.resized_length)
    # pixel.PlotPixelHDversusBin(args.datapaths, args.outpath, args.selected_pixels, args.resized_length)
    # pixel.PlotGlobalHDversusBin_pixel(args.datapaths, args.outpath, args.resized_length)
    pixel.GlobalHD_pixel(args.datapaths, args.outpath, args.n_bins_pixel, args.resized_length)
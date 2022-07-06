import argparse
import copy
import glob
import json
import logging
import logging.config
import os

import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from argparse import Namespace
from astropy.modeling import fitting, models
from astropy.stats import sigma_clipped_stats
from astropy.table import QTable
from astropy.visualization import ZScaleInterval
from ccdproc import CCDData
from logging import Logger
from numpy import unique, where
from pandas import DataFrame
from pathlib import Path
from photutils import DAOStarFinder
from photutils import CircularAperture
from sklearn.cluster import MeanShift
from scipy import optimize
from typing import Union, List, Tuple
from mpl_toolkits.axes_grid1 import make_axes_locatable


plt.style.use('dark_background')

log = logging.getLogger(__name__)


def get_args(arguments: Union[list, None] = None) -> Namespace:
    parser = argparse.ArgumentParser(
        description="Get best focus value using a sequence of images with "
                    "different focus value"
    )

    parser.add_argument('--data-path',
                        action='store',
                        dest='data_path',
                        default=os.getcwd(),
                        help='Folder where data is located')

    parser.add_argument('--file-pattern',
                        action='store',
                        dest='file_pattern',
                        default='*.fits',
                        help='Pattern for filtering files.')

    parser.add_argument('--obstype',
                        action='store',
                        dest='obstype',
                        default='Focus',
                        help='Only the files whose OBSTYPE matches what you '
                             'enter here will be used. The default should '
                             'always work.')

    parser.add_argument('--brightest',
                        action='store',
                        dest='brightest',
                        default=1,
                        help='Pick N-brightest sources. Default 1.')

    # parser.add_argument('--features-model',
    #                     action='store',
    #                     dest='features_model',
    #                     choices=['gaussian', 'moffat'],
    #                     default='gaussian',
    #                     help='Model to use in fitting the features in order to'
    #                          'obtain the FWHM for each of them')

    parser.add_argument('--plot-results',
                        action='store_true',
                        dest='plot_results',
                        help='Show a plot when it finishes the focus '
                             'calculation')

    parser.add_argument('--show-mask',
                        action='store_true',
                        dest='show_mask',
                        help='Show the image and the masked areas highlighted in red.')

    parser.add_argument('--debug',
                        action='store_true',
                        dest='debug',
                        help='Activate debug mode')

    parser.add_argument('--debug-plots',
                        action='store_true',
                        dest='debug_plots',
                        help='Show debugging plots.')

    args = parser.parse_args(args=arguments)
    return args


# def get_sharpest_image(sources: DataFrame) -> Series:
#     """Finds the sharpest image by a series of criteria
#
#     This method chooses the best image based on the following criteria:
#
#     - Maximum Peak, a star in the best focused image usually has the highest peak intensity.
#     - Maximum Flux, a star in the best focused image usually has the highest flux.
#     - Minimum Magnitude, a star in the best focused image is usually brightest a lower magnitude means brightest.
#     - Roundness2 Closest to 0. According to :py:class:`photutils.detection.core.DAOSTarFinder` best focused image
#       has roundness2 closest to 0.
#
#     All these parameters are considered and the row index is obtained, then the most recurrent index is used to
#     select the row that contains the best image.
#
#     Args:
#         sources (DataFrame): The output of :py:class:`photutils.detection.core.DAOSTarFinder` converted to
#          a :py:class:`pandas.DataFrame`.
#
#     Returns:
#         The row containing the best image as a :py:class:`pandas.Series`.
#
#     """
#
#     max_peak = sources['peak'].idxmax()
#     max_flux = sources['flux'].idxmax()
#     min_mag = sources['mag'].idxmin()
#     min_roundness_2 = sources['roundness2'].abs().idxmin()
#     log.debug(sources.to_string())
#     log.debug(f"Max Peak {max_peak}")
#     log.debug(f"Max Flux {max_flux}")
#     log.debug(f"Min Mag {min_mag}")
#     log.debug(f"Min roundness2 {min_roundness_2}")
#     arg_best = [max_peak, max_flux, min_mag, min_roundness_2]
#     arg_best_set = set(arg_best)
#     if len(arg_best_set) == 1:
#         return sources.iloc[arg_best[0]]
#     elif len(arg_best_set) == len(arg_best):
#         log.warning("All values are different, Choosing max peak")
#         return sources.iloc[max_peak]
#     else:
#         log.warning("Not all values equal also not all different, choosing the most common.")
#         return sources.iloc[max(arg_best_set, key=arg_best.count)]


def setup_logging(debug: bool = False) -> Logger:
    log_format = '[%(asctime)s][%(levelname)s]: %(message)s'
    if debug:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO

    date_format = '%H:%M:%S'

    logging.basicConfig(level=log_level,
                        format=log_format,
                        datefmt=date_format)
    logger = logging.getLogger(__name__)
    return logger


def sources_to_pandas(valid_sources: List[QTable]) -> DataFrame:
    """Helper method to convert sources to pandas DataFrame

    Returns:

    """
    all_pandas_sources = []
    for source in valid_sources:
        pd_source = source.to_pandas().reset_index(drop=True)
        all_pandas_sources.append(pd_source)
    pd_sources = pd.concat(all_pandas_sources).sort_values(by='focus').reset_index(level=0)
    pd_sources['index'] = pd_sources.index
    return pd_sources


class TripleSpecFocus(object):

    def __init__(self, debug: bool = False,
                 saturation: float = 40000,
                 plot_results: bool = False,
                 debug_plots: bool = False) -> None:
        self.fig = None
        self.file_list: List = []
        self.saturation_level = saturation
        self.sources_df: DataFrame = DataFrame()
        self.i = None
        self.log = setup_logging(debug=debug)
        self.max_sources_count: int = 0
        self.minimum_sources: int = 6
        self.fitter = fitting.LevMarLSQFitter()
        self.plot_results: bool = plot_results
        self.debug_plots: bool = debug_plots

    def __call__(self,
                 data_path: Union[str, Path],
                 file_list: List = [],
                 source_fwhm: float = 5.0,
                 det_threshold: float = 5.0,
                 mask_threshold: float = 1,
                 trim_section: str = '[272:690,472:890]',
                 brightest: int = 1,
                 saturation_level: float = 40000.,
                 minimum_sources: int = 6,
                 show_mask: bool = False,
                 show_source: bool = False,
                 plot_results: bool = False,
                 debug_plots: bool = False,
                 print_all_data: bool = False) -> List[dict]:
        """Find focus for triplespec SV camera

        Finds best focus for TripleSpec Slit Viewer camera
        Illuminated Section: '[23:940,115:890]'
        Best Region of Interest: '[272:690,472:890]'
        Args:
            data_path:
            source_fwhm:
            det_threshold:
            mask_threshold:
            trim_section:
            brightest (int): Return N-brightest sources where N is the given value.
            show_mask:
            show_source:

        Returns:

        """
        self.data_path = data_path
        self.source_fwhm = source_fwhm
        self.det_threshold = det_threshold
        self.mask_threshold = mask_threshold
        self.trim_section = trim_section
        self.brightest = brightest
        self.saturation_level = saturation_level
        self.minimum_sources = minimum_sources
        self.show_mask: bool = show_mask
        self.show_source: bool = show_source
        self.max_sources_count: int = 0
        self.plot_results: bool = plot_results
        self.results = []

        self.polynomial = models.Polynomial1D(degree=5)

        self.file_list = sorted(glob.glob(os.path.join(data_path, '*.fits')))

        valid_sources = []
        for self.i in range(len(self.file_list)):
            self.log.info("Processing file: {}".format(self.file_list[self.i]))
            ccd = CCDData.read(self.file_list[self.i], unit='adu')
            sources = self.detect_sources(ccd, debug_plots=self.debug_plots)
            if sources is not None:
                valid_sources.append(sources)
                count = len(sources)
                if count > self.max_sources_count:
                    self.max_sources_count = count

        self.sources_df = sources_to_pandas(valid_sources=valid_sources)
        if self.debug_plots:
            plt.show()

        self.sources_df['distances'] = np.sqrt(self.sources_df['xcentroid'] ** 2 + self.sources_df['ycentroid'] ** 2)
        self.sources_df['angle'] = np.arctan((self.sources_df['ycentroid']/self.sources_df['xcentroid']))

        self.log.debug(f"Max Sources Per image: {self.max_sources_count}")

        self.filter_sources(plot=debug_plots)
        # self.__fit_2d_spatial_profile()

        cluster_ids = self.sources_df['cluster_id'].unique().tolist()

        all_best_focus = []

        for id in cluster_ids:
            self.log.info(f"Processing cluster {id}")
            cluster = self.sources_df[self.sources_df['cluster_id'] == id]
            best_focus = self.fit_best_focus_by_instrumental_magnitude(cluster=cluster, debug_plots=self.debug_plots)
            all_best_focus.append(best_focus)
        mean_focus, median_focus, focus_std = sigma_clipped_stats(all_best_focus, sigma=3.0)
        self.log.info(f"Best Focus: {mean_focus}")
        best_image = self.__get_best_image(best_focus=mean_focus)
        best_image_focus = self.sources_df[self.sources_df['filename'] == best_image]['focus'].unique().tolist()[0]

        self.results.append({'date': 'focus_group',
                             'time': '',
                             'notes': '',
                             'mean_focus': round(mean_focus, 10),
                             'median_focus': round(median_focus, 10),
                             'focus_std': round(focus_std, 10),
                             # 'fwhm': round(self.__best_fwhm, 10),
                             'best_image_name': best_image,
                             'best_image_focus': best_image_focus,
                             # 'best_image_fwhm': round(self.__best_image_fwhm, 10),
                             'focus_data': cluster['focus'].tolist(),
                             'mag_data': cluster['mag'].tolist()
                             })
        self.log.debug(f"Best Focus... Mean: {mean_focus}, median: {median_focus}, std: {focus_std}")

        if self.plot_results:
            scale = ZScaleInterval()
            matplotlib.rc('axes', edgecolor='red')
            for cid in cluster_ids:
                cluster = self.sources_df[self.sources_df['cluster_id'] == cid]
                file_list = cluster['filename'].tolist()
                files_count = len(file_list)
                fig, axes = plt.subplots(files_count, 3, figsize=(20, 5 * files_count))
                i = 0
                x = cluster['xcentroid'].mean()
                y = cluster['ycentroid'].mean()
                for index, r in cluster.iterrows():
                    print(r['filename'], best_image, r['filename'] == best_image)
                    if r['filename'] == best_image:
                        is_best_focus = True
                    else:
                        is_best_focus = False
                    print(f"Is best focus: {is_best_focus}")
                    ccd = CCDData.read(os.path.join(self.data_path, r['filename']), unit='adu')

                    width, height = ccd.data.shape

                    factor = 3
                    x_high = np.min([width, int(x + factor * self.source_fwhm)])
                    y_high = np.min([height, int(y + factor * self.source_fwhm)])

                    # make sure the sample has the same size
                    x_low0 = np.min([int(x - factor * self.source_fwhm), int(x_high - 2 * factor * self.source_fwhm)])
                    y_low0 = np.min([int(y - factor * self.source_fwhm), int(y_high - 2 * factor * self.source_fwhm)])

                    # Make sure the indexes are positive
                    x_low = np.max([0, x_low0])
                    y_low = np.max([0, y_low0])

                    self.log.debug(f"x: {x} y: {y} x_low: {x_low} x_high: {x_high} y_low: {y_low} y_high: {y_high}")
                    fig.suptitle(f"Best Focus: {mean_focus:.2f} {best_image}", fontsize=20)
                    # axes[i][0].set_title(ccd.header['FILENAME'])
                    if is_best_focus:
                        for e in [0, 1, 2]:
                            axes[i][e].spines['top'].set_color('#00c513')
                            axes[i][e].spines['left'].set_color('#00c513')
                            axes[i][e].spines['right'].set_color('#00c513')
                            axes[i][e].spines['bottom'].set_color('#00c513')
                    axes[i][0].plot(range(x_low, x_high), ccd.data[int(y), x_low:x_high])
                    axes[i][0].set_xlabel('X')
                    # axes[i][0].set_ylabel('Counts at row {}'.format(int(y)))
                    axes[i][1].plot(range(y_low, y_high), ccd.data[y_low:y_high, int(x)])
                    axes[i][1].set_xlabel('Y')
                    # axes[i][1].set_ylabel('Counts at column {}'.format(int(x)))
                    z1, z2 = scale.get_limits(ccd.data[y_low:y_high, x_low:x_high])

                    axes[i][2].set_xticks([])
                    axes[i][2].set_yticks([])
                    axes[i][2].imshow(ccd.data, clim=(z1, z2), origin='lower', interpolation='nearest')
                    axes[i][2].set_xlim(x_low, x_high)
                    axes[i][2].set_ylim(y_low, y_high)
                    axes[i][2].set_xlabel('X')
                    axes[i][2].set_ylabel(f"Focus: {ccd.header['TELFOCUS']}", rotation=0, labelpad=45, verticalalignment='center')
                    i += 1
                plt.tight_layout(rect=(0.04, 0.067, 1, 0.929))
                plt.show()

        if print_all_data:
            print(self.sources_df.to_string())
        return self.results

    def __sigma_clip_dataframe(self, lower: float = 2, upper: float = 1) -> Tuple[DataFrame, DataFrame]:
        """
        It takes a dataframe, and returns two dataframes, one with the sources that are within the limits of the mean and
        standard deviation of the cluster_std column, and one with the sources that are outside of those limits

        Args:
          lower (float): float = 2, upper: float = 1. Defaults to 2
          upper (float): float = 1. Defaults to 1

        Returns:
          A tuple of two dataframes.
        """
        cluster_std = self.sources_df['cluster_std'].unique().tolist()
        cluster_mean = np.average(cluster_std)
        cluster_std = np.std(cluster_std)
        lower_limit = cluster_mean - lower * cluster_std
        upper_limit = cluster_mean + upper * cluster_std
        selected_sources = self.sources_df[(self.sources_df['cluster_std'] >= lower_limit) &
                                           (self.sources_df['cluster_std'] <= upper_limit)]
        rejected_sources = self.sources_df[(self.sources_df['cluster_std'] < lower_limit) |
                                           (self.sources_df['cluster_std'] > upper_limit)]
        return selected_sources, rejected_sources

    def __fit_2d_spatial_profile(self):
        """
        It takes a list of files, and for each file, it takes a list of sources, and for each source, it fits a 2D Gaussian
        to the source
        """
        files = self.sources_df['filename'].unique().tolist()

        for file_name in files:
            # get File
            full_path = os.path.join(self.data_path, file_name)
            self.log.info(f"Fitting profile to sources in {file_name}")
            ccd = CCDData.read(full_path, unit='adu')

            sources_in_file = self.sources_df[self.sources_df['filename'] == file_name]

            for index, row in sources_in_file.iterrows():
                model = models.Gaussian2D(amplitude=row['peak'], x_mean=row['xcentroid'], y_mean=row['ycentroid'])

                x, y = ccd.data.shape
                y_axis, x_axis = np.mgrid[:x, :y]
                fitted_model = self.fitter(model=model, x=x_axis, y=y_axis, z=ccd.data)
                self.sources_df.loc[index, 'source_fwhm_x'] = fitted_model.x_stddev.value
                self.sources_df.loc[index, 'source_fwhm_y'] = fitted_model.y_stddev.value

    def __get_best_image(self, best_focus: float):
        """
        It takes the best focus value from the previous function and finds the image with the closest focus value to it

        Args:
          best_focus (float): The focus value that you want to use to find the best image.

        Returns:
          The best image is being returned.
        """
        focus_values = self.sources_df['focus'].unique().tolist()
        arg_best_focus = np.argmin([np.abs(i - best_focus) for i in focus_values])

        best_raw_focus = focus_values[arg_best_focus]

        best_image = self.sources_df[self.sources_df['focus'] == best_raw_focus]['filename'].unique().tolist()[0]
        print(f"Best Image: {best_image}")

        print(f"Best Focus Value: {focus_values[arg_best_focus]}")

        return best_image

    def filter_sources(self, bandwidth: int = 7, plot: bool = False):
        model = MeanShift(bandwidth=bandwidth)

        data = list(map(list, zip(*[self.sources_df['xcentroid'].tolist(),
                                    self.sources_df['ycentroid'].tolist()])))

        fitted_model = model.fit_predict(data)
        self.sources_df['cluster_id'] = fitted_model
        self.sources_df['cluster_std'] = -1
        self.sources_df['fwhm'] = (2.355 * self.sources_df['flux'])/(np.sqrt(2 * np.pi) * self.sources_df['peak'])
        self.sources_df['source_fwhm_x'] = -1
        self.sources_df['source_fwhm_y'] = -1

        cluster_ids = unique(fitted_model)

        centroids_x = []
        centroids_y = []
        centroids_size = []
        clusters_stdv = []
        for cluster_id in cluster_ids:
            row_ix = where(fitted_model == cluster_id)
            x_data = [data[i][0] for i in row_ix[0]]
            y_data = [data[i][1] for i in row_ix[0]]

            x_mean = np.mean(x_data)
            centroids_x.append(x_mean)

            y_mean = np.mean(y_data)
            centroids_y.append(y_mean)

            standard_deviation = np.mean(np.sqrt((x_data - x_mean) ** 2 + (y_data - y_mean) ** 2))
            self.sources_df.iloc[row_ix[0], [self.sources_df.columns.to_list().index('cluster_std')]] = standard_deviation

            centroids_size.append(standard_deviation * 100)
            self.log.debug(f"STD: {standard_deviation}")

        cluster_ids = self.sources_df['cluster_id'].unique()
        removed_clusters = []
        self.log.info(f"Found {len(cluster_ids)} unique clusters")
        for cluster_id in cluster_ids:
            cluster = self.sources_df[self.sources_df['cluster_id'] == cluster_id]
            if cluster.shape[0] < self.minimum_sources:
                indexes = self.sources_df[self.sources_df['cluster_id'] == cluster_id].index
                removed_clusters.append(cluster)
                self.sources_df.drop(indexes, inplace=True)
                # self.sources_df = self.sources_df[self.sources_df.cluster_id != cluster_id]
        self.log.info(f"Number of clusters after removing clusters with less than {self.minimum_sources} sources: {self.sources_df['cluster_id'].unique().shape[0]}")

        self.sources_df, rejected_by_clipping = self.__sigma_clip_dataframe()

        removed_clusters_df = pd.concat(removed_clusters).reset_index(level=0)

        if plot:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9))

            ccd = CCDData.read(os.path.join(self.data_path, os.path.basename(self.file_list[4])), unit='adu')
            scale = ZScaleInterval()
            z1, z2 = scale.get_limits(ccd.data)
            ax1.imshow(ccd.data, clim=(z1, z2), cmap='gray')
            cluster_ids = self.sources_df['cluster_id'].unique()
            cmap = cm.get_cmap('viridis', len(cluster_ids))
            colors = cmap(np.linspace(0, 1, len(cluster_ids)))
            centroids_x = []
            centroids_y = []
            centroids_size = []

            for i in range(len(cluster_ids)):
                cluster = self.sources_df[self.sources_df['cluster_id'] == cluster_ids[i]]
                centroids = cluster.loc[:, ['xcentroid', 'ycentroid']].mean(axis=0)
                centroids_x.append(centroids['xcentroid'])
                centroids_y.append(centroids['ycentroid'])
                centroids_size.append(cluster['cluster_std'].unique()[0] * 150)
                cluster.plot.scatter(x='xcentroid', y='ycentroid', ax=ax1, color=colors[i])

            removed_clusters_df.plot.scatter(x='xcentroid', y='ycentroid', ax=ax1, marker='x', color='r', label=f"Removed by Count < {self.minimum_sources} sources")
            rejected_by_clipping.plot.scatter(x='xcentroid', y='ycentroid', ax=ax1, marker='d', color='m', label="Removed by Clipping Dispersion")
            ax1.scatter(centroids_x, centroids_y, s=centroids_size, alpha=0.5)
            ax1.scatter([], [], label="Clusters")
            ax1.legend(loc='best')
            clusters_stdv = self.sources_df['cluster_std'].unique().tolist()
            ax2.hist(clusters_stdv, bins=self.max_sources_count)
            ax2.set_xlabel("Standard Deviation")
            ax2.set_title("Standard Deviation Distribution")

            plt.tight_layout()
            plt.show()

    def fit_best_focus_by_instrumental_magnitude(self, cluster: DataFrame, debug_plots: bool = False) -> np.float64:
        focus = cluster['focus'].tolist()
        peaks = cluster['peak'].tolist()
        mags = cluster['mag'].tolist()
        flux = cluster['flux'].tolist()
        round2 = cluster['roundness2'].tolist()
        fwhm = cluster['fwhm'].tolist()
        sharp = cluster['sharpness'].tolist()
        min_focus = np.min(focus)
        max_focus = np.max(focus)

        x_axis = np.linspace(min_focus, max_focus, 2000)

        fitted_mags = self.fitter(self.polynomial, focus, mags)

        modeled_data = fitted_mags(x_axis)

        index_of_minimum = np.argmin(modeled_data)
        middle_point = x_axis[index_of_minimum]
        self.log.debug(f"Min. Focus: {min_focus}, Middle Point: {middle_point}, Max Focus: {max_focus}")

        try:
            best_focus = optimize.brent(fitted_mags, brack=[min_focus, middle_point, max_focus])
            self.log.debug("Found best focus using Brent's optimization algorithm.")
            self.log.info(f"Best focus found: {best_focus}")
        except ValueError as error:
            self.log.error(str(error))

            best_focus = middle_point

        if debug_plots:
            fitted_peaks = self.fitter(self.polynomial, focus, peaks)
            fitted_flux = self.fitter(self.polynomial, focus, flux)
            fitted_fwhm = self.fitter(self.polynomial, focus, fwhm)
            fitted_round2 = self.fitter(self.polynomial, focus, round2)
            fitted_sharp = self.fitter(self.polynomial, focus, sharp)

            fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(16, 9))

            ax1.set_title("Peaks")
            ax1.set_xlabel("Focus")
            ax1.set_ylabel("Peaks")
            ax1.axvline(best_focus, color='r', label='Best Focus')
            ax1.plot(focus, peaks, label="data points")
            ax1.plot(x_axis, fitted_peaks(x_axis), label='Fitted Poly')
            ax1.legend(loc='best')

            ax2.set_title("Magnitudes")
            ax2.set_xlabel("Focus")
            ax2.set_ylabel("Magnitude (instr)")
            ax2.axvline(best_focus, color='r', label='Best Focus')
            ax2.plot(focus, mags, label="data points")
            ax2.plot(x_axis, fitted_mags(x_axis), label='Fitted Poly')
            ax2.legend(loc='best')

            ax3.set_title("Flux")
            ax3.set_xlabel("Focus")
            ax3.axvline(best_focus, color='r', label='Best Focus')
            ax3.set_ylabel("Flux")
            ax3.plot(focus, flux, label="data points")
            ax3.plot(x_axis, fitted_flux(x_axis), label='Fitted Poly')
            ax3.legend(loc='best')

            ax4.set_title("Roundness2")
            ax4.set_xlabel("Focus")
            ax4.set_ylabel("Roundness 2")
            ax4.axvline(best_focus, color='r', label='Best Focus')
            ax4.plot(focus, round2, label="data points")
            ax4.plot(x_axis, fitted_round2(x_axis), label='Fitted Poly')
            ax4.legend(loc='best')

            ax5.set_title("Sharpness")
            ax5.set_xlabel("Focus")
            ax5.set_ylabel("Sharpness")
            ax5.axvline(best_focus, color='r', label='Best Focus')
            ax5.plot(focus, sharp, label="data points")
            ax5.plot(x_axis, fitted_sharp(x_axis), label='Fitted Poly')
            ax5.legend(loc='best')

            ax6.set_title("FWHM")
            ax6.set_xlabel("Focus")
            ax6.set_ylabel("FWHM")
            ax6.axvline(best_focus, color='r', label='Best Focus')
            ax6.plot(focus, fwhm, label="data points")
            ax6.plot(x_axis, fitted_fwhm(x_axis), label='Fitted Poly')
            ax6.legend(loc='best')

            plt.tight_layout()
            plt.show()
        return best_focus

    def detect_sources(self, ccd: CCDData, debug_plots: bool = False) -> QTable:
        mean, median, std = sigma_clipped_stats(ccd.data, sigma=3.0)
        self.log.debug(f"Mean: {mean}, Median: {median}, Standard Dev: {std}")

        ccd.mask = ccd.data <= (median - self.mask_threshold * std)

        color_map = copy.copy(cm.gray)
        color_map.set_bad(color='red')

        self.log.debug(f"Show Mask: {self.show_mask}")
        if self.show_mask:
            fig, ax = plt.subplots(figsize=(20, 15))
            ax.set_title(f"Bad Pixel Mask\nValues {self.mask_threshold} Std below median are masked")
            ax.imshow(ccd.mask, cmap=color_map, origin='lower', interpolation='nearest')

        daofind = DAOStarFinder(fwhm=self.source_fwhm,
                                threshold=median + self.det_threshold * std,
                                exclude_border=True,
                                brightest=None,
                                peakmax=self.saturation_level)
        sources = daofind(ccd.data - median, mask=ccd.mask)

        if sources is not None:
            sources.add_column([ccd.header['TELFOCUS']], name='focus')
            sources.add_column([ccd.header['FILENAME']], name='filename')
            for col in sources.colnames:
                if col != 'filename':
                    sources[col].info.format = '%.8g'

            if debug_plots:
                positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
                apertures = CircularAperture(positions, r=self.source_fwhm)
                scale = ZScaleInterval()

                if self.show_source or self.show_mask:
                    z1, z2 = scale.get_limits(ccd.data)
                    fig, ax = plt.subplots(figsize=(20, 15))
                    ax.set_title(ccd.header['FILENAME'])

                    if self.show_mask:
                        masked_data = np.ma.masked_where(ccd.data <= (median - self.mask_threshold * std), ccd.data)
                        im = ax.imshow(masked_data, cmap=color_map, origin='lower', clim=(z1, z2),
                                       interpolation='nearest')
                    else:
                        im = ax.imshow(ccd.data, cmap=color_map, origin='lower', clim=(z1, z2),
                                       interpolation='nearest')
                    apertures.plot(color='blue', lw=1.5, alpha=0.5)

                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes('right', size="3%", pad=0.1)
                    plt.colorbar(im, cax=cax)

        else:
            self.log.critical("Unable to detect sources in file: {}".format(ccd.header['FILENAME']))
        return sources


def run_triplespec_focus(args=None):
    """
    It takes a directory of TripleSpec data, finds the brightest source, and returns the focus position

    :param args: The arguments passed to the script
    """
    args = get_args(arguments=args)

    focus = TripleSpecFocus(debug=args.debug, debug_plots=args.debug_plots)
    results = focus(data_path=args.data_path, det_threshold=6, brightest=args.brightest, show_mask=args.show_mask, show_source=True, plot_results=args.plot_results)
    log.info(json.dumps(results, indent=4))


if __name__ == '__main__':
    run_triplespec_focus()

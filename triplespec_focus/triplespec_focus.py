import argparse
import copy
import glob
import json
import logging
import logging.config
import os

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from argparse import Namespace
from astropy.modeling import fitting, models
from astropy.visualization import ZScaleInterval
from astropy.table import QTable
from ccdproc import CCDData
from ccdproc import trim_image
from astropy.stats import sigma_clipped_stats
from pandas import DataFrame, Series
from pathlib import Path
from photutils import DAOStarFinder
from photutils import CircularAperture
from scipy import optimize
from typing import Union
from mpl_toolkits.axes_grid1 import make_axes_locatable


plt.style.use('dark_background')


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

    args = parser.parse_args(args=arguments)
    return args


def get_sharpest_image(sources: DataFrame) -> Series:
    """Finds the sharpest image by a series of criteria

    This method chooses the best image based on the following criteria:

    - Maximum Peak, a star in the best focused image usually has the highest peak intensity.
    - Maximum Flux, a star in the best focused image usually has the highest flux.
    - Minimum Magnitude, a star in the best focused image is usually brightest a lower magnitude means brightest.
    - Roundness2 Closest to 0. According to :py:class:`photutils.detection.core.DAOSTarFinder` best focused image
      has roundness2 closest to 0.

    All these parameters are considered and the row index is obtained, then the most recurrent index is used to
    select the row that contains the best image.

    Args:
        sources (DataFrame): The output of :py:class:`photutils.detection.core.DAOSTarFinder` converted to
        a :py:class:`pandas.DataFrame`.

    Returns:
        The row containing the best image as a :py:class:`pandas.Series`.

    """
    max_peak = sources['peak'].idxmax()
    max_flux = sources['flux'].idxmax()
    min_mag = sources['mag'].idxmin()
    min_roundness_2 = sources['roundness2'].abs().idxmin()
    print(sources.to_string())
    print(f"Max Peak {max_peak}")
    print(f"Max Flux {max_flux}")
    print(f"Min Mag {min_mag}")
    print(f"Min roundness2 {min_roundness_2}")
    arg_best = [max_peak, max_flux, min_mag, min_roundness_2]
    arg_best_set = set(arg_best)
    if len(arg_best_set) == 1:
        return sources.iloc[arg_best[0]]
    elif len(arg_best_set) == len(arg_best):
        print("All values are different, Choosing max peak")
        return sources.iloc[max_peak]
    else:
        print("Not all values equal also not all different, choosing the most common.")
        return sources.iloc[max(arg_best_set, key=arg_best.count)]


def sources_to_pandas(valid_sources: list[QTable]) -> DataFrame:
    """Helper method to convert sources to pandas DataFrame

    Returns:

    """
    all_pandas_sources = []
    for source in valid_sources:
        pd_source = source.to_pandas()
        all_pandas_sources.append(pd_source)
    pd_sources = pd.concat(all_pandas_sources).reset_index()
    return pd_sources


class TripleSpecFocus(object):

    def __init__(self):
        self.fig = None
        self.valid_sources = None
        self.i = None

    def __call__(self,
                 data_path: Union[str, Path],
                 source_fwhm: float = 10.0,
                 det_threshold: float = 5.0,
                 mask_threshold: float = 1,
                 trim_section: str = '[23:940,115:890]',
                 brightest: int = 1,
                 show_mask: bool = False,
                 show_source: bool = False):
        """Find focus for triplespec SV camera

        Finds best focus for TripleSpec Slit Viewer camera

        Args:
            data_path:
            source_fwhm:
            det_threshold:
            mask_threshold:
            trim_section:
            brightest:
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
        self.show_mask = show_mask
        self.show_source = show_source
        self.valid_sources = []
        self.results = []

        self.polynomial = models.Polynomial1D(degree=5)
        # self.polynomial = models.Gaussian1D()
        self.fitter = fitting.LevMarLSQFitter()
        self.linear_fitter = fitting.LinearLSQFitter()

        file_list = sorted(glob.glob(os.path.join(data_path, '*.fits')))

        self.fig, self.axes = plt.subplots(len(file_list), 3, figsize=(20, 5 * len(file_list)))

        for self.i in range(len(file_list)):
            print("Processing file: {}".format(file_list[self.i]))
            ccd = CCDData.read(file_list[self.i], unit='adu')
            sources = self.detect_sources(ccd)
            if sources is not None:
                self.valid_sources.append(sources)

        pd_sources = sources_to_pandas(valid_sources=self.valid_sources)
        plt.show()

        sharpest_image = get_sharpest_image(sources=pd_sources)
        print(type(sharpest_image))
        print(sharpest_image)

        best_focus = self.fit_best_focus_by_peak_value(sources=pd_sources)
        self.results.append({'date': 'focus_group',
                             'time': '',
                             'mode_name': 'mode_name',
                             'notes': '',
                             'focus': round(best_focus, 10),
                             # 'fwhm': round(self.__best_fwhm, 10),
                             # 'best_image_name': self.__best_image,
                             # 'best_image_focus': round(self.__best_image_focus, 10),
                             # 'best_image_fwhm': round(self.__best_image_fwhm, 10),
                             'focus_data': pd_sources['focus'].tolist(),
                             'mag_data': pd_sources['mag'].tolist()
                             })
        return self.results

    def fit_best_focus_by_peak_value(self, sources: DataFrame) -> float:
        focus = sources['focus'].tolist()
        peaks = sources['peak'].tolist()
        mags = sources['mag'].tolist()
        flux = sources['flux'].tolist()
        round2 = sources['roundness2'].tolist()
        sharp = sources['sharpness'].tolist()
        min_focus = np.min(focus)
        max_focus = np.max(focus)

        x_axis = np.linspace(min_focus, max_focus, 2000)

        fitted_peaks = self.fitter(self.polynomial, focus, peaks)
        fitted_mags = self.fitter(self.polynomial, focus, mags)
        fitted_flux = self.fitter(self.polynomial, focus, flux)
        fitted_round2 = self.fitter(self.polynomial, focus, round2)
        fitted_sharp = self.fitter(self.polynomial, focus, sharp)

        modeled_data = fitted_mags(x_axis)
        index_of_minimum = np.argmin(modeled_data)
        middle_point = x_axis[index_of_minimum]
        print(min_focus, middle_point, max_focus)

        best_focus = optimize.brent(fitted_mags, brack=[min_focus, middle_point, max_focus])

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

        plt.tight_layout()
        plt.show()
        return best_focus

    def get_focus_from_sequence(self):
        """Get best focus from sequence of images

        """
        pass

    def detect_sources(self, ccd: CCDData) -> QTable:
        ccd = trim_image(ccd, fits_section=self.trim_section)
        # print(ccd.data.shape)
        ccd.write(os.path.join(self.data_path, 'trimmed', ccd.header['FILENAME']), overwrite=True)
        # show_files(ccd)
        mean, median, std = sigma_clipped_stats(ccd.data, sigma=3.0)
        print(f"Mean: {mean}, Median: {median}, Standard Dev: {std}")

        ccd.mask = ccd.data <= (median - self.mask_threshold * std)

        color_map = copy.copy(cm.gray)
        color_map.set_bad(color='red')

        scale = ZScaleInterval()
        print(f"Show Mask: {self.show_mask}")
        if self.show_mask:
            fig, ax = plt.subplots(figsize=(20, 15))
            ax.set_title(f"Bad Pixel Mask\nValues {self.mask_threshold} Std below median are masked")
            ax.imshow(ccd.mask, cmap=color_map, origin='lower', interpolation='nearest')
            # plt.show()

        daofind = DAOStarFinder(fwhm=self.source_fwhm,
                                threshold=median + self.det_threshold * std,
                                exclude_border=True,
                                brightest=self.brightest)
        sources = daofind(ccd.data - median, mask=ccd.mask)

        if sources is not None:
            sources.add_column([ccd.header['TELFOCUS']], name='focus')
            sources.add_column([ccd.header['FILENAME']], name='filename')
            for col in sources.colnames:
                if col != 'filename':
                    sources[col].info.format = '%.8g'
            #         print(sources)
            sources.pprint_all()
            if True:  # actually need boolean to trigger plots.
                positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
                print(positions)
                apertures = CircularAperture(positions, r=self.source_fwhm)

                x = positions[0][0]
                y = positions[0][1]
                x_low = int(x - 3 * self.source_fwhm)
                x_high = int(x + 3 * self.source_fwhm)
                y_low = int(y - 3 * self.source_fwhm)
                y_high = int(y + 3 * self.source_fwhm)
                self.axes[self.i][0].set_title(ccd.header['FILENAME'])
                self.axes[self.i][0].plot(range(x_low, x_high), ccd.data[int(y), x_low:x_high])
                self.axes[self.i][0].set_xlabel('X Axis')
                self.axes[self.i][0].set_ylabel('Counts at row {}'.format(int(y)))
                self.axes[self.i][1].plot(range(y_low, y_high), ccd.data[y_low:y_high, int(x)])
                self.axes[self.i][1].set_xlabel('Y Axis')
                self.axes[self.i][1].set_ylabel('Counts at column {}'.format(int(x)))
                z1, z2 = scale.get_limits(ccd.data[y_low:y_high, x_low:x_high])
                self.axes[self.i][2].imshow(ccd.data, clim=(z1, z2), origin='lower', interpolation='nearest')
                self.axes[self.i][2].set_xlim(x_low, x_high)
                self.axes[self.i][2].set_ylim(y_low, y_high)
                self.axes[self.i][2].set_xlabel('X Axis')
                self.axes[self.i][2].set_ylabel('Y Axis')

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
            print("Unable to detect sources in file: {}".format(ccd.header['FILENAME']))
        return sources


def run_triplespec_focus(args=None):
    args = get_args(arguments=args)
    LOG_FORMAT = '[%(asctime)s][%(levelname)s]: %(message)s'
    LOG_LEVEL = logging.INFO

    DATE_FORMAT = '%H:%M:%S'

    logging.basicConfig(level=LOG_LEVEL,
                        format=LOG_FORMAT,
                        datefmt=DATE_FORMAT)

    log = logging.getLogger(__name__)

    focus = TripleSpecFocus()
    results = focus(data_path=args.data_path, brightest=args.brightest, show_mask=args.show_mask)
    log.info(json.dumps(results, indent=4))


if __name__ == '__main__':
    files_path = '/home/simon/data/soar/tspec_focus/UT20201122'
    run_triplespec_focus()

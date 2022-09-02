import argparse
import copy
import os
import logging

import numpy as np
import numpy.ma as ma
import pandas as pd

from argparse import Namespace

from astropy.visualization import ZScaleInterval
from ccdproc import CCDData
from logging import Logger

from matplotlib import cm, pyplot as plt
from pandas import DataFrame
from typing import Union, List

from photutils import CircularAperture, CircularAnnulus, ApertureStats

log = logging.getLogger(__name__)


def circular_aperture_photometry(ccd: CCDData,
                                 positions: np.ndarray,
                                 aperture_radius: float = 10,
                                 filename_key: str = 'FILENAME',
                                 focus_key: str = 'TELFOCUS',
                                 plot: bool = False) -> DataFrame:
    apertures = CircularAperture(positions, r=aperture_radius)

    aper_stats = ApertureStats(ccd.data, apertures, sigma_clip=None)

    columns = ('id', 'mean', 'fwhm', 'max', 'xcentroid', 'ycentroid')
    as_table = aper_stats.to_table(columns=columns)
    as_table['focus'] = ccd.header[focus_key]
    as_table['filename'] = ccd.header[filename_key]
    for col in as_table.colnames:
        if col not in ['filename']:
            as_table[col].info.format = '%.8g'

    if plot:
        title = f"Photometry of {ccd.header['FILENAME']}"
        plot_sources_and_masked_data(ccd=ccd, positions=positions, title=title)

    return as_table.to_pandas()


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


def get_best_image_by_peak(file_list: List, saturation_level: float = 40000., focus_key: str = 'TELFOCUS') -> List:

    data = []
    for f in file_list:
        ccd = CCDData.read(f, unit='adu')
        ccd.mask = ccd.data >= saturation_level
        masked_data = ma.masked_array(ccd.data, mask=ccd.mask)
        np_max = ma.MaskedArray.max(masked_data)
        log.debug(f"File: {os.path.basename(f)} Max: {np_max} Focus: {ccd.header[focus_key]}")
        data.append([f, np_max, ccd.header[focus_key]])

    df = pd.DataFrame(data, columns=['file', 'peak', 'focus'])
    best_image = df.iloc[df['peak'].idxmax()]
    log.info(f"Best Image: {best_image.file} Peak: {best_image.peak} Focus: {best_image.focus}")

    return best_image.to_list()


def plot_sources_and_masked_data(ccd: CCDData, positions: np.ndarray, title: str = '', mask=None, aperture_radius: int = 10):
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_title(title)

    color_map = copy.copy(cm.gray)
    color_map.set_bad(color='green')

    scale = ZScaleInterval()
    z1, z2 = scale.get_limits(ccd.data)

    apertures = CircularAperture(positions, r=aperture_radius)
    annulus_apertures = CircularAnnulus(positions, r_in=aperture_radius + 5, r_out=aperture_radius + 10)

    ax.imshow(ccd.data, cmap=color_map, clim=(z1, z2))
    apertures.plot(color='red')
    annulus_apertures.plot(color='yellow')

    plt.tight_layout()
    plt.show()


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
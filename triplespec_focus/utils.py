import copy
import os
import logging

import numpy as np
import numpy.ma as ma

from argparse import ArgumentParser, Namespace

from astropy.visualization import ZScaleInterval
from ccdproc import CCDData
from logging import Logger

from matplotlib import cm, pyplot as plt
from pandas import DataFrame
from typing import Union, List

from photutils.aperture import CircularAperture, CircularAnnulus, ApertureStats

log = logging.getLogger(__name__)


def circular_aperture_statistics(ccd: CCDData,
                                 positions: np.ndarray,
                                 aperture_radius: float = 10,
                                 filename_key: str = 'FILENAME',
                                 focus_key: str = 'TELFOCUS',
                                 plot: bool = False) -> DataFrame:
    """Obtain aperture statistics from a FITS file with point sources

    Uses CircularAperture from photutils to obtain several data but the most important is the FWHM

    Args:
        ccd (CCDData): An image with point sources.
        positions (numpy.ndarray): Coordinates of the sources to be measured.
        aperture_radius (float): Aperture size for obtaining measurements. Default 10.
        filename_key (str): FITS keyword name for obtaining the file name from the FITS file. Default FILENAME.
        focus_key (str): FITS keyword name for obtaining the focus value from the FITS file. Default TELFOCUS.
        plot (bool): If set to True will display a plot of the image and the measured sources.

    Returns:
        A DataFrame containing the following columns: id, mean, fwhm, max, xcentroid, ycentroid and filename.

    """
    apertures = CircularAperture(positions, r=aperture_radius)

    aper_stats = ApertureStats(ccd.data, apertures, sigma_clip=None)

    columns = ('id', 'mean', 'fwhm', 'max', 'xcentroid', 'ycentroid')
    as_table = aper_stats.to_table(columns=columns)
    as_table['focus'] = ccd.header[focus_key]
    as_table['filename'] = ccd.header[filename_key]
    for col in as_table.colnames:
        if col not in ['filename']:
            as_table[col].info.format = '%.8g'

    if plot:  # pragma: no cover
        title = f"Photometry of {ccd.header['FILENAME']}"
        plot_sources_and_masked_data(ccd=ccd, positions=positions, title=title)

    return as_table.to_pandas()


def get_args(arguments: Union[List, None] = None) -> Namespace:
    """Helper function to get the console arguments using argparse.

    Args:
        arguments (List, None): Optional list of arguments. Default None.

    Returns:
        An instance of arparse's Namespace.

    """
    parser = ArgumentParser(
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

    parser.add_argument('--focus-key',
                        action='store',
                        dest='focus_key',
                        default='TELFOCUS',
                        help='FITS header keyword to find the focus value.')

    parser.add_argument('--filename-key',
                        action='store',
                        dest='filename_key',
                        default='FILENAME',
                        help='FITS header keyword to find the current file name.')

    parser.add_argument('--brightest',
                        action='store',
                        dest='brightest',
                        type=int,
                        default=5,
                        help='Pick N-brightest sources to perform statistics. Default 5.')

    parser.add_argument('--saturation',
                        action='store',
                        dest='saturation',
                        type=int,
                        default=40000,
                        help='Saturation value for data')

    parser.add_argument('--source-fwhm',
                        action='store',
                        dest='source_fwhm',
                        default=7.0,
                        help='FWHM for source detection.')

    parser.add_argument('--detection-threshold',
                        action='store',
                        dest='detection_threshold',
                        default=6,
                        help='Number of standard deviation above median for source detection.')

    parser.add_argument('--mask-threshold',
                        action='store',
                        dest='mask_threshold',
                        default=1,
                        help='Number of standard deviation below median to mask values.')

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
    """Select the best image by its peak value

    The best peak must be the highest below the saturation level, therefore the data is first masked and then sorted
    by the peak.

    Args:
        file_list (List): The list of images to be used.
        saturation_level (float): Data level at which the image saturates. Default 40000.
        focus_key (str): FITS keyword name for obtaining the focus value from the FITS file. Default TELFOCUS.

    Returns:
        A list wit the best image according to the criteria described above.

    """
    data = []
    for f in file_list:
        ccd = CCDData.read(f, unit='adu')
        ccd.mask = ccd.data >= saturation_level
        masked_data = ma.masked_array(ccd.data, mask=ccd.mask)
        np_max = ma.MaskedArray.max(masked_data)
        if not ma.is_masked(np_max):
            log.debug(f"File: {os.path.basename(f)} Max: {np_max} Focus: {ccd.header[focus_key]}")
            data.append([f, np_max, ccd.header[focus_key]])
        else:
            log.debug(f"Rejected masked value {np_max} from file {f}")

    df = DataFrame(data, columns=['file', 'peak', 'focus'])
    best_image = df.iloc[df['peak'].idxmax()]
    log.info(f"Best Image: {best_image.file} Peak: {best_image.peak} Focus: {best_image.focus}")

    return best_image.to_list()


def plot_sources_and_masked_data(ccd: CCDData, positions: np.ndarray, title: str = '', aperture_radius: int = 10):  # pragma: no cover
    """Helper function to plot data and sources

    Args:
        ccd (CCDData): The image to be shown.
        positions (numpy.ndarray): Array of source positions to be plotted over the image.
        title (str): Title to put to the plot. Default '' (empty string).
        aperture_radius (int): Radius size in pixels for the drawing of the sources. Default 10.

    """
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


def setup_logging(debug: bool = False, enable_astropy_logger: bool = False) -> Logger:
    """Helper function to setup the logger.

    Args:
        debug (bool): If set to True will create a logger in debug level. Default False.
        enable_astropy_logger (bool): If set to True will allow the astropy logger to remain active. Default False.

    Returns:
        A Logger instance from python's logging.

    """
    if debug:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO

    logger = logging.getLogger(__name__)
    logger.setLevel(level=log_level)

    console = logging.StreamHandler()
    console.setLevel(log_level)

    log_format = '[%(asctime)s][%(levelname)s]: %(message)s'
    formatter = logging.Formatter(log_format)

    console.setFormatter(formatter)

    logger.addHandler(console)

    astropy_logger = logging.getLogger('astropy')
    if enable_astropy_logger or debug:
        for handler in astropy_logger.handlers:
            astropy_logger.removeHandler(handler)
        astropy_logger.addHandler(console)
    else:
        astropy_logger.disabled = True

    return logger

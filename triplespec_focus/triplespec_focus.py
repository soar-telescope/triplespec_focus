import copy
import os
import sys

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

from astropy.modeling import Model
from astropy.modeling import fitting, models
from astropy.stats import sigma_clipped_stats
from astropy.table import QTable
from astropy.visualization import ZScaleInterval
from ccdproc import CCDData

from pandas import DataFrame
from pandas import concat
from pathlib import Path
from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAperture

from scipy import optimize
from typing import List, Union
from mpl_toolkits.axes_grid1 import make_axes_locatable


from .utils import (circular_aperture_statistics,
                    get_best_image_by_peak,
                    plot_sources_and_masked_data,
                    setup_logging)

plt.style.use('dark_background')

# log = logging.getLogger(__name__)


class TripleSpecFocus(object):

    def __init__(self,
                 debug: bool = False,
                 date_key: str = 'DATE',
                 date_time_key: str = 'DATE-OBS',
                 focus_key: str = 'TELFOCUS',
                 filename_key: str = 'FILENAME',
                 file_pattern: str = '*.fits',
                 n_brightest: int = 5,
                 saturation: float = 40000,
                 plot_results: bool = False,
                 debug_plots: bool = False) -> None:
        """Focus Calculator for the TripleSpec's Slit Viewer Camera

        Args:
            debug (bool): If set to True will set the logger to debug level.
            date_key (str): FITS keyword name for obtaining the date from the FITS file. Default DATE.
            date_time_key (str): FITS keyword name for obtaining the date and time from the FITS file. Default DATE-OBS.
            focus_key (str): FITS keyword name for obtaining the focus value from the FITS file. Default TELFOCUS.
            filename_key (str): FITS keyword name for obtaining the file name from the FITS file. Default FILENAME.
            file_pattern (str): Pattern for searching files in the provided data path. Default *.fits.
            n_brightest (int): Number of the brightest sources to use for measuring source statistics. Default 5.
            saturation (float): Data value at which the detector saturates. Default 40000.
            plot_results (bool): If set to True will display information plots at the end. Default False.
            debug_plots (bool): If set to True will display several plots useful for debugging or viewing the process.
              Default False.
        """

        self.best_fwhm = None
        self.best_focus = None
        self.fitted_model = None
        self.best_image_overall = None
        self.date_key: str = date_key
        self.date_time_key: str = date_time_key
        self.focus_key: str = focus_key
        self.filename_key: str = filename_key
        self.file_pattern: str = file_pattern
        self.saturation_level = saturation
        self.debug: bool = debug
        self.debug_plots: bool = debug_plots
        self.plot_results: bool = plot_results
        self.mask_threshold: float = 1
        self.source_fwhm: float = 7.0
        self.det_threshold: float = 5.0
        self.show_mask: bool = False

        self.masked_data = None
        self.file_list: List
        self.sources_df: DataFrame
        self.log = setup_logging(debug=self.debug)

        self.n_brightest: int = n_brightest
        self.model: Model = models.Chebyshev1D(degree=6)
        self.fitter = fitting.LinearLSQFitter()
        self.scale = ZScaleInterval()
        self.color_map = copy.copy(cm.gray)
        self.color_map.set_bad(color='red')

    def __call__(self,
                 data_path: Union[Path, str, None] = None,
                 file_list: Union[List, None] = None,
                 source_fwhm: float = 7.0,
                 det_threshold: float = 5.0,
                 mask_threshold: float = 1,
                 n_brightest: int = 5,
                 saturation_level: float = 40000.,
                 show_mask: bool = False,
                 plot_results: bool = False,
                 debug_plots: bool = False,
                 print_all_data: bool = False) -> dict:
        """Runs the focus calculation routine

        This method contains all the logic to obtain the best focus, additionally you can parse the following parameters

        Args:
            data_path (Path, str, None): Optional data path to obtain files according to file_pattern. Default None.
            file_list (List, None): Optional file list with files to be used to obtain best focus. Default None.
            source_fwhm (float): Full width at half maximum to use for source detection and statistics.
            det_threshold (float): Number of standard deviation above median to use as detection threshold. Default 5.0.
            mask_threshold (float): Number of standard deviation below median to use as a threshold for masking values.
              Default 1.
            n_brightest (int): Number of the brightest sources to use for measuring source statistics. Default 5.
            saturation_level (float): Data value at which the detector saturates. Default 40000.
            show_mask (bool): If set to True will display masked values in red when debug_plots is also True:
              Default False.
            plot_results (bool): If set to True will display information plots at the end. Default False.
            debug_plots (bool): If set to True will display several plots useful for debugging or viewing the process.
              Default False.
            print_all_data (bool): If set to True will print the entire dataset at the end.

        Returns:
            A dictionary containing information regarding the current process.
        """
        if file_list:
            self.log.debug(f"Using provided file list containing {len(file_list)} files.")
            self.file_list = file_list
        elif data_path:
            self.data_path: Path = Path(data_path)
            self.log.debug(f"File list not provided, creating file list from path: {data_path}")
            self.file_list = sorted(self.data_path.glob(pattern=self.file_pattern))
            self.log.info(f"Found {len(self.file_list)} files at {self.data_path}")
        else:
            self.log.critical("You must provide at least a data_path or a file_list value")
            sys.exit(0)

        self.source_fwhm: float = source_fwhm
        self.det_threshold: float = det_threshold
        self.mask_threshold: float = mask_threshold
        self.saturation_level: float = saturation_level
        self.show_mask: bool = show_mask
        self.n_brightest: int = n_brightest
        self.plot_results: bool = plot_results
        self.results = []

        best_image, peak, focus = get_best_image_by_peak(file_list=self.file_list,
                                                         saturation_level=self.saturation_level,
                                                         focus_key=self.focus_key)
        self.best_image_overall = best_image

        self.log.info(f"Processing best file: {best_image}, selected by highest peak below saturation")
        best_image_ccd = CCDData.read(best_image, unit='adu')

        sources = self.detect_sources(ccd=best_image_ccd, debug_plots=self.debug_plots)
        sources_positions = np.transpose((sources['xcentroid'], sources['ycentroid']))

        aperture_stats = circular_aperture_statistics(ccd=best_image_ccd,
                                                      positions=sources_positions,
                                                      aperture_radius=self.source_fwhm,
                                                      filename_key=self.filename_key,
                                                      focus_key=self.focus_key,
                                                      plot=self.debug_plots)

        brightest = aperture_stats.nlargest(self.n_brightest, 'max')

        if self.debug_plots:  # pragma: no cover
            title = f"{self.n_brightest} Brightest Sources"
            positions = np.transpose((brightest['xcentroid'].tolist(), brightest['ycentroid'].tolist()))
            plot_sources_and_masked_data(ccd=best_image_ccd, positions=positions, title=title)

        positions = np.transpose((brightest['xcentroid'].tolist(), brightest['ycentroid'].tolist()))

        all_photometry = []
        self.log.info("Starting photometry of all images using selected stars")
        for _file in self.file_list:
            self.log.info(f"Processing file: {_file}")
            ccd = CCDData.read(_file, unit='adu')
            photometry = circular_aperture_statistics(ccd=ccd,
                                                      positions=positions,
                                                      aperture_radius=self.source_fwhm,
                                                      filename_key=self.filename_key,
                                                      focus_key=self.focus_key,
                                                      plot=self.debug_plots)
            if photometry is not None:
                all_photometry.append(photometry)

        self.sources_df = concat(all_photometry).sort_values(by='focus').reset_index(level=0)
        self.sources_df['index'] = self.sources_df.index

        if self.debug_plots:   # pragma: no cover
            plt.show()

        star_ids = self.sources_df.id.unique().tolist()

        all_stars_photometry = []
        all_focus = []
        all_fwhm = []
        for star_id in star_ids:
            star_phot = self.sources_df[self.sources_df['id'] == star_id]
            interpolated_data = self.get_best_focus(df=star_phot)
            if interpolated_data:
                all_stars_photometry.append([star_phot, interpolated_data, self.best_focus])
                if self.best_focus and self.best_fwhm:
                    all_focus.append(self.best_focus)
                    all_fwhm.append(self.best_fwhm)

        mean_focus = np.mean(all_focus)
        median_focus = np.median(all_focus)
        focus_std = np.std(all_focus)
        mean_fwhm = np.mean(all_fwhm)

        best_image_overall = CCDData.read(self.best_image_overall, unit='adu')
        self.best_image_fwhm = self.sources_df[self.sources_df['filename'] == os.path.basename(
            self.best_image_overall)]['fwhm'].mean()

        focus_data = []
        fwhm_data = []
        images = self.sources_df.filename.unique().tolist()
        for image in images:
            summary_df = self.sources_df[self.sources_df['filename'] == image]
            focus = summary_df.focus.unique().tolist()
            fwhm = summary_df.fwhm.tolist()
            if len(focus) == 1:
                focus_data.append(focus[0])
                fwhm_data.append(round(np.mean(fwhm), 10))

        self.results = {'date': best_image_overall.header[self.date_key],
                        'time': best_image_overall.header[self.date_time_key],
                        'mean_focus': round(mean_focus, 10),
                        'median_focus': round(median_focus, 10),
                        'focus_std': round(focus_std, 10),
                        'fwhm': round(mean_fwhm, 10),
                        'best_image_name': os.path.basename(self.best_image_overall),
                        'best_image_focus': best_image_overall.header[self.focus_key],
                        'best_image_fwhm': round(self.best_image_fwhm, 10),
                        'focus_data': focus_data,
                        'fwhm_data': fwhm_data
                        }
        self.log.debug(f"Best Focus... Mean: {mean_focus}, median: {median_focus}, std: {focus_std}")

        if self.plot_results:   # pragma: no cover
            fig, (ax1, ax2) = plt.subplots(1, 2)

            apertures = CircularAperture(positions=positions, r=1.5 * self.source_fwhm)

            z1, z2 = self.scale.get_limits(best_image_ccd.data)
            ax1.set_title(f"Best Image: {os.path.basename(best_image)}\nFocus: {best_image_ccd.header[self.focus_key]}")
            ax1.imshow(best_image_ccd, cmap=self.color_map, clim=(z1, z2), interpolation='nearest', origin='lower')
            apertures.plot(axes=ax1, color='lawngreen')

            best_image_df = self.sources_df[self.sources_df['filename'] == os.path.basename(best_image)]
            text_offset = 1.3 * self.source_fwhm
            for index, row in best_image_df.iterrows():
                ax1.text(row['xcentroid'] - 2 * text_offset, row['ycentroid'] + text_offset, row['id'],
                         color='lawngreen', fontsize='large', fontweight='bold')
            for idx, (star_phot, interpolated_data, current_focus) in enumerate(all_stars_photometry):
                star_id = star_phot.id.unique().tolist()[0]
                ax2.plot(star_phot['focus'].tolist(), star_phot['fwhm'].tolist(), color=f"C{idx}",
                         label=f"Star ID: {star_id}", linestyle=':', alpha=0.7)
                ax2.plot(interpolated_data[0], interpolated_data[1], color=f"C{idx}", alpha=0.8)
                ax2.axvline(current_focus, color=f"C{idx}", alpha=0.8, linestyle='--')
                ax2.set_xlabel("Focus Value")
                ax2.set_ylabel('FWHM')
            ax2.axvline(mean_focus, color="lawngreen", label='Best Focus')
            ax2.set_title(f"Best Focus: {mean_focus}")
            ax2.legend(loc='best')
            plt.tight_layout()
            plt.show()
        if print_all_data:  # pragma: no cover
            print(self.sources_df.to_string())
        return self.results

    def detect_sources(self, ccd: CCDData, debug_plots: bool = False) -> QTable:
        """Detects sources in the sharpest image

        Using DAOStarFinder will detect the stellar sources in it.

        Args:
            ccd (CCDData): An image with point sources.
            debug_plots (bool): If set to True will display the image with the sources.
              Default False.

        Returns:
            An Astropy's QTable containing ids, centroids, focus value and image name.

        """
        mean, median, std = sigma_clipped_stats(ccd.data, sigma=3.0)
        self.log.debug(f"Mean: {mean}, Median: {median}, Standard Dev: {std}")

        ccd.mask = ccd.data <= (median - self.mask_threshold * std)
        self.masked_data = np.ma.masked_where(ccd.data <= (median - self.mask_threshold * std), ccd.data)

        self.log.debug(f"Show Mask: {self.show_mask}")
        if self.show_mask:  # pragma: no cover
            fig, ax = plt.subplots(figsize=(20, 15))
            ax.set_title(f"Bad Pixel Mask\nValues {self.mask_threshold} Std below median are masked")
            ax.imshow(self.masked_data, cmap=self.color_map, origin='lower', interpolation='nearest')

        daofind = DAOStarFinder(fwhm=self.source_fwhm,
                                threshold=median + self.det_threshold * std,
                                exclude_border=True,
                                brightest=None,
                                peakmax=self.saturation_level)
        sources = daofind(ccd.data - median, mask=ccd.mask)

        if sources is not None:
            sources.add_column([ccd.header[self.focus_key]], name='focus')
            sources.add_column([ccd.header[self.filename_key]], name='filename')
            for col in sources.colnames:
                if col != 'filename':
                    sources[col].info.format = '%.8g'

            if debug_plots:  # pragma: no cover
                positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
                apertures = CircularAperture(positions, r=self.source_fwhm)

                z1, z2 = self.scale.get_limits(ccd.data)
                fig, ax = plt.subplots(figsize=(20, 15))
                ax.set_title(ccd.header[self.filename_key])

                if self.show_mask:
                    masked_data = np.ma.masked_where(ccd.data <= (median - self.mask_threshold * std), ccd.data)
                    im = ax.imshow(masked_data, cmap=self.color_map, origin='lower', clim=(z1, z2),
                                   interpolation='nearest')
                else:
                    im = ax.imshow(ccd.data, cmap=self.color_map, origin='lower', clim=(z1, z2),
                                   interpolation='nearest')
                apertures.plot(color='blue', lw=1.5, alpha=0.5)

                divider = make_axes_locatable(ax)
                cax = divider.append_axes('right', size="3%", pad=0.1)
                plt.colorbar(im, cax=cax)

        else:
            self.log.critical(f"Unable to detect sources in file: {ccd.header[self.filename_key]}")
        return sources

    def get_best_focus(self, df: DataFrame, x_axis_size: int = 2000) -> List[np.ndarray]:
        """Obtains the best focus for a single source

        Args:
            df (DataFrame): Pandas DataFrame containing at least a 'focus' and a 'fwhm' column.
              The data should belong to a single source.
            x_axis_size (int): Size of the x-axis used to sample the fitted model. Is not an interpolation size.

        Returns:
            A list with the x-axis and the sampled data using the fitted model, None if it is not possible to find the
              focus.

        """
        focus_start = df['focus'].min()
        focus_end = df['focus'].max()

        x_axis = np.linspace(start=focus_start, stop=focus_end, num=x_axis_size)

        self.fitted_model = self.fitter(self.model, df['focus'].tolist(), df['fwhm'].tolist())
        modeled_data = self.fitted_model(x_axis)
        index_of_minimum = np.argmin(modeled_data)
        middle_point = x_axis[index_of_minimum]
        if middle_point == focus_start or middle_point == focus_end:
            self.log.warning("The focus vs FWHM curve does not seem to have a V or U shape. Trying by forcing the "
                             "mean focus as the middle point for Brent's optimization bracket definition.")
            middle_point = (focus_start + focus_end) / 2.

        self.log.debug(f"Brent optimization bracket, Start (xa): {focus_start} Middle (xb): {middle_point} End (xc): {focus_end}")

        try:
            self.best_focus = optimize.brent(self.fitted_model, brack=(focus_start, middle_point, focus_end))
            self.best_fwhm = modeled_data[index_of_minimum]
            self.log.info(f"Found best focus at {self.best_focus} with a fwhm of {self.best_fwhm}")
            return [x_axis, modeled_data]
        except ValueError as error:
            self.log.error(error)


if __name__ == '__main__':
    pass

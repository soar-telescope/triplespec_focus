import copy
import logging.config
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
from photutils import DAOStarFinder
from photutils import CircularAperture

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
                 focus_key: str = 'TELFOCUS',
                 filename_key: str = 'FILENAME',
                 n_brightest: int = 5,
                 saturation: float = 40000,
                 plot_results: bool = False,
                 debug_plots: bool = False) -> None:

        self.best_focus = None
        self.fitted_model = None
        self.focus_key: str = focus_key
        self.filename_key: str = filename_key
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
            show_mask:

        Returns:

        """
        if file_list:
            self.log.debug(f"Using provided file list containing {len(file_list)} files.")
            self.file_list = file_list
        elif data_path:
            self.data_path: Path = Path(data_path)
            self.log.debug(f"File list not provided, creating file list from path: {data_path}")
            self.file_list = sorted(self.data_path.glob(pattern='*fits'))
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
        for star_id in star_ids:
            star_phot = self.sources_df[self.sources_df['id'] == star_id]
            interpolated_data = self.get_best_focus(df=star_phot)
            all_stars_photometry.append([star_phot, interpolated_data, self.best_focus])
            all_focus.append(self.best_focus)

        mean_focus = np.mean(all_focus)
        median_focus = np.median(all_focus)
        focus_std = np.std(all_focus)

        self.results.append({'date': 'focus_group',
                             'time': '',
                             'notes': '',
                             'mean_focus': round(mean_focus, 10),
                             'median_focus': round(median_focus, 10),
                             'focus_std': round(focus_std, 10),
                             # 'fwhm': round(self.__best_fwhm, 10),
                             'best_image_name': os.path.basename(best_image),
                             'best_image_focus': best_image_ccd.header[self.focus_key],
                             # 'best_image_fwhm': round(self.__best_image_fwhm, 10),
                             # 'focus_data': cluster['focus'].tolist(),
                             # 'mag_data': cluster['mag'].tolist()
                             })
        self.log.debug(f"Best Focus... Mean: {mean_focus}, median: {median_focus}, std: {focus_std}")

        if self.plot_results:   # pragma: no cover
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

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
            plt.show()
        if print_all_data:  # pragma: no cover
            print(self.sources_df.to_string())
        return self.results

    def detect_sources(self, ccd: CCDData, debug_plots: bool = False) -> QTable:
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
        focus_start = df['focus'].min()
        focus_end = df['focus'].max()

        x_axis = np.linspace(start=focus_start, stop=focus_end, num=x_axis_size)

        self.fitted_model = self.fitter(self.model, df['focus'].tolist(), df['fwhm'].tolist())
        modeled_data = self.fitted_model(x_axis)
        index_of_minimum = np.argmin(modeled_data)
        middle_point = x_axis[index_of_minimum]

        self.best_focus = optimize.brent(self.fitted_model, brack=(focus_start, middle_point, focus_end))

        return [x_axis, modeled_data]


if __name__ == '__main__':
    pass

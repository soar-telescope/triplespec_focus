import copy
import glob
import os

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from astropy.visualization import SqrtStretch, ZScaleInterval, MinMaxInterval
from ccdproc import ccdmask
from ccdproc import CCDData
from ccdproc import trim_image
from astropy.stats import sigma_clipped_stats
from pandas import DataFrame
from photutils import DAOStarFinder
from photutils import CircularAperture
from mpl_toolkits.axes_grid1 import make_axes_locatable


plt.style.use('dark_background')


class TripleSpecFocus(object):

    def __init__(self):
        self.fig = None
        self.valid_sources = None
        self.i = None

    def __call__(self,
                 data_path: str,
                 source_fwhm: float = 10.0,
                 det_threshold: float = 5.0,
                 mask_threshold: float = 1,
                 trim_section: str = '[23:940,115:890]',
                 brightest: int = 1,
                 show_mask: bool = False):
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

        Returns:

        """
        self.data_path = data_path
        self.source_fwhm = source_fwhm
        self.det_threshold = det_threshold
        self.mask_threshold = mask_threshold
        self.trim_section = trim_section
        self.brightest = brightest
        self.show_mask = show_mask
        self.valid_sources = []

        file_list = sorted(glob.glob(os.path.join(data_path, '*.fits')))

        self.fig, self.axes = plt.subplots(len(file_list), 3, figsize=(20, 5 * len(file_list)))

        for self.i in range(len(file_list)):
            print("Processing file: {}".format(file_list[self.i]))
            ccd = CCDData.read(file_list[self.i], unit='adu')
            #     print(ccd.data.shape)
            sources = self.detect_sources(ccd)
            if sources is not None:
                self.valid_sources.append(sources)

        pd_sources = self._sources_to_pandas()
        print(pd_sources)
        plt.show()

    def _sources_to_pandas(self) -> DataFrame:
        """Helper method to convert sources to pandas DataFrame

        Returns:

        """
        all_pandas_sources = []
        for source in self.valid_sources:
            pd_source = source.to_pandas()
            all_pandas_sources.append(pd_source)
        pd_sources = pd.concat(all_pandas_sources)
        return pd_sources

    def detect_sources(self, ccd:  CCDData):
        ccd = trim_image(ccd, fits_section=self.trim_section)
        #     print(ccd.data.shape)
        ccd.write(os.path.join(self.data_path, 'trimmed', ccd.header['FILENAME']), overwrite=True)
        #     show_files(ccd)
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

            positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
            print(positions)

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

            z1, z2 = scale.get_limits(ccd.data)
            apertures = CircularAperture(positions, r=self.source_fwhm)
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


if __name__ == '__main__':
    files_path = '/home/simon/data/soar/tspec_focus/UT20201122'
    focus = TripleSpecFocus()
    focus(data_path=files_path, brightest=1, show_mask=True)

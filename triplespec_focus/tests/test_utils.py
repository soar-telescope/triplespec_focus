import os

import logging

from argparse import Namespace
from unittest import TestCase

import numpy as np
from astropy.modeling import models
from astropy.nddata import CCDData
from astropy.io import fits
from pandas import DataFrame

from ..utils import (circular_aperture_statistics,
                     get_args,
                     get_best_image_by_peak,
                     plot_sources_and_masked_data,
                     setup_logging)


class TestGetArgs(TestCase):

    def test_default_args_values(self):
        args = get_args(arguments=[])

        self.assertIsInstance(args, Namespace)
        self.assertEqual(args.data_path, os.getcwd())
        self.assertEqual(args.file_pattern, '*.fits')
        self.assertEqual(args.focus_key, 'TELFOCUS')
        self.assertEqual(args.filename_key, 'FILENAME')
        self.assertEqual(args.brightest, 5)
        self.assertEqual(args.saturation, 40000)
        self.assertEqual(args.source_fwhm, 7.0)
        self.assertEqual(args.detection_threshold, 6)
        self.assertEqual(args.mask_threshold, 1)
        self.assertFalse(args.plot_results)
        self.assertFalse(args.show_mask)
        self.assertFalse(args.debug)
        self.assertFalse(args.debug_plots)

    def test_modify_all_arguments(self):

        data_path = '/home/user/'
        file_pattern = '*.jpg'
        focus_key = 'FOCUS'
        filename_key = 'FILE'
        brightest = '10'
        saturation = '10000'
        source_fwhm = '40'
        detection_threshold = '3'
        mask_threshold = '3'

        new_args = ['--data-path', data_path,
                    '--file-pattern', file_pattern,
                    '--focus-key', focus_key,
                    '--filename-key', filename_key,
                    '--brightest', brightest,
                    '--saturation', saturation,
                    '--source-fwhm', source_fwhm,
                    '--detection-threshold', detection_threshold,
                    '--mask-threshold', mask_threshold,
                    '--plot-results',
                    '--show-mask',
                    '--debug',
                    '--debug-plots']

        args = get_args(arguments=new_args)
        self.assertIsInstance(args, Namespace)
        self.assertEqual(args.data_path, data_path)
        self.assertEqual(args.file_pattern, file_pattern)
        self.assertEqual(args.focus_key, focus_key)
        self.assertEqual(args.filename_key, filename_key)
        self.assertEqual(args.brightest, int(brightest))
        self.assertEqual(args.saturation, int(saturation))
        self.assertEqual(args.source_fwhm, source_fwhm)
        self.assertEqual(args.detection_threshold, detection_threshold)
        self.assertEqual(args.mask_threshold, mask_threshold)
        self.assertTrue(args.plot_results)
        self.assertTrue(args.show_mask)
        self.assertTrue(args.debug)
        self.assertTrue(args.debug_plots)


class TestLoggingSettings(TestCase):

    def test_debug_logging_setup(self):
        logger = setup_logging(debug=True)
        astropy_logger = logging.getLogger('astropy')
        self.assertEqual(logger.level, logging.DEBUG)
        self.assertTrue(astropy_logger.disabled)

    def test_enable_astropy_logger(self):
        logger = setup_logging(enable_astropy_logger=True)
        self.assertEqual(logger.level, logging.INFO)

    def test_default_logging_setup(self):
        logger = setup_logging()
        self.assertEqual(logger.level, logging.INFO)


class TestCircularApertureStatistics(TestCase):

    def setUp(self) -> None:
        self.positions = np.array(((100, 100), (300, 300)))
        self.aperture_radius = 10
        self.filename_key = 'FILE'
        self.focus_key = 'FOCUS'
        self.plot = False

        data_shape = (400, 400)
        noise = np.random.random_sample(size=data_shape)
        bias_level = 300.
        ones = np.ones(shape=data_shape)

        stars = None
        for x_mean, y_mean in self.positions:
            star_profile = models.Gaussian2D(amplitude=20000,
                                             x_mean=x_mean,
                                             y_mean=y_mean,
                                             x_stddev=self.aperture_radius / 2.,
                                             y_stddev=self.aperture_radius / 2.)
            if not stars:
                stars = star_profile
            else:
                stars += star_profile

        data = ones * bias_level + ones * stars.render(ones) + noise

        self.ccd = CCDData(data=data,
                           meta=fits.Header(),
                           unit='adu')
        self.ccd.header.set(self.filename_key,
                            value='some_file_name.fits',
                            comment='File name')
        self.ccd.header.set(self.focus_key,
                            value=1000,
                            comment='Focus value')

    def test_output(self):

        result = circular_aperture_statistics(
            ccd=self.ccd,
            positions=self.positions,
            aperture_radius=self.aperture_radius,
            filename_key=self.filename_key,
            focus_key=self.focus_key,
            plot=self.plot)

        self.assertIsInstance(result, DataFrame)
        self.assertTrue(all(column in result.columns.tolist() for column in ['id',
                                                                             'mean',
                                                                             'fwhm',
                                                                             'max',
                                                                             'xcentroid',
                                                                             'ycentroid',
                                                                             'focus',
                                                                             'filename']))


class TestFastBestImageDetection(TestCase):

    def setUp(self) -> None:
        peaks = [5000, 10000, 15000, 20000, 25000]
        focus = [500, 1000, 1500, 2000, 2500]
        self.saturation_level = 21000
        self.focus_key = 'FOCUS'

        self.file_list = []
        for i in range(1, 6):
            _file = f"test_file_{i}.fits"
            self.file_list.append(_file)
            data = np.ones(shape=(400, 400)) * peaks[i - 1]

            ccd = CCDData(data=data, header=fits.Header(), unit='adu')

            ccd.header.set('FILENAME', value=_file, comment="This file name")
            ccd.header.set('PEAK', value=peaks[i - 1], comment="This file set peak value")
            ccd.header.set(self.focus_key, value=focus[i - 1], comment="This file set peak value")
            ccd.write(_file, overwrite=True)


    def tearDown(self) -> None:
        for _file in self.file_list:
            if os.path.isfile(_file):
                os.unlink(_file)

    def test_get_best_image_by_peak(self):
        best_image = get_best_image_by_peak(file_list=self.file_list,
                                            saturation_level=self.saturation_level,
                                            focus_key=self.focus_key)
        image_name, peak, focus = best_image

        self.assertIsInstance(best_image, list)
        self.assertEqual(len(best_image), 3)
        self.assertEqual(image_name, 'test_file_4.fits')
        self.assertEqual(peak, 20000.)
        self.assertEqual(focus, 2000)



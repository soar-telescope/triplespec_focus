import os
import requests

from astropy.modeling import Model
from astropy.table import QTable
from ccdproc import CCDData
from pandas import DataFrame
from pathlib import Path
from unittest import TestCase

from ..triplespec_focus import TripleSpecFocus


class TripleSpecFocusTest(TestCase):

    def setUp(self) -> None:
        self.n_brightest_sources = 10
        self.tspec_focus = TripleSpecFocus()

        self.zenodo_id = '7055458'

        self.all_files = [
            'SV_ARC_13-04-2022_0010.fits',
            'SV_ARC_13-04-2022_0012.fits',
            'SV_ARC_13-04-2022_0014.fits',
            'SV_ARC_13-04-2022_0016.fits',
            'SV_ARC_13-04-2022_0019.fits',
            'SV_ARC_13-04-2022_0021.fits',
            'SV_ARC_13-04-2022_0023.fits',
            'SV_ARC_13-04-2022_0025.fits',
            'SV_ARC_13-04-2022_0011.fits',
            'SV_ARC_13-04-2022_0013.fits',
            'SV_ARC_13-04-2022_0015.fits',
            'SV_ARC_13-04-2022_0018.fits',
            'SV_ARC_13-04-2022_0020.fits',
            'SV_ARC_13-04-2022_0022.fits',
            'SV_ARC_13-04-2022_0024.fits',
            'SV_ARC_13-04-2022_0026.fits']

        self.test_file_name = 'SV_ARC_13-04-2022_0013.fits'

        for _file in self.all_files:
            if not os.path.exists(_file):
                url = f"https://zenodo.org/record/{self.zenodo_id}/files/{_file}?download=1"
                response = requests.get(url)
                with open(_file, "wb") as test_file:
                    print(f"Downloading file: {_file}")
                    test_file.write(response.content)

    def test_instance_is_correct(self):
        self.assertIsInstance(self.tspec_focus, TripleSpecFocus)

    def test_call_method_whithout_data_path_nor_file_list(self):
        self.assertRaises(SystemExit, self.tspec_focus)

    def test_get_focus_with_no_data(self):
        df = DataFrame()
        self.assertRaises(KeyError, self.tspec_focus.get_best_focus, df)

    def test_get_focus_with_data(self):
        focus_values = [-1300, -1200, -1100, -1000, -900, -800, -700]
        fwhm_values = [9, 8, 7, 6, 7, 8, 9]
        data = {'focus': focus_values, 'fwhm': fwhm_values}
        df = DataFrame(data=data)
        self.assertIsNone(self.tspec_focus.best_focus)
        self.assertIsNone(self.tspec_focus.fitted_model)
        results = self.tspec_focus.get_best_focus(df=df, x_axis_size=2000)
        self.assertAlmostEqual(self.tspec_focus.best_focus, -1000, places=4)
        self.assertIsNotNone(self.tspec_focus.fitted_model)
        self.assertIsInstance(self.tspec_focus.fitted_model, Model)
        self.assertIsInstance(results, list)

    def test_get_focus_with_fwhm_monotonically_increasing(self):
        focus_values = [-1300, -1200, -1100, -1000, -900, -800, -700]
        fwhm_values = [1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7]
        data = {'focus': focus_values, 'fwhm': fwhm_values}
        df = DataFrame(data=data)
        self.assertIsNone(self.tspec_focus.best_focus)
        self.assertIsNone(self.tspec_focus.fitted_model)
        results = self.tspec_focus.get_best_focus(df=df, x_axis_size=2000)
        self.assertIsNone(results)

    def test_detect_sources(self):
        ccd = CCDData.read(self.test_file_name, unit='adu')
        self.tspec_focus.show_mask = True
        sources = self.tspec_focus.detect_sources(ccd=ccd, debug_plots=False)
        self.assertIsInstance(sources, QTable)
        self.assertEqual(len(sources), 94)

    def test_call_with_data_path(self):

        results = self.tspec_focus(data_path='./')
        self.assertIsInstance(results, dict)
        self.assertEqual(len(results), 11)

    def test_call_with_file_list(self):
        data_path = Path('./')

        results = self.tspec_focus(file_list=sorted(data_path.glob(pattern='*.fits')))
        self.assertIsInstance(results, dict)
        self.assertEqual(len(results), 11)

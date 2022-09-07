Data and Process Overview
#########################

You have seen already a sample of the data. You will get typically 10 to 15 images taken at different focus values.

A quick examination routine will select the sharpest image by selecting the most intense image whose peak is also below
the saturation level which for TripleSpec Slit Viewer camera is set to :math:`40.000 ADU`

On this selected image the ``DAOStarFinder`` routing will be used to detect all the sources, the the most intense sources
are selected to obtain the :math:`FWHM` using ``ApertureStats`` which is then fitted with a ``Chebyshev1D`` of order :math:`6`.
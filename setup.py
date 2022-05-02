import os

from codecs import open
from setuptools import find_packages, setup

try:
    from ConfigParser import ConfigParser
except ImportError:
    from configparser import ConfigParser

CONF = ConfigParser()

HERE = os.path.abspath(os.path.dirname(__file__))


def create_version_py(packagename, version, source_dir='.'):
    package_dir = os.path.join(source_dir, packagename)
    version_py = os.path.join(package_dir, 'version.py')

    version_str = "# This is an automatic generated file please do not edit\n" \
                  "__version__ = '{:s}'".format(version)

    with open(version_py, 'w') as f:
        f.write(version_str)


# read content from README.md
with open(os.path.join(HERE, 'README.md')) as f:
    long_description = f.read()

CONF.read([os.path.join(os.path.dirname(__file__), 'setup.cfg')])

metadata = dict(CONF.items('metadata'))

PACKAGENAME = metadata['name']

LICENSE = metadata['license']

DESCRIPTION = metadata['description']

LONG_DESCRIPTION = long_description

LONG_DESCRIPTION_CONTENT_TYPE = 'text/markdown'

AUTHOR = metadata['author']

AUTHOR_EMAIL = metadata['author_email']

options = dict(CONF.items('options'))

INSTALL_REQUIRES = options['install_requires'].split()

SETUP_REQUIRES = options['setup_requires'].split()

# freezes version information in version.py
# create_version_py(PACKAGENAME, VERSION)


setup(
    name=PACKAGENAME,

    use_scm_version=True,

    description=DESCRIPTION,

    long_description=LONG_DESCRIPTION,

    long_description_content_type=LONG_DESCRIPTION_CONTENT_TYPE,

    # The project's main homepage.
    url='https://github.com/soar-telescope/triplespec_focus',

    # Author details
    author=AUTHOR,

    author_email=AUTHOR_EMAIL,

    # Choose your license
    license=LICENSE,

    packages=find_packages(exclude=['tests']),

    python_requires=">=3.6",

    install_requires=INSTALL_REQUIRES,

    setup_requires=SETUP_REQUIRES,

    entry_points={
        'console_scripts': [
            'triplespec-focus=triplespec_focus:run_triplespec_focus',
        ]
    }

)

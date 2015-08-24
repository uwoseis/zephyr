'''Zephyr
'''

import os
from distutils.core import setup
from setuptools import find_packages
import numpy as np

CLASSIFIERS = [
'Development Status :: 3 - Alpha',
'Intended Audience :: Developers',
'Intended Audience :: Science/Research',
'License :: OSI Approved :: MIT License',
'Programming Language :: Python',
'Topic :: Scientific/Engineering',
'Topic :: Scientific/Engineering :: Mathematics',
'Topic :: Scientific/Engineering :: Physics',
'Operating System :: Microsoft :: Windows',
'Operating System :: POSIX',
'Operating System :: Unix',
'Operating System :: MacOS',
'Natural Language :: English',
]

with open('README.md') as fp:
    LONG_DESCRIPTION = ''.join(fp.readlines())

setup(
    name = 'Zephyr',
    # version = '0.1.1',
    packages = find_packages(),
    install_requires = ['numpy>=1.7',
                        'scipy>=0.13',
                        'IPython>=4.0',
                        'ipyparallel>=4.0'
                       ],
    author = 'Brendan Smithyman',
    author_email = 'brendan@bitsmithy.net',
    description = 'Zephyr',
    long_description = LONG_DESCRIPTION,
    license = 'MIT',
    keywords = 'full-waveform inversion',
    # url = '',
    download_url = 'http://github.com/bsmithyman/zephyr',
    classifiers = CLASSIFIERS,
    platforms = ['Windows', 'Linux', 'Solaris', 'Mac OS-X', 'Unix'],
    use_2to3 = False,
    include_dirs=[np.get_include()],
)

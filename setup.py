import sys
if sys.version_info < (3, 6,):
    sys.exit('scmmd requires Python >= 3.6')
from pathlib import Path

from setuptools import setup, find_packages

try:
    from scmmd import __author__, __email__
except ImportError:  # Deps not yet installed
    __author__ = __email__ = ''

long_description = '''
There is a classical problem in statistics known as the **two-sample problem.**
In this setting, you are given discrete observations of two different distributions and asked to determine if the distributions are significantly different.
A special univariate case of this problem is familiar to many biologists as a "difference in means" comparison -- as performed using Student's *t*-test.\n
\n
This problem becomes somewhat more complex in high-dimensions, as differences between distributions may manifest not only in the mean location, but also in the covariance structure and modality of the data.
One approach to comparing distributions in this setting leverages kernel similarity metrics to find the maximum mean discrepancy (Gretton et. al. 2012, *JMLR*) -- the largest difference in the means of the distributions under a flexible transformation.\n
\n
Here, we adapt the MMD method to compare cell populations in single cell measurement data.
In the frame of the two-sample problem, each cell population of interest is considered as a distribution and each cell is a single observation from the source distribution.
We use the MMD to compute (1) a metric of the magnitude of difference between two cell populations, and (2) a p-value for the significance of this difference.\n
'''

setup(
    name='scmmd',
    version='0.1.0',
    description='Maximum mean discrepancy comparisons single cell profiles',
    long_description=long_description,
    url='http://github.com/calico/scmmd',
    author=__author__,
    author_email=__email__,
    license='Apache',
    python_requires='>=3.6',
    install_requires=[
        l.strip() for l in
        Path('requirements.txt').read_text('utf-8').splitlines()
    ],
    packages=find_packages(),
    entry_points=dict(
        console_scripts=['scmmd=scmmd.mmd_ad:main'],
    ),
    classifiers=[
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
    ],
)

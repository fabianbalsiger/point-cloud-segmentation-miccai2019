import os
import sys
from setuptools import find_packages, setup

if sys.version_info < (3, 6):
    sys.exit('Requires Python 3.6 or higher')

directory = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join('README.md'), encoding='utf-8') as f:
    readme = f.read()

REQUIRED_PACKAGES = [
    'tensorflow-gpu == 1.12.0',  # install this dependency before all others to ensure correct package versions
    'matplotlib >= 3.0.3',
    'numpy == 1.14.5',
    'pandas >= 0.24.1',
    'plotly >= 3.7.0',
    'SimpleITK >= 1.2.0',
    'scikit-image >= 0.4.12',
    'scipy >= 1.1.0',
    'pymia == 0.2.1',  # newer versions might not work due to active development of pymia!
    'torch',  # only used for data loading (and already a dependency of pymia)
]

TEST_PACKAGES = [

]

setup(
    name='point-cloud-segmentation-miccai2019',
    version='0.1.0',
    description='Learning Shape Representation on Sparse Point Clouds for Volumetric Image Segmentation',
    long_description=readme,
    long_description_content_type='text/markdown',
    author='Fabian Balsiger',
    author_email='fabian.balsiger@artorg.unibe.ch',
    url='https://doi.org/10.1007/978-3-030-32245-8_31',
    license='GNU GENERAL PUBLIC LICENSE',
    packages=find_packages(exclude=['test', 'docs']),
    install_requires=REQUIRED_PACKAGES,
    tests_require=REQUIRED_PACKAGES + TEST_PACKAGES,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Topic :: Scientific/Engineering :: Physics',
    ],
    keywords=[
        'image segmentation',
        'deep learning',
        'point cloud'
    ]
)

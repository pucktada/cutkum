from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['numpy>=1.0', 'ConfigArgParse>=0.12', 'tensorflow']

setup(
    name='cutkum',
    version='1.0',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='Thai Word-Segmentation with LSTM in Tensorflow'
)
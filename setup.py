from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['numpy>=1.14', 'ConfigArgParse>=0.12', 'tensorflow>=1.3']

def readme():
	with open('README.md') as f:
		return f.read()

setup(name='cutkum',
	version='1.4',
	description='Thai Word-Segmentation with LSTM in Tensorflow',
	long_description=readme(),
	keywords='thai tokenizer tensorflow lstm',
	url='https://github.com/pucktada/cutkum',
	author='Puck Treeratpituk',
	author_email='pucktada@gmail.com',
	license='MIT',
    packages=find_packages(),
    install_requires=REQUIRED_PACKAGES,
	entry_points={
		'console_scripts': ['cutkum=cutkum.command_line:main'],
	},
	include_package_data=True,
	zip_safe=True)

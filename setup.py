from setuptools import find_packages
from setuptools import setup
import codecs

REQUIRED_PACKAGES = ['numpy>=1.14', 'ConfigArgParse>=0.12', 'tensorflow>=1.4']

def readme():
	with codecs.open('README.md','r',encoding='utf-8') as f:
		return f.read()

setup(name='cutkum',
	version='2.4',
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

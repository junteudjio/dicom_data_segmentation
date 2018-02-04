# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='dicom_data_segmentation',
    version='0.0.1',
    description='A framework for dicom images segmentation.',
    long_description=readme,
    author='Junior Teudjio Mbativou',
    author_email='jun.teudjio@gmail.com',
    url='https://github.com/junteudjio',
    license=license,
    packages=find_packages(exclude=('tests')),
    install_requires=[
        'scikit-image==0.13.1',
        'dicom_data_pipeline==0.0.1'
    ],
    dependency_links=[
        'git+https://github.com/junteudjio/dicom_data_pipeline.git@develop#egg=dicom_data_pipeline-0.0.1',
    ]

)


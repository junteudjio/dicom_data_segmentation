#!/usr/bin/python

from dicom_data_pipeline import data_loader
from segmentation import FixedTresholdHistogramSegmentation

__author__ = 'Junior Teudjio'


dicom_mask_dataset = \
    data_loader.DicomMasksDataset(img_masks_filepath='_data/image-masks.i-contours.o-contours.csv')

segmentation_algo = FixedTresholdHistogramSegmentation(
    data_iterator = dicom_mask_dataset,
    segmentation_method = FixedTresholdHistogramSegmentation.__name__,
    plots_prefix = '_plots',
    results_prefix = '_results'
)

segmentation_algo.fit()
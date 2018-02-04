#!/usr/bin/python

from dicom_data_pipeline import data_loader
from segmentation import FixedThresholdSegmentation, FlexibleThresholdSegmentation, ActiveContourSegmentation

__author__ = 'Junior Teudjio'


get_data_iterator = lambda : \
    data_loader.DicomMasksDataset(img_masks_filepath='_data/image-masks.i-contours.o-contours.csv')


########################################################################################################################
# Fit the Fixed Threshold segmentation model
########################################################################################################################
FixedThresholdSegmentation(
    data_iterator = get_data_iterator(),
    segmentation_method = FixedThresholdSegmentation.__name__,
    plots_prefix = '_plots',
    results_prefix = '_results'
).fit()

########################################################################################################################
# Fit the Flexible Threshold segmentation model
########################################################################################################################
FlexibleThresholdSegmentation(
    data_iterator = get_data_iterator(),
    segmentation_method = FlexibleThresholdSegmentation.__name__,
    plots_prefix = '_plots',
    results_prefix = '_results'
).fit()


########################################################################################################################
# Fit the Active Contour segmentation model
########################################################################################################################
ActiveContourSegmentation(
    data_iterator = get_data_iterator(),
    segmentation_method = ActiveContourSegmentation.__name__,
    plots_prefix = '_plots',
    results_prefix = '_results'
).fit()

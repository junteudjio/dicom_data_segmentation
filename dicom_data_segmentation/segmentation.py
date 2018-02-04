import os

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu
from skimage.segmentation import active_contour
from skimage.morphology import watershed

from abstract_segmentation import AbstractSegmentation
import segmentation_helpers

from dicom_data_pipeline.data_parser import DataParser

__author__ = 'Junior Teudjio'
__all__ = ['FixedThresholdSegmentation', 'FlexibleThresholdSegmentation', 'ActiveContourSegmentation']


class AbstractThresholdSegmentation(AbstractSegmentation):
    def plot_segmentation_results(self, sample, show_plots=False):
        # in addition to these parent plots, histogram approaches also plots the histograms intersection
        super(AbstractThresholdSegmentation, self).plot_segmentation_results(sample, show_plots=False)

        # now plot the histograms intersection
        plt.clf()
        fig = plt.figure()
        title='hist intersect - sample: #{} - threshold: {}'.format(self.sample_idx, sample['threshold'])
        segmentation_helpers.plot_histograms_intersection(sample['blood-pool-pixels'], sample['muscle-pixels'],
                                                          sample['threshold'], plt, title)

        savepath = os.path.join(self.plots_dir, '{}.hist-intersect.jpg'.format(self.sample_idx))
        fig.savefig(savepath)
        if show_plots: plt.show()

        
class FixedThresholdSegmentation(AbstractThresholdSegmentation):
    def __init__(self, data_iterator, segmentation_method, plots_prefix, results_prefix, threshold=0.3):
        super(FixedThresholdSegmentation, self).__init__(data_iterator,
                                                         segmentation_method, plots_prefix, results_prefix)

        self.threshold = threshold

    def do_segmentation(self, sample):
        img_norm, o_contour = sample['img-norm'], sample['o-contours']
        threshold_mask = img_norm > self.threshold
        pred_i_contour = np.logical_and(o_contour, threshold_mask)

        # enrich the sample with segmentation info
        sample['pred-i-contour'] = pred_i_contour
        sample['threshold'] = self.threshold


class FlexibleThresholdSegmentation(AbstractThresholdSegmentation):
    def do_segmentation(self, sample):
        img_norm, o_contour = sample['img-norm'], sample['o-contours']
        threshold = threshold_otsu(img_norm[o_contour])
        threshold_mask = img_norm > threshold
        pred_i_contour = np.logical_and(o_contour, threshold_mask)

        # enrich the sample with segmentation info
        sample['pred-i-contour'] = pred_i_contour
        sample['threshold'] = threshold

class ActiveContourSegmentation(AbstractSegmentation):
    def do_segmentation(self, sample):
        img_norm, o_polygon = sample['img-norm'], sample['o-polygon']
        init_snake = np.array(o_polygon)
        pred_i_polygons = active_contour(img_norm, snake=init_snake,
                                         alpha=0.005, beta=30,
                                         gamma=0.01
                                         ).tolist()
        pred_i_polygons = [tuple(el) for el in pred_i_polygons] # convert to tuple if not error by Pillow.
        pred_i_contour = DataParser._poly_to_mask(pred_i_polygons, img_norm.shape[0], img_norm.shape[1])

        # enrich the sample with segmentation info
        sample['pred-i-contour'] = pred_i_contour


class WatershedSegmentation(AbstractSegmentation):
    def do_segmentation(self, sample):
        #TODO(junior): do this one too if time.
        raise NotImplementedError



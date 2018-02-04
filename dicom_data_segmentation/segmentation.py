import os

import numpy as np
import matplotlib.pyplot as plt

from abstract_segmentation import AbstractSegmentation
import segmentation_helpers

__author__ = 'Junior Teudjio'
__all__ = ['FixedThresholdSegmentation']


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








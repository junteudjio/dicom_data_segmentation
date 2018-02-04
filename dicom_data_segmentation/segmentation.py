import numpy as np

from abstract_segmentation import AbstractSegmentation

__author__ = 'Junior Teudjio'


class FixedTresholdHistogramSegmentation(AbstractSegmentation):
    def do_segmentation(self, sample, threshold=0.21):
        img_norm, o_contour = sample['img-norm'], sample['o-contour']
        threshold_mask = img_norm > threshold
        pred_i_contour = np.logical_and(o_contour, threshold_mask)
        sample['pred-i-contour'] = pred_i_contour

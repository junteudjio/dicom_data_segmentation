import os
import abc

import numpy as np
import matplotlib.pyplot as plt

import segmentation_helpers
import utils

__author__ = 'Junior Teudjio'
__all__ = ['AbstractSegmentation']

class AbstractSegmentation(object):
    '''Abstract class on which all segmentation experiments are built by defining the do_segmentation method'''
    __metaclass__ = abc.ABCMeta

    def __init__(self, data_iterator, segmentation_method, plots_prefix, results_prefix):
        '''
        Construct the Segmentation experiment object
        Parameters
        ----------
        data_iterator: dicom_data_pipeline.dataset
            An iterator generating one sample dict(img, i-contour, o-contour) at a time.
            #TODO(junior): try to handle batch_samples if time.
        segmentation_method : basestring
            The name of the segmentation experiment.
        plots_prefix: basestring
            The prefix of directory where to save all plots.
        results_prefix: basestring
            The prefix of directory where to save the accumulated results.
        '''
        self.data_iterator = data_iterator
        self.segmentation_method = segmentation_method

        self.plots_dir = os.path.join(plots_prefix, segmentation_method)
        self.results_dir = os.path.join(results_prefix, segmentation_method)
        utils.mkdir_p(self.plots_dir)
        utils.mkdir_p(self.results_dir)

    def _init_accumulators(self, results_to_accumulate=None):
        '''
        Initialize the accumulators for the values we want to save to disk.
        Parameters
        ----------
        results_to_accumulate: None | list
            list of intermediate values to accumulate computed when segmenting one sample.
            if None default to :
            ['dice-score', 'iou-score', 'o-polygon', 'i-polygon', 'pred-i-polygon']
        Returns
        -------
        '''
        if not results_to_accumulate:
            results_to_accumulate = ['dice-score', 'iou-score',
                                    'o-polygon', 'i-polygon', 'pred-i-polygon']
        self.results_to_accumulate = results_to_accumulate
        self.accumulators = dict()
        for result_key in self.results_to_accumulate:
            self.accumulators[result_key] = []

    def pre_segmentation_enrich(self, sample):
        img, i_mask, o_mask = sample['img'], sample['i-contour'], sample['o-contour']
        # add normalized img
        sample['img-norm'] = img_norm = img / 255.0

        # add i-contour pixels values: blood pool
        sample['blood-pool-pixels'] = img_norm[i_mask]

        # add muscles pixels
        sample['muscle-pixels'] = img_norm[np.logical_and(o_mask, ~ i_mask)]

        # add i-polygon
        sample['i-polygon'] = segmentation_helpers.mask2polygon(i_mask)

        # add o-polygon
        sample['o-polygon'] = segmentation_helpers.mask2polygon(o_mask)

    @abc.abstractmethod
    def do_segmentation(self, sample):
        pass

    def post_segmentation_enrich(self, sample):
        pred_i_polygon = segmentation_helpers.mask2polygon(sample['pred-i-contour'])

        # add pred-i-polygon
        sample['pred-i-polygon'] = pred_i_polygon

    def plot_segmentation_results(self, sample, show_plots=False):
        plt.clf()
        img, o_polygon, i_polygon, pred_i_polygon = sample['img'], sample['o-polygon'],\
                                                    sample['i-polygon'], sample['pred-i-polygon']

        fig, axes = plt.subplots(1, 2, figsize=(15, 20))

        segmentation_helpers.plot_img_polygons_overlay(img, (i_polygon, o_polygon), axes[0, 0],
                                                       title='true i-contour #{}'.format(self.sample_idx))

        segmentation_helpers.plot_img_polygons_overlay(img, (pred_i_polygon, o_polygon), axes[0, 1],
                                                       title='pred i-contour #{}'.format(self.sample_idx))

        savepath = os.path.join(self.plots_dir, '{}.img.i-contour.pred-contour.jpg'.format(self.sample_idx))
        fig.savefig(savepath)
        if show_plots: plt.show()

    def compute_scores(self, sample):
        true_mask, pred_mask = sample['i-contour', 'pred-i-contour']
        sample['dice-score'] = segmentation_helpers.dice_score(true_mask, pred_mask)
        sample['iou-score'] = segmentation_helpers.iou_score(true_mask, pred_mask)

    def accumulate_results(self, sample):
        for result_key in self.results_to_accumulate:
            # convert to numpy array if list
            if isinstance(sample[result_key], list):
                sample[result_key] = np.array(sample[result_key])
                
            # only accumulate if numpy array
            if isinstance(sample[result_key], np.ndarray):
                self.accumulators[result_key].append(sample[result_key])

    def plot_scores_statistics(self, score_type='dice-score', show_plots=False):
        plt.clf()
        scores = self.accumulators[score_type]
        fig, axes = plt.subplots(1, 3, figsize=(15, 30))

        segmentation_helpers.plot_scores_violin(scores, axes[0, 0], title='{}s - violin plot'.format(score_type))
        segmentation_helpers.plot_scores_histogram(scores, axes[0, 1], title='{}s - histogram'.format(score_type))
        segmentation_helpers.plot_scores_qq(scores, axes[1, 2], title='{}s - QQ plot'.format(score_type))

        savepath = os.path.join(self.plots_dir, '{}.{}.violin-hist-qq-plots.jpg'.format(self.sample_idx, score_type))
        fig.savefig(savepath)
        if show_plots: plt.show()


    def save_accumulated_results(self):
        saved_results_keys = '.'.join(self.results_to_accumulate)
        savepath = os.path.join(self.results_dir, 'accumulator.{}.npz'.format(saved_results_keys))
        np.savez(savepath, **self.accumulators)

    def fit(self, show_plots=False,
                results_to_accumulate=None):
        '''
        Applies the segmentation method to each sample in the dataset and save plots and results to disk.
        Parameters
        ----------
        show_plots: bool
            Show plots or not (tip: set to True when in jupyter notebook)
        results_to_accumulate: None | list
            list of computed intermediate values to accumulate when segmenting one sample.
            if None default to :
            ['dice-score', 'iou-score', 'o-polygon', 'i-polygon', 'pred-i-polygon']

        Returns
        -------
        '''

        self._init_accumulators(results_to_accumulate)

        for sample_idx, sample in enumerate(self.data_iterator):
            self.sample_idx = sample_idx #implicitly pass the idx to subsequent methods to keep a clean api.
            self.pre_segmentation_enrich(sample)
            self.do_segmentation(sample)
            self.post_segmentation_enrich(sample)
            self.plot_segmentation_results(sample, show_plots)
            self.compute_scores(sample)
            self.accumulate_results(sample)
            
        self.plot_scores_statistics(score_type='dice-score', show_plots=show_plots)
        self.plot_scores_statistics(score_type='iou-score', show_plots=show_plots)
        self.save_accumulated_results()
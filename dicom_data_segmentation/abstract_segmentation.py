import os
import abc
import logging
import time

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

import segmentation_helpers
import utils

__author__ = 'Junior Teudjio'
__all__ = ['AbstractSegmentation']

class AbstractSegmentation(object):
    '''Abstract class on which all segmentation experiments are built by defining the do_segmentation method'''
    __metaclass__ = abc.ABCMeta

    def __init__(self, data_iterator, segmentation_method,
                 plots_prefix='_plots', results_prefix='_results', logs_prefix='_logs'):
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
        logs_prefix: basestring
            The prefix of directory where to save the logs files.
        '''
        self.data_iterator = data_iterator
        self.segmentation_method = segmentation_method

        self.plots_dir = os.path.join(plots_prefix, segmentation_method)
        self.results_dir = os.path.join(results_prefix, segmentation_method)
        self.logs_prefix = logs_prefix
        utils.mkdir_p(self.plots_dir)
        utils.mkdir_p(self.results_dir)
        utils.mkdir_p(self.logs_prefix)

        self._set_logger()

    def _init_accumulators(self, results_to_accumulate=None):
        '''
        Initialize the accumulators for the values we want to save to disk.
        Parameters
        ----------
        results_to_accumulate: None | list
            list of intermediate values to accumulate computed when segmenting one sample.
            if None default to :
            ['dice-score', 'iou-score', 'o-polygon', 'i-polygon', 'pred-i-polygon', 'i-contours', 'pred-i-contour']
        Returns
        -------
        '''
        if not results_to_accumulate:
            results_to_accumulate = ['dice-score', 'iou-score',
                                    'o-polygon', 'i-polygon', 'pred-i-polygon', 'i-contours', 'pred-i-contour']
        self.results_to_accumulate = results_to_accumulate
        self.accumulators = dict()
        for result_key in self.results_to_accumulate:
            self.accumulators[result_key] = []

    def _set_logger(self):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(levelname)s:%(asctime)s:%(name)s:%(message)s')

        file_handler = logging.FileHandler(os.path.join(self.logs_prefix, '{}.log'.format(self.segmentation_method)))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))

        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

        self.logger = logger

    def pre_segmentation_enrich(self, sample):
        '''
        Enrich the sample dict with new data before the segmentation step
        Parameters
        ----------
        sample: dict
            expected keys: img, i-contours, o-contours

        Returns
        -------
            added keys: img-norm, blood-pool-pixels, muscle-pixels, i-polygon, o-polygon
        '''
        img, i_mask, o_mask = sample['img'], sample['i-contours'], sample['o-contours']
        # add normalized img
        sample['img-norm'] = img_norm = img / 255.

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
        '''
        Enrich the sample dict with new data after the segmentation step
        Parameters
        ----------
        sample: dict
            expected keys: pred-i-contour

        Returns
        -------
            added keys: pred-i-polygon
        '''
        pred_i_polygon = segmentation_helpers.mask2polygon(sample['pred-i-contour'])

        # add pred-i-polygon
        sample['pred-i-polygon'] = pred_i_polygon

    def plot_segmentation_results__polygons(self, sample, show_plots=False):
        '''
        Plot ground truth and predicted polygons overlay on image.
        Parameters
        ----------
        sample: dict
           expected keys: img, o-polygon, i-polygon, pred-i-polygon

        show_plots: bool

        Returns
        -------
           added keys:
       '''
        plt.clf()

        img, o_polygon, i_polygon, pred_i_polygon = sample['img'], sample['o-polygon'], \
                                                    sample['i-polygon'], sample['pred-i-polygon']
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        segmentation_helpers.plot_img_polygons_overlay(img, (i_polygon, o_polygon), axes[0],
                                                       title='true i-polygon #{}'.format(self.sample_idx))

        segmentation_helpers.plot_img_polygons_overlay(img, (pred_i_polygon, o_polygon), axes[1],
                                                       title='pred i-polygon #{}'.format(self.sample_idx))

        savepath = os.path.join(self.plots_dir, '{}.img.i-polygon.pred-polygon.jpg'.format(self.sample_idx))
        fig.savefig(savepath)
        if show_plots: plt.show()

    def plot_segmentation_results__masks(self, sample, show_plots=False):
        '''
        Plot ground truth and predicted masks overlay on image.
        Parameters
        ----------
        sample: dict
           expected keys: img, o-contours, i-contours, pred-i-contour

        show_plots: bool

        Returns
        -------
           added keys:
       '''
        plt.clf()

        img, o_contour, i_contour, pred_i_contour = sample['img'], sample['o-contours'], \
                                                    sample['i-contours'], sample['pred-i-contour']
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        segmentation_helpers.plot_img_masks_overlay(img, (i_contour, o_contour), axes[0],
                                                       title='true i-mask #{}'.format(self.sample_idx))

        segmentation_helpers.plot_img_masks_overlay(img, (pred_i_contour, o_contour), axes[1],
                                                       title='pred i-mask #{}'.format(self.sample_idx))

        savepath = os.path.join(self.plots_dir, '{}.img.i-mask.pred-mask.jpg'.format(self.sample_idx))
        fig.savefig(savepath)
        if show_plots: plt.show()


    def plot_segmentation_results(self, sample, show_plots=False):
        '''
        Plot ground truth and predicted polygons and masks overlay on image.
        Parameters
        ----------
        sample: dict
           expected keys: img, o-polygon, i-polygon, pred-i-polygon, o-contours, i-contours, pred-i-contour

        show_plots: bool

        Returns
        -------
           added keys:
       '''
        self.plot_segmentation_results__polygons(sample, show_plots)
        self.plot_segmentation_results__masks(sample, show_plots)

    def compute_scores(self, sample):
        '''
        Compute the segmentation scores on this image.
        Parameters
        ----------
        sample: dict
           expected keys: i-contours, pred-i-contour

        Returns
        -------
           added keys: dice-score, iou-score
       '''
        true_mask, pred_mask = sample['i-contours'], sample['pred-i-contour']
        sample['dice-score'] = segmentation_helpers.dice_score(true_mask, pred_mask)
        sample['iou-score'] = segmentation_helpers.iou_score(true_mask, pred_mask)

    def accumulate_results(self, sample):
        '''
        Save some of the enriched sample data in an accumulator for aggregation/saving purporses.
        Parameters
        ----------
        sample: dict
           expected keys: see self.results_to_accumulate

        Returns
        -------
           added keys:
       '''
        for result_key in self.results_to_accumulate:
            # convert to numpy array if list
            if isinstance(sample[result_key], list):
                sample[result_key] = np.array(sample[result_key])
                
            # only accumulate if numpy array or float
            if isinstance(sample[result_key], np.ndarray) or isinstance(sample[result_key], float):
                self.accumulators[result_key].append(sample[result_key])

    def plot_scores_statistics(self, score_type='dice-score', show_plots=False):
        '''
        Plot some statistics on the scores over all images.
        Parameters
        ----------
        score_type: basestring
           The Id of the score to compute. one of : dice-score or iou-score

        Returns
        -------
       '''
        plt.clf()
        scores = self.accumulators[score_type]
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        segmentation_helpers.plot_scores_violin(scores, axes[0], title='{}s - violin plot (median)'.format(score_type))
        segmentation_helpers.plot_scores_histogram(scores, axes[1], title='{}s - histogram'.format(score_type))
        segmentation_helpers.plot_scores_qq(scores, axes[2], title='{}s - QQ plot'.format(score_type))

        fig.suptitle('Segmentation {}s statistics over all patients'.format(score_type), fontsize=15)
        savepath = os.path.join(self.plots_dir, '_scores.{}.violin-hist-qq-plots.jpg'.format(score_type))
        fig.savefig(savepath)
        if show_plots: plt.show()


    def save_accumulated_results(self):
        '''
        Save the variables accumulated in self.accumulators over all samples to disk.
        also see: self.results_to_accumulate , self._init_accumulators()
       '''
        saved_results_keys = '.'.join(self.results_to_accumulate)
        savepath = os.path.join(self.results_dir, 'accumulator.{}.npz'.format(saved_results_keys))
        np.savez(savepath, **self.accumulators)

    def fit(self, show_plots=False, max_plots=10,
                results_to_accumulate=None):
        '''
        Applies the segmentation method to each sample in the dataset and save plots and results to disk.
        Parameters
        ----------
        show_plots: bool
            Show plots or not (tip: set to True when in jupyter notebook)
        max_plots: int
            Maximum number of patients for which to plot data (handy to not have huge notebooks for instance).
        results_to_accumulate: None | list
            list of computed intermediate values to accumulate when segmenting one sample.
            if None default to :
            ['dice-score', 'iou-score', 'o-polygon', 'i-polygon', 'pred-i-polygon', 'i-contours', 'pred-i-contour']

        Returns
        -------
        '''

        self._init_accumulators(results_to_accumulate)
        _initial_show_plots_value = show_plots
        segmentation_start = time.time()
        for sample_idx, sample in enumerate(self.data_iterator):
            self.logger.info('''
            Doing {} on patient #{}
            ###########################################################################################'''.format(
                self.segmentation_method, sample_idx
            ))
            sample_start = time.time()

            self.sample_idx = sample_idx #implicitly pass the idx to subsequent methods to keep a clean api.
            self.pre_segmentation_enrich(sample)
            self.do_segmentation(sample)
            self.post_segmentation_enrich(sample)
            self.plot_segmentation_results(sample, show_plots)
            self.compute_scores(sample)
            self.accumulate_results(sample)

            sample_end = time.time()

            self.logger.info('''
            Patient #{} data segmentation done in {} seconds.
            ###########################################################################################
            
            '''.format(sample_idx, sample_end - sample_start))

            # if max_plots reached, toggle show_plots off
            if sample_idx >= max_plots: show_plots=False


        # reset show_plots initial value
        show_plots = _initial_show_plots_value

        segmentation_end = time.time()
        self.logger.info('''
        All Patients Segmentation done in {} seconds.
        ###########################################################################################
        
        '''.format( segmentation_end - segmentation_start))

        self.logger.info('''
        Segmentation Scores statistics over all patients :
        ###########################################################################################
        '''.format(segmentation_end - segmentation_start))

        self.logger.info('''
        Quantitative Statistics on Scores:
            dice_scores stats :
                {dice_scores_stats}
            dice_scores median :
                {dice_scores_median}
                
                
            Intersection_over_Union_scores stats:
                {iou_scores_stats}
             Intersection_over_Union_scores median :
                {iou_scores_median}
        
        ###########################################################################################
        '''.format(dice_scores_stats=stats.describe(self.accumulators['dice-score']),
                   iou_scores_stats=stats.describe(self.accumulators['iou-score']),
                   dice_scores_median=np.median(self.accumulators['dice-score']),
                   iou_scores_median=np.median(self.accumulators['iou-score'])
                   ))

        self.plot_scores_statistics(score_type='dice-score', show_plots=show_plots)
        self.plot_scores_statistics(score_type='iou-score', show_plots=show_plots)
        self.save_accumulated_results()

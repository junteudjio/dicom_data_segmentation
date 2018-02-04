import os
import glob
import logging
import itertools

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

__author__ = 'Junior Teudjio'


RESULTS_DIR = '_results'
PLOTS_DIR = '_plots'
LOGS_PREFIX = '_logs'
METHOD_NAMES = [
    'FixedThresholdSegmentation',
    'FlexibleThresholdSegmentation',
    'ActiveContourSegmentation'
]
ALL_METHODS_STR = '.'.join(METHOD_NAMES)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(levelname)s:%(asctime)s:%(name)s:%(message)s')

file_handler = logging.FileHandler(os.path.join(LOGS_PREFIX, 'comparison-{}'.format(ALL_METHODS_STR)))
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logging.Formatter('%(message)s'))

logger.addHandler(file_handler)
logger.addHandler(stream_handler)


def load_accumulated_results(method_name):
    results_filepath = glob.glob(os.path.join(RESULTS_DIR, method_name, 'accumulator*'))[0]
    return np.load(results_filepath)

def compare_scores_difference_significance(method1, method2, score_type='dice-score'):
    res1 = load_accumulated_results(method_name=method1)
    res2 = load_accumulated_results(method_name=method2)
    return stats.mannwhitneyu(res1[score_type], res2[score_type])

def plot_all_methods_boxplots(method_names=METHOD_NAMES, score_type='dice-score'):
    plt.clf()
    fig = plt.figure(figsize=(15,5))
    all_scores = []
    for method_name in method_names:
        scores = load_accumulated_results(method_name)[score_type]
        all_scores.append(scores)

    plt.boxplot(all_scores, labels=METHOD_NAMES, vert=False)
    #fig.set(xticklabels=METHOD_NAMES)
    fig.suptitle('{}s - all methods'.format(score_type), fontsize=15)
    savepath = os.path.join(PLOTS_DIR, '_comparison.{}.{}.box-plots.jpg'.format(score_type, ALL_METHODS_STR))
    fig.savefig(savepath)
    plt.show()

def run():
    all_possible_comparisons = itertools.combinations(METHOD_NAMES, 2)
    for method1, method2 in all_possible_comparisons:
        wilcoxon_mann_whitney_dice_score = compare_scores_difference_significance(method1, method2, 'dice-score')
        wilcoxon_mann_whitney_iou_score = compare_scores_difference_significance(method1, method2, 'iou-score')

        logger.info('''
        compare_scores_difference_significance ({method1} vs. {method2}):
        ---------------------------------------------------------------------
            wilcoxon_mann_whitney_dice_score:
            {wilcoxon_mann_whitney_dice_score}
            -----------------------------------------------------------------
            wilcoxon_mann_whitney_iou_score:
            {wilcoxon_mann_whitney_iou_score}
        ############################################################################################
        
        
        '''.format(
            wilcoxon_mann_whitney_dice_score = wilcoxon_mann_whitney_dice_score,
            wilcoxon_mann_whitney_iou_score = wilcoxon_mann_whitney_iou_score,
            method1=method1,
            method2=method2
        ))


    plot_all_methods_boxplots(score_type='dice-score')
    plot_all_methods_boxplots(score_type='iou-score')

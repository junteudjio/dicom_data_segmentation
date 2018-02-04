from skimage import measure
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

__author__ = 'Junior Teudjio'
__all__ = ['mask2polygon', 'plot_img_polygons_overlay', 'dice_score', 'iou_score',
           'plot_scores_hist', 'plot_scores_qq', 'plot_scores_violin', '']

mask2polygon = lambda mask: measure.find_contours(mask, 0.8)[0]

_plots_colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w']

def plot_img_polygons_overlay(img, polygons, plot_ax, title=None):
    '''
    Merge the img and contour points in a single figure and save to disk
    Parameters
    ----------
    img: numpy array
    polygons: list of list of 2D tuples
    plot_ax: matplotlib plt or ax object
    title: basestring
        title of the plot
    Returns
    -------

    '''
    plot_ax.imshow(img)
    for idx, polygon in enumerate(polygons):
        x = [point[0] for point in polygon]
        y = [point[1] for point in polygon]
        color = _plots_colors[idx % len(_plots_colors)]
        plot_ax.plot(x, y, alpha=1, color=color)
    if title: plot_ax.set_title(title)

def dice_score(true_mask, pred_mask):
    '''
    Compute the dice coefficient score.
    Parameters
    ----------
    true_mask: bool numpy array
    pred_mask: bool numpy array

    Returns
    -------
    float
    '''
    intersection = np.logical_and(pred_mask, true_mask).sum()
    dice_score = (2.0 * intersection) / (pred_mask.sum() + true_mask.sum())
    return dice_score

def iou_score(true_mask, pred_mask):
    '''
    Compute the intersection over union (iou) score.
    Parameters
    ----------
    true_mask: bool numpy array
    pred_mask: bool numpy array

    Returns
    -------
    float
    '''
    intersection = np.logical_and(pred_mask, true_mask).sum()
    union = np.logical_or(pred_mask, true_mask).sum()
    return float(intersection) / union

def plot_scores_qq(scores, plot_ax, title=None):
    '''

    Parameters
    ----------
    scores
    plot_ax
    title

    Returns
    -------

    '''
    stats.probplot(scores, dist='norm', plot=plot_ax)
    if title: plot_ax.set_title(title)

def plot_scores_violin(scores, plot_ax, title=None):
    '''

    Parameters
    ----------
    scores
    plot_ax
    title

    Returns
    -------

    '''
    plot_ax.violinplot(scores, points=200, vert=True, widths=1.1,
                      showmeans=True, showextrema=True, showmedians=True,
                      bw_method=0.5)
    if title: plot_ax.set_title(title)

def plot_scores_hist(scores, plot_ax, title=None):
    '''

    Parameters
    ----------
    scores
    plot_ax
    title

    Returns
    -------

    '''
    # hist(x, 50, normed=1, facecolor='green', alpha=0.75)
    plot_ax.hist(scores, 50, normed=1, facecolor='green', alpha=0.75)
    if title: plot_ax.set_title(title)


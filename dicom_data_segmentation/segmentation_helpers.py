from skimage import measure
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

__author__ = 'Junior Teudjio'
__all__ = ['mask2polygon', 'plot_img_polygons_overlay', 'dice_score', 'iou_score',
           'plot_scores_histogram', 'plot_scores_qq', 'plot_scores_violin', 'plot_img_masks_overlay',
           'plot_histograms_intersection']

mask2polygon = lambda mask: measure.find_contours(mask, 0.8)[0]

_plots_colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w']

def plot_img_polygons_overlay(img, polygons, plot_ax, title=None):
    '''
    Merge the img and polygons points in a single figure
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
        if isinstance(polygon, list):
            polygon = np.array(polygon)
        color = _plots_colors[idx % len(_plots_colors)]
        plot_ax.plot(polygon[:, 1], polygon[:, 0], color=color)
    if title: plot_ax.set_title(title)


def plot_img_masks_overlay(img, masks,  plot_ax, title=None):
    '''
    Merge the img and masks in a single figure
    Parameters
    ----------
    img: numpy array
    masks: list of numpy array
    plot_ax: matplotlib plt or ax object
    title: basestring
        title of the plot

    Returns
    -------

    '''
    plot_ax.imshow(img)
    for mask in masks:
        plot_ax.imshow(mask, alpha=0.4)
    if title: plot_ax.set_title(title)


def plot_histograms_intersection(blood_pool, muscle, threshold, plot_ax, title=None):
    '''
    Plots the histograms intersection with the separating line
    Parameters
    ----------
    blood_pool_mask: np.float array
        normalized pixels of blood_pool
    muscle_mask: np.float array
        normalized pixels of muscle
    threshold: float

    Returns
    -------

    '''

    plot_ax.hist(blood_pool, alpha=0.5, label='blood pool')
    plot_ax.hist(muscle, alpha=0.5, label='muscle')
    plot_ax.axvline(x=threshold, label='threshold', color='g')
    plot_ax.legend()
    plot_ax.xlabel('normalized pixels values - threshold={:1f}'.format(threshold))
    #if title: plt.set_title(title)



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
    try:
        intersection = np.logical_and(pred_mask, true_mask).sum()
        score = (2.0 * intersection) / (pred_mask.sum() + true_mask.sum())
    except ValueError:
        score = 0.0
    #print 'dice_score :', score
    return score

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
    try:
        intersection = np.logical_and(pred_mask, true_mask).sum()
        union = np.logical_or(pred_mask, true_mask).sum()
        score = float(intersection) / union
    except ValueError:
        score = 0.0
    #print 'iou_score :', score
    return score

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
    plot_ax.violinplot(scores, vert=True, widths=0.5,
                      showmeans=False, showextrema=True, showmedians=True)
    if title: plot_ax.set_title(title)

def plot_scores_histogram(scores, plot_ax, title=None):
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
    plot_ax.hist(scores, normed=1, facecolor='green', alpha=0.75)
    if title: plot_ax.set_title(title)


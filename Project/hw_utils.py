import os
import sklearn
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, roc_auc_score

data = './data/'
out = './out/'

axistitlesize = 20
axisticksize = 17
axislabelsize = 26
axislegendsize = 23
axistextsize = 20
axiscbarfontsize = 15

def accuracy_metric(y_test, y_pred):
    """
    Calculate accuracy of model prediction.
    
    Parameters
    ----------
    y_test : array-like of shape (N, )
        Original labels of the test dataset.
    
    y_pred : array-like of shape (N, )
        Predicted labels of the test dataset.
    
    Returns
    -------
    Accuracy of model in reference of the true test labels.
    """
    # Binarize labels
    y_test = label_binarize(y_test, classes=np.unique(y_test))
    y_pred = label_binarize(y_pred, classes=np.unique(y_pred))

    correct = 0
    for (t, p) in zip(y_test, y_pred):
        if t.tolist() == p.tolist():
            correct += 1
    return correct / len(y_test) * 100


def plot_confusion_matrix(conf_mat, y, labels=None, title=None,
                          figsize=8, textsize=26,
                          save=False, save_filename='image'):
    """
    Plots a confusion matrix.
    """
    fig, axes = plt.subplots(figsize=(figsize,figsize))
    axes.set_aspect('equal')

    im = axes.imshow(conf_mat)
    # Loop over data dimensions and create text annotations.
    for X in range(conf_mat.shape[0]):
        for Y in range(conf_mat.shape[1]):
            axes.text(Y, X, conf_mat[X, Y], fontsize=textsize,
                      ha='center', va='center', color='white', fontweight='bold', 
                      bbox=dict(color=np.array((0,0,0,0.2)), lw=0)
                     )

    # Set axis tick locations and labels
    ticks = [i for i in range(len(set(y)))]
    if labels is None:
        ticklabels = [i+1 for i in range(len(set(y)))]
    else:
        ticklabels = list(labels)

    axes.set_xticks(ticks)
    axes.set_xticklabels(ticklabels)
    axes.set_yticks(ticks)
    axes.set_yticklabels(ticklabels)

    axes.set_xlabel('Predicted labels', fontsize=axislabelsize+5, fontweight='bold')
    axes.set_ylabel('True labels', fontsize=axislabelsize+5, fontweight='bold')
    axes.tick_params(axis='both', which='major', labelsize=axisticksize+5)
    axes.xaxis.tick_top()
    axes.xaxis.set_label_position('top') 

    axes.grid(False)

    # Create an axis on the right side of `axes`. The width of `cax` will be 5%
    # of `axes` and the padding between `cax` and axes will be fixed at 0.1 inch
    divider = make_axes_locatable(axes)
    cax = divider.append_axes('right', size='5%', pad=0.1)
    cbar = plt.colorbar(mappable=im, cax=cax)
    cbar.ax.tick_params(labelsize=axiscbarfontsize, colors='black')
    cbar.set_label('Number of occurences', fontsize=axiscbarfontsize+10, labelpad=15, rotation=90)

    plt.suptitle(title,
                 fontsize=axistitlesize+5, y=0.1)
    
    if save:
        if not os.path.exists(out):
            os.makedirs(out)
        f = save_filename.split('.')
        fn = f[0]
        ff = 'png' if len(f) == 1 else f[1]
        plt.savefig(out + fn + '.' + ff,
                    format=ff, dpi=200,
                    #facecolor='black', edgecolor='black',
                    bbox_inches='tight')

    plt.show()
    
def compute_roc(estimator, X, y):
    """
    Creates the ROC curve and computes AUC values an input X-y data-target set.
    
    Paramters
    ---------
    estimator : estimator instance
        Fitted classifier or a fitted :class:`~sklearn.pipeline.Pipeline`
        in which the last estimator is a classifier.
    
    X : {array-like, sparse matrix} of shape (n_test, n_features)
        Input values.
    
    y : array-like of shape (n_train,)
        Target values.
    
    Returns
    -------
    fpr : ndarray
        False positive rates.
    tpr : ndarray
        True positive rates.
    roc_auc : float
        Area under ROC curves.
    """
    from sklearn.metrics import _plot
    
    # Calculate scores for ROC curve
    # Small hack because I want to automate it. Also I have no trust in S.O. currently.
    # Using this: https://github.com/scikit-learn/scikit-learn/blob/0fb307bf3/sklearn/metrics/_plot/roc_curve.py#L114
    c_method = _plot.base._check_classifer_response_method(estimator,
                                                           response_method='auto')
    y_score = c_method(X)

    # Calculate ROC and AUC values
    pos_label = estimator.classes_[1]
    if y_score.ndim != 1:
        y_score = y_score[:, 1]
    fpr, tpr, _ = roc_curve(y, y_score, pos_label=pos_label)
    roc_auc = auc(fpr, tpr)
    
    return fpr, tpr, roc_auc
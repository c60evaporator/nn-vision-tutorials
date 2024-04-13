import numpy as np
from sklearn.metrics import auc
#from scipy.integrate import trapezoid
#from sklearn.metrics import average_precision_score

def trapezoid(y, x=None, dx=1.0, axis=-1):
    y = np.asanyarray(y)
    if x is None:
        d = dx
    else:
        x = np.asanyarray(x)
        if x.ndim == 1:
            d = np.diff(x)
            # reshape to correct shape
            shape = [1]*y.ndim
            shape[axis] = d.shape[0]
            d = d.reshape(shape)
        else:
            d = np.diff(x, axis=axis)
    nd = y.ndim
    slice1 = [slice(None)]*nd
    slice2 = [slice(None)]*nd
    slice1[axis] = slice(1, None)
    slice2[axis] = slice(None, -1)
    try:
        a1 = y[tuple(slice1)]
        a2 = y[tuple(slice2)]
        ave = (a1 + a2) / 2.0
        ret = (d * (y[tuple(slice1)] + y[tuple(slice2)]) / 2.0).sum(axis)
    except ValueError:
        # Operations didn't work, cast to ndarray
        d = np.asarray(d)
        y = np.asarray(y)
        ret = np.add.reduce(d * (y[tuple(slice1)]+y[tuple(slice2)])/2.0, axis)
    return ret

def aucc(x, y):
    if x.shape[0] < 2:
        raise ValueError(
            "At least 2 points are needed to compute area under curve, but x.shape = %s"
            % x.shape
        )

    direction = 1
    dx = np.diff(x)
    if np.any(dx < 0):
        if np.all(dx <= 0):
            direction = -1
        else:
            raise ValueError("x is neither increasing nor decreasing : {}.".format(x))

    area = direction * trapezoid(y, x)
    if isinstance(area, np.memmap):
        # Reductions such as .sum used internally in trapezoid do not return a
        # scalar by default for numpy.memmap instances contrary to
        # regular numpy.ndarray instances.
        area = area.dtype.type(area)
    return area

def _get_recall_precision(scores: np.ndarray, corrects: np.ndarray,
                          smoothe: bool=True):
    """
    Calculate precision-recall curve (PR Curve).

    Parameters
    ----------
    scores : array-like of shape (n_boxes,)
        A float array of confidence scores.

    corrects : array-like of shape (n_boxes,)
        A boolean array which indicates whether the bounding box is correct or not.

    smoothe : bool
        If True, the precision-recall curve is smoothed to fix the zigzag pattern.
    """
    # Order by confidence scores
    ordered_indices = np.argsort(-scores)
    corrects_ordered = corrects[ordered_indices]
    # Calculate the precision and the recall
    n_boxes = len(corrects_ordered)
    n_trues = corrects.sum()
    accumlulated_trues = np.array([np.sum(corrects_ordered[:i+1]) for i in range(n_boxes)])
    precision = accumlulated_trues / np.arange(1, n_boxes + 1)
    recall = accumlulated_trues / n_trues
    # Smooth the precision
    if smoothe:
        precision = np.array([np.max(precision[i:]) for i in range(n_boxes)])
    # to make sure that the precision-recall curve starts at (0, 1)
    recall = np.r_[0, recall]
    precision = np.r_[1, precision]

    return precision, recall

def _average_precision(precision: np.ndarray, recall: np.ndarray,
                       precision_center: bool=False):
    """
    Calculate average precision.

    .. note::
        This average precision is based on Area under curve (AUC) AP, NOT based on Interpolated AP. 
        Reference: https://jonathan-hui.medium.com/map-mean-average-precision-for-object-detection-45c121a31173

    Parameters
    ----------
    precision : array-like of shape (n_boxes,)
        Precision scores.

    recall : array-like of shape (n_boxes,)
        Recall scores.

    precision_center : bool
        This parameter is used for specifying which value is used as the y value during the calculation of area under curve (AUC).

        If True, use the center value (average of the left and the right value) as the y value like `sklearn.metrics.auc()`

        If False, use the right value as the y value like `sklearn.metrics.average_precision_score()`
    """
    # Calculate AUC (y value calculation method can be changed by `preciision_center` argument)
    if precision_center:
        average_precision = auc(recall, precision)
    else:
        average_precision = np.sum(np.diff(recall) * np.array(precision)[:-1])
    
    return average_precision


scores = np.array([72, 74, 78, 80, 83, 84, 88, 89, 92, 96])
corrects = np.array([True, False, False, True, True, False, False, False, True, True])
precision, recall = _get_recall_precision(scores, corrects, smoothe=True)
average_precision = _average_precision(precision, recall, precision_center=False)

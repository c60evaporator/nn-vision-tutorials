from typing import List
from sklearn.metrics import confusion_matrix
import numpy as np
from torch import Tensor

def segmentation_ious_one_image(labels_pred: Tensor, target: Tensor, labels: List[int]):
    """
    Calculate segmentation IoUs, TP, FP, FN in one image

    Reference: https://stackoverflow.com/questions/31653576/how-to-calculate-the-mean-iu-score-in-image-segmentation
    
    Parameters
    ----------
    labels_pred : List[Tensor(H x W)]
        The predicted labels of each pixel

    target : List[Tensor(H x W)]
        The true labels of each pixel
    
    labels : List[int]
        The list of labels
    """
    labels_flatten = labels_pred.cpu().detach().numpy().flatten()
    target_flatten = target.cpu().detach().numpy().flatten()
    confmat = confusion_matrix(target_flatten, labels_flatten, labels=labels)
    tps = np.diag(confmat)
    gts = np.sum(confmat, axis=1)  # Ground 
    preds = np.sum(confmat, axis=0)
    union = gts + preds - tps
    fps = preds - tps
    fns = gts - tps
    ious = np.divide(tps, union.astype(np.float32), out=np.full((len(labels),), np.nan), where=(union!=0))
    return ious, tps, fps, fns

def compute_iou(y_pred, y_true):
    # ytrue, ypred is a flatten vector
    pred_flatten = y_pred.flatten()
    true_flatten = y_true.flatten()
    current = confusion_matrix(true_flatten, pred_flatten, labels=list(range(6)))
    # compute mean iou
    intersection = np.diag(current)
    ground_truth_set = current.sum(axis=1)
    predicted_set = current.sum(axis=0)
    union = ground_truth_set + predicted_set - intersection
    IoU = intersection / union.astype(np.float32)
    return IoU

y_true = np.array([[0, 1, 0, 1, 0],
                   [0, 2, 0, 2, 0],
                   [0, 4, 0, 4, 0]])
y_pred = np.array([[0, 0, 1, 1, 0],
                   [0, 2, 0, 2, 0],
                   [0, 4, 0, 5, 0]])
iou = compute_iou(y_pred, y_true)
iou2, tps, fps, fns = segmentation_ious_one_image(Tensor(y_pred), Tensor(y_true), labels=list(range(6)))
print(iou)
print(iou2)
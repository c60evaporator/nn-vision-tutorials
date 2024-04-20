from typing import Dict, List, Literal
from torch import nn, Tensor, no_grad
from torchvision import ops
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import numpy as np
import copy
import time

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

def segmentation_ious(prediction_list: List[Dict[Literal['out', 'aux'], Tensor]],
                      target_list: List[Tensor],
                      idx_to_class: Dict[int, str], border_idx:int = None):
    """
    Calculate the average precision of each class label

    .. note::
        This average precision is based on Area under curve (AUC) AP, NOT based on Interpolated AP. 
        Reference: https://jonathan-hui.medium.com/map-mean-average-precision-for-object-detection-45c121a31173
    
    Parameters
    ----------
    prediction_list : List[Dict[Literal['out', 'aux'], Tensor(class x H x W)]]
        List of the predicted segmentation images

    target_list : List[Tensor(H x W)]
        List of the true segmentation images

    idx_to_class : Dict[int, str]
        A dict for converting class IDs to class names.

    border_idx : int
        Index of the border class
    
    Returns
    -------
    aps : Dict[int, Dict[Literal['label_name', 'average_precision', 'precision', 'recall'], Any]]
        Calculated average precisions with the label_names and the PR Curve
    """
    # List for storing scores
    tps_all = []
    fps_all = []
    fns_all = []
    ###### Calculate IoUs of each image ######
    # Add the border label to idx_to_class
    idx_to_class_bd = {k: v for k, v in idx_to_class.items()}
    if border_idx is not None:
        idx_to_class_bd[border_idx] = 'border'
    # Loop of images
    for i, (prediction, target) in enumerate(zip(prediction_list, target_list)):
        # Get the predicted labels
        labels_pred = prediction['out'].argmax(0)
        # Calculate the IoUs
        ious, tps, fps, fns = segmentation_ious_one_image(labels_pred, target, labels=list(idx_to_class_bd.keys()))
        tps_all.append(tps)
        fps_all.append(fps)
        fns_all.append(fns)
        if i % 500 == 0:  # Show progress every 500 images
            print(f'Calculating IoUs: {i}/{len(prediction_list)} images')
    ###### Accumulate IoUs ######
    tps_all = np.array(tps_all).sum(axis=0)
    fps_all = np.array(fps_all).sum(axis=0)
    fns_all = np.array(fns_all).sum(axis=0)
    ious_all = tps_all / (tps_all + fps_all + fns_all)
    # Store the result
    iou_dict = {
        k: {
            'label_name': v,
            'tps': tps_all[i],
            'fps': fps_all[i],
            'fns': fns_all[i],
            'iou': ious_all[i]
        }
        for i, (k, v) in enumerate(idx_to_class.items()) 
    }

    return iou_dict

def segmentation_ious_torchvison(dataloader: DataLoader, model: nn.Module, device: Literal['cuda', 'cpu'],
                                 idx_to_class: Dict[int, str], border_idx:int = None):
    """
    Calculate average precisions with TorchVision models and DataLoader

    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
        List of the predicted bounding boxes

    model : torch.nn.Module
        List of the true bounding boxes

    smoothe : bool
        If True, the precision-recall curve is smoothed to fix the zigzag pattern.

    idx_to_class : Dict[int, str]
        A dict for converting class IDs to class names.

    Returns
    -------
    aps : Dict[int, Dict[Literal['label_name', 'average_precision', 'precision', 'recall'], Any]]
        Calculated average precisions with the label_names and the PR Curve
    """
    # Predict
    target_list = []
    prediction_list = []
    start = time.time()  # For elapsed time
    for i, (imgs, targets) in enumerate(dataloader):
        imgs_gpu = imgs.to(device)
        model.eval()  # Set the evaluation mode
        with no_grad():  # Avoid memory overflow
            predictions = model(imgs_gpu)
        # Store the result
        target_list.extend([target for target in targets])
        prediction_list.extend([{'out': out} for out in predictions['out']])
        if i%100 == 0:  # Show progress every 100 images
            print(f'Prediction for mAP: {i}/{len(dataloader)} batches, elapsed_time: {time.time() - start}')
    ious_all = segmentation_ious(prediction_list, target_list, idx_to_class, border_idx)
    return ious_all

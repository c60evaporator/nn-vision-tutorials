from typing import Dict, List, Literal
import torch
from torch import nn, Tensor, no_grad
from torchvision import ops
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import copy
import time

def iou_object_detection(box_pred, label_pred, boxes_true, labels_true,
                         match_label=True):
    """
    Calculate IoU of object detection
    https://learnopencv.com/intersection-over-union-iou-in-object-detection-and-segmentation/
    """
    # If the label should be matched
    if match_label:
        boxes_gt = []
        labels_gt = []
        for box_true, label_true in zip(boxes_true, labels_true):
            if label_true == label_pred:
                boxes_gt.append(box_true)
                labels_gt.append(label_true)
    # If the label should NOT be matched
    else:
        boxes_gt = copy.deepcopy(boxes_true)
        labels_gt = copy.deepcopy(labels_true)

    # Calculate IoU with every ground truth bbox
    ious = []
    for box_true, label_true in zip(boxes_gt, labels_gt):
        box_true = box_true.view(1, -1)
        box_pred = box_pred.view(1, -1)
        iou = float(ops.box_iou(box_true, box_pred))
        ious.append(iou)

    # Extract max IoU
    if len(ious) > 0:
        max_iou = max(ious)
    else:
        max_iou = 0.0
    return max_iou


def extract_cofident_boxes(scores, boxes, labels, score_threshold):
    """Extract bounding boxes whose score > score_threshold"""
    boxes_confident = []
    labels_confident = []
    scores_confident = []
    for score, box, label in zip(scores, boxes.tolist(), labels):
        if score > score_threshold:
            labels_confident.append(label)
            boxes_confident.append(Tensor(box))
            scores_confident.append(score)
    return boxes_confident, labels_confident, scores_confident

def _average_precision(iou_mat):
    mappings = torch.zeros_like(iou_mat)
    gt_count, pr_count = iou_mat.shape
    
    #first mapping (max iou for first pred_box)
    if not iou_mat[:,0].eq(0.).all():
        # if not a zero column
        mappings[iou_mat[:,0].argsort()[-1],0] = 1

    for pr_idx in range(1,pr_count):
        # Sum of all the previous mapping columns will let 
        # us know which gt-boxes are already assigned
        not_assigned = torch.logical_not(mappings[:,:pr_idx].sum(1)).long()

        # Considering unassigned gt-boxes for further evaluation 
        targets = not_assigned * iou_mat[:,pr_idx]

        # If no gt-box satisfy the previous conditions
        # for the current pred-box, ignore it (False Positive)
        if targets.eq(0).all():
            continue

        # max-iou from current column after all the filtering
        # will be the pivot element for mapping
        pivot = targets.argsort()[-1]
        mappings[pivot,pr_idx] = 1
    return mappings

def average_precisions(predictions_list: List[Dict[Literal['boxes', 'labels', 'scores'], Tensor]],
                       targets_list: List[Dict[Literal['boxes', 'labels', 'scores'], Tensor]],
                       idx_to_class: Dict[int, str],
                       iou_threshold=0.5, score_threshold=None):
    """Calculate average precisions"""
    if score_threshold is None:
        score_threshold = 0.0
    # List for storing scores
    labels_pred_all = []
    scores_all = []
    ious_all = []
    correct_all = []
    ###### Calculate IoU ######
    # Loop of images
    for i, (prediction, target) in enumerate(zip(predictions_list, targets_list)):
        # Get predicted bounding boxes
        boxes_pred = prediction['boxes'].cpu().detach()
        labels_pred = prediction['labels'].cpu().detach().numpy()
        labels_pred = np.where(labels_pred >= max(idx_to_class.keys()),-1, labels_pred)  # Modify labels to 0 if the predicted labels are background
        scores_pred = prediction['scores'].cpu().detach().numpy()
        # Get true bounding boxes
        boxes_true = target['boxes']
        labels_true = target['labels']
        #labelnames_true = [idx_to_class[label.item()] for label in labels_true]
        # Extract predicted boxes whose score > score_threshold
        boxes_confident, labels_confident, scores_confident = extract_cofident_boxes(
                scores_pred, boxes_pred, labels_pred, score_threshold)
        # Calculate IoU
        ious_confident = [
            iou_object_detection(box_pred, label_pred, boxes_true, labels_true)
            for box_pred, label_pred in zip(boxes_confident, labels_confident)
        ]
        # IoU thresholding
        iou_judgement = np.where(np.array(ious_confident) > iou_threshold, True, False).tolist()
        # Store the data on DataFrame
        labels_pred_all.extend(labels_confident)
        scores_all.extend(scores_confident)
        ious_all.extend(ious_confident)
        correct_all.extend(iou_judgement)
        if i % 500 == 0:  # Show progress every 500 images
            print(f'Calculating IoU: {i}/{len(predictions_list)}')
    ###### Calculate Average Precision ######
    # Loop of predicted labels
    for label_pred in sorted(set(labels_pred_all)):
        label_name = idx_to_class[label_pred]
        label_indices = np.where(np.array(labels_pred_all) == label_pred)
        scores_label = np.array(scores_all)[label_indices]
        ious_label = np.array(ious_all)[label_indices]
        correct_label = np.array(correct_all)[label_indices]

def average_precisions_torchvison(dataloader: DataLoader, idx_to_class: Dict[int, str],
                                  model: nn.Module, device: Literal['cuda', 'cpu'],
                                  iou_threshold=0.5, score_threshold=None):
    """Calculate average precisions with TorchVision models and DataLoader"""
    avg_precisions = {}
    # Predict
    targets_list = []
    predictions_list = []
    start = time.time()  # For elapsed time
    for i, (imgs, targets) in enumerate(dataloader):
        imgs_gpu = [img.to(device) for img in imgs]
        model.eval()  # Set the evaluation mode
        with no_grad():  # Avoid memory overflow
            predictions = model(imgs_gpu)
        # Store the result
        targets_list.extend(targets)
        predictions_list.extend(predictions)
        if i%100 == 0:  # Show progress every 100 images
            print(f'Prediction for mAP: {i}/{len(dataloader)}, elapsed_time: {time.time() - start}')
    aps = average_precisions(predictions_list, targets_list, idx_to_class, iou_threshold, score_threshold)
    return aps


def mean_average_precision_torchvison(dataloader: DataLoader, idx_to_class: Dict[int, str],
                                      model: nn.Module, device: Literal['cuda', 'cpu'],
                                      iou_threshold=0.5, score_threshold=0.5):
    avg_precisions = average_precisions(dataloader, idx_to_class, iou_threshold, score_threshold)
    mean_average_precision = np.mean(avg_precisions.values())
    return mean_average_precision
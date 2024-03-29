import torch
from torchvision import ops
from torchvision.utils import draw_bounding_boxes
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import copy
import math
import numpy as np

def get_object_detection_iou(box_pred, label_pred, boxes_true, labels_true,
            match_label=True):
    """
    Calculate IoU
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

def show_bounding_boxes(image, boxes, labels=None, idx_to_class=None,
                        colors=None, fill=False, width=1,
                        font=None, font_size=None,
                        ax=None):
    """
    Show the image with the segmentation.

    Parameters
    ----------
    image : torch.Tensor (C x H x W)
        Input image
    boxes : torch.Tensor (N, 4)
        Bounding boxes with Torchvision object detection format
    labels : List[str]
        Target labels of the bounding boxes
    idx_to_class : Dict[int, str]
        A dict for converting class IDs to class names.
        If None, class ID is used for the plot
    colors : color or list of colors, optional
        List containing the colors of the boxes or single color for all boxes. The color can be represented as PIL strings e.g. "red" or "#FF00FF", or as RGB tuples e.g. ``(240, 10, 157)``.
        By default, random colors are generated for boxes.
    fill : bool
        If `True` fills the bounding box with specified color.
    width : int
        Width of bounding box.
    font : str
        A filename containing a TrueType font. If the file is not found in this filename, the loader may
        also search in other directories, such as the `fonts/` directory on Windows or `/Library/Fonts/`,
        `/System/Library/Fonts/` and `~/Library/Fonts/` on macOS.
    font_size : int
        The requested font size in points.
    """
    # If ax is None, use matplotlib.pyplot.gca()
    if ax is None:
        ax=plt.gca()
    # Convert class IDs to class names
    if idx_to_class is not None:
        labels = [idx_to_class[int(label.item())] for label in labels]
    # Show All bounding boxes
    print(boxes)
    image_with_boxes = draw_bounding_boxes(image, boxes, labels=labels, colors=colors,
                                           fill=fill, width=width,
                                           font=font, font_size=font_size)
    image_with_boxes = image_with_boxes.permute(1, 2, 0)  # Change axis order from (ch, x, y) to (x, y, ch)
    ax.imshow(image_with_boxes)

def show_pred_true_boxes(image, 
                         boxes_pred, labels_pred,
                         boxes_true, labels_true,
                         idx_to_class = None,
                         color_true = 'green', color_pred = 'red', ax=None,
                         scores=None, score_threshold=0.0, score_decimal=3,
                         calc_iou=False, iou_decimal=3):
    """
    Show the true bounding boxes and the predicted bounding boxes

    Parameters
    ----------
    image : torch.Tensor (C x H x W)
        Input image
    boxes_pred : torch.Tensor (N_boxes_pred, 4)
        Predicted bounding boxes with Torchvision object detection format
    labels_pred : torch.Tensor (N_boxes_pred)
        Predicted labels of the bounding boxes
    boxes_true : torch.Tensor (N_boxes, 4)
        True bounding boxes with Torchvision object detection format
    labels_true : torch.Tensor (N_boxes)
        True labels of the bounding boxes
    idx_to_class : Dict[int, str]
        A dict for converting class IDs to class names.
        If None, class ID is used for the plot
    color_true : str (color name)
        A color for the ture bounding boxes. The color can be represented as PIL strings e.g. "red" or "#FF00FF", or as RGB tuples e.g. ``(240, 10, 157)``.
    color_pred : str (color name)
        A color for the predicted bounding boxes. The color can be represented as PIL strings e.g. "red" or "#FF00FF", or as RGB tuples e.g. ``(240, 10, 157)``.
    scores : torch.Tensor (N_boxes_pred)
        Confidence scores for the predicted bounding boxes.
        
        If None, the confidence scores are not displayed and score_threshold are not applied. 
    score_threshold : float
        A threshold of the confidence score for selecting predicted bounding boxes shown.
        
        If True, predicted bounding boxes whose confidence score is higher than the score_threshold are displayed.

        If False, all predicted bounding boxes are displayed.
    score_decimal : str
        A decimal for the displayed confidence scores.
    calc_iou : True
        If True, IoUs are calculated and shown
    iou_decimal : str
        A decimal for the displayed IoUs.
    """
    # If ax is None, use matplotlib.pyplot.gca()
    if ax is None:
        ax=plt.gca()
    # Convert class IDs to class names
    if idx_to_class is not None:
        labels_pred = [idx_to_class[label.item()] for label in labels_pred]
        labels_true = [idx_to_class[label.item()] for label in labels_true]

    # Display raw image
    img_permuted = image.permute(1, 2, 0)  # Change axis order from (ch, x, y) to (x, y, ch)
    ax.imshow(img_permuted)

    # Display true boxes
    for box_true, label_true in zip(boxes_true.tolist(), labels_true):
        r = patches.Rectangle(xy=(box_true[0], box_true[1]), 
                              width=box_true[2]-box_true[0], 
                              height=box_true[3]-box_true[1], 
                              ec=color_true, fill=False)
        ax.add_patch(r)
        plt.text(box_true[0], box_true[1], label_true, color=color_true, fontsize=8)

    # Extract predicted boxes whose score > score_threshold
    if scores is not None:
        boxes_confident = []
        labels_confident = []
        scores_confident = []
        for score, box, label in zip(scores, boxes_pred.tolist(), labels_pred):
            if score > score_threshold:
                labels_confident.append(label)
                boxes_confident.append(torch.Tensor(box))
                scores_confident.append(score)
        print(f'Confident boxes={boxes_confident}, labels={labels_confident}')
    # Extract all predicted boxes if score is not set
    else:
        boxes_confident = copy.deepcopy(boxes_pred)
        labels_confident = copy.deepcopy(labels_pred)
        scores_confident = [float('nan')] * len(boxes_pred)
    # Calculate IoU
    if calc_iou:
        ious_confident = []
        for box_pred, label_pred in zip(boxes_confident, labels_confident):
            ious_confident.append(get_object_detection_iou(box_pred, label_pred, boxes_true, labels_true))
    else:
        ious_confident = [float('nan')] * len(boxes_pred)
    # Display predicted boxes
    for box_pred, label_pred, score, iou in zip(boxes_confident, labels_confident, scores_confident, ious_confident):
        # Show Rectangle
        r = patches.Rectangle(xy=(box_pred[0], box_pred[1]), 
                              width=box_pred[2]-box_pred[0], 
                              height=box_pred[3]-box_pred[1], 
                              ec=color_pred, fill=False)
        ax.add_patch(r)
        # Show label, score, and IoU
        text_pred = label_pred if isinstance(label_pred, str) else str(int(label_pred))
        if not math.isnan(score):
            text_pred += f', score={round(float(score),score_decimal)}'
        if calc_iou:
            if iou > 0.0:
                text_pred += f', TP, IoU={round(float(iou),iou_decimal)}'
            else:
                text_pred += ', FP'
        ax.text(box_pred[0], box_pred[1], text_pred, color=color_pred, fontsize=8)
    # Return result
    return boxes_confident, labels_confident, scores_confident, ious_confident

def show_predicted_detection_minibatch(imgs, predictions, targets, idx_to_class,
                                       max_displayed_images=None, score_threshold=0.5):
    """
    Show predicted minibatch images with bounding boxes.

    Parameters
    ----------
    imgs : List[torch.Tensor (C x H x W)]
        List of the images which are standardized to [0, 1]
    
    predictions : Dict[str, Any] (TorchVision Format)
        List of the prediction result (including 'boxes', 'labels' and 'scores')
    
    targets : Dict[str, Any] (TorchVision Format)
        List of the ground truths (including 'boxes', 'labels' and 'scores')
    
    idx_to_class : Dict[int, str]
        A dict for converting class IDs to class names.
        If None, class ID is used for the plot
    """
    for i, (img, prediction, target) in enumerate(zip(imgs, predictions, targets)):
        img = (img*255).to(torch.uint8)  # Change from float[0, 1] to uint[0, 255]
        boxes = prediction['boxes'].cpu().detach()
        labels = prediction['labels'].cpu().detach().numpy()
        labels = np.where(labels>=len(idx_to_class),-1, labels)  # Modify labels to 0 if the predicted labels are background
        scores = prediction['scores'].cpu().detach().numpy()
        print(f'idx={i}')
        print(f'labels={labels}')
        print(f'scores={scores}')
        print(f'boxes={boxes}')
        # Show all bounding boxes
        show_bounding_boxes(img, boxes, labels=labels, idx_to_class=idx_to_class)
        plt.title('All bounding boxes')
        plt.show()
        # Show Pred bounding boxes whose confidence score > score_threshold with True boxes
        boxes_true = target['boxes']
        labels_true = target['labels']
        boxes_confident, labels_confident, scores_confident, ious_confident = \
            show_pred_true_boxes(img, boxes, labels, boxes_true, labels_true,
                                idx_to_class=idx_to_class,
                                scores=scores, score_threshold=score_threshold,
                                calc_iou=True)
        plt.title('Confident bounding boxes')
        plt.show()
        if max_displayed_images is not None and i >= max_displayed_images - 1:
            break

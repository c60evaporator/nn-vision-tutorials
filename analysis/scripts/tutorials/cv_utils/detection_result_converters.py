import torch
from cv_utils.detection_conversion_utils import convert_bbox_centerxywh_to_xyxy

def convert_yolov5_hub_result(yolov5_results):
    """
    Convert YOLOv5 PyTorch Hub results to TorchVision object detection prediction format

    from: {'pred': [Tensor([[xmin1, ymin1, xmax1, ymax1, confidence1, labelindex1], [xmin2,..],..]])]}

    to: [{'boxes': Tensor([[xmin1, ymin1, xmax1, ymax1],..]), 'labels': Tensor([labelindex1,..]), 'scores': Tensor([confidence1,..])}]

    Parameters
    ----------
    yolov5_results : dict
        YOLOv5 PyTorch Hub results
    
        {'pred': [Tensor([[xmin1, ymin1, xmax1, ymax1, confidence1, ], [xmin2,..],..]])]}
    """
    tv_pred = [{
        'boxes': pred[:,:4],
        'labels': pred[:,5],
        'scores': pred[:,4],
    } for pred in yolov5_results.pred]
    return tv_pred

def convert_yolov8_hub_result(yolov8_results):
    """
    Convert YOLOv8 ultralytics YOLO results to TorchVision object detection prediction format

    from: [{'boxes': {'data': Tensor([[xmin1, ymin1, xmax1, ymax1, confidence1, labelindex1], [xmin2,..],..]])}}]

    to: [{'boxes': Tensor([[xmin1, ymin1, xmax1, ymax1],..]), 'labels': Tensor([labelindex1,..]), 'scores': Tensor([confidence1,..])}]

    Parameters
    ----------
    yolov8_results : dict
        YOLOv8 ultralytics YOLO results
    
        [{'boxes': {'data': Tensor([[xmin1, ymin1, xmax1, ymax1, confidence1, labelindex1], [xmin2,..],..]])}}]
    """
    tv_pred = [{
        'boxes': result.boxes.data[:,:4],
        'labels': result.boxes.data[:,5],
        'scores': result.boxes.data[:,4],
    } for result in yolov8_results]
    return tv_pred

def convert_detr_hub_result(detr_results, img_sizes, same_img_size, prob_threshold=None):
    """
    Convert DETR PyTorch Hub results to TorchVision object detection prediction format

    from: 
        
        {'pred_logits': Tensor([[[class1_logit1, class2_logit1],..],..]), 'pred_boxes': Tensor([[[nxmin1, nymin1, nxmax1, nymax1],..],..]} --- If same_img_size is True

        or

        ['pred_logits': Tensor([[[class1_logit1, class2_logit1],..]]), {'pred_logits': Tensor([[[nxmin1, nymin1, nxmax1, nymax1],..]])} --- If same_img_size is False

    to: [{'boxes': Tensor([[xmin1, ymin1, xmax1, ymax1],..]), 'labels': Tensor([labelindex1,..]), 'scores': Tensor([confidence1,..])}]

    Parameters
    ----------
    detr_results : dict
        DETR PyTorch Hub results
    
    img_sizes : List[Tensor] or Tensor
        Image sizes (N, h, w)
    
    same_img_size : True
        If True, the image sizes are the same. If False, the image sizes are different.

    prob_threshold : True
        A threshold for the max class probability. Bounding boxes whose max class probability is higher than the threshold are returend. If None, all bounding boxes are returned.
    """
    if same_img_size:
        detr_results_dst = [detr_results]
    else:
        detr_results_dst = detr_results

    tv_pred = []

    for detr_result, img_size in zip(detr_results_dst, img_sizes):
        # Get the bounding boxes whose whose max class probability is higher than the threshold
        ps = detr_result["pred_logits"].softmax(-1)[0, :, :-1]
        extracted  = ps.max(-1).values > prob_threshold
        ps_selected = ps[extracted]
        boxes_selected = detr_result["pred_boxes"][0, extracted]
        # Get labels
        labels = torch.argmax(ps_selected, dim=1)
        # バウンディングボックスの座標計算
        h, w = img_size
        boxes = [convert_bbox_centerxywh_to_xyxy(*box) for box in boxes_selected]
        boxes = torch.tensor(boxes) * torch.tensor([w, h, w, h], dtype=float)

        tv_pred.append({
            'boxes': boxes,
            'labels': labels,
            'scores': ps_selected.max(-1).values  # Use max probability instead of confidence score
        })

    return tv_pred

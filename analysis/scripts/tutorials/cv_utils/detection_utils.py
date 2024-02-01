import torch

CLASS_TO_IDX_VOC = {
    'person': 0,
    'bird': 1,
    'cat': 2,
    'cow': 3,
    'dog': 4,
    'horse': 5,
    'sheep': 6,
    'aeroplane': 7,
    'bicycle': 8,
    'boat': 9,
    'bus': 10,
    'car': 11,
    'motorbike': 12,
    'train': 13,
    'bottle': 14,
    'chair': 15,
    'diningtable': 16,
    'pottedplant': 17,
    'sofa': 18,
    'tvmonitor': 19
    }

# Convert [x_min, y_min, w, h] to [xmin, ymin, xmax, ymax]
def convert_bbox_xywh_to_xyxy(x_min, y_min, w, h):
    """Convert [x_min, y_min, w, h] to [xmin, ymin, xmax, ymax]"""
    return [x_min, y_min, x_min + w, y_min + h]

# Convert [x_c, y_c, w, h] to [xmin, ymin, xmax, ymax]
def convert_bbox_centerxywh_to_xyxy(x_c, y_c, w, h):
    """Convert [x_c, y_c, w, h] to [xmin, ymin, xmax, ymax]"""
    return [x_c - 0.5 * w, y_c - 0.5 * h, x_c + 0.5 * w, y_c + 0.5 * h]

def target_transform_to_torchvision(target, in_format, class_to_idx=None):
    """
    Transform target data to adopt to TorchVision object detection format

    {'boxes': Tensor([[xmin1, ymin1, xmax1, ymax1],..]), 'labels': Tensor([labelindex1, labelindex2,..])}

    Parameters
    ----------
    target : {dict, list}
        Input target data (dict or list that includes bounding boxes and labels)

    in_format : {'pascal_voc', 'coco', 'yolo'}
        Annotation format of the input target data.

        'pascal_voc': {'annotation': {'object': [{'bndbox': {'xmin':xmin1, 'ymin':ymin1, 'xmax':xmax1, 'ymax':ymax1}, 'name': labelname1},..]}}

        'coco': [{'bbox': [xmin1, ymin1, w1, h1], 'category_id': labelindex1,..},..]
    
    class_to_idx : dict
        A dict which convert class name to class id. Only necessary for 'pascal_voc' format.
    """
    # From Pascal VOC Object detection format
    if in_format == 'pascal_voc':
        objects = target['annotation']['object']
        box_keys = ['xmin', 'ymin', 'xmax', 'ymax']
        boxes = [[int(obj['bndbox'][k]) for k in box_keys] for obj in objects]
        boxes = torch.tensor(boxes)
        # Get labels
        labels = [class_to_idx[obj['name']] for obj in objects]
        labels = torch.tensor(labels)
    # From COCO format
    elif in_format == 'coco':
        boxes = [[int(k) for k in convert_bbox_xywh_to_xyxy(*obj['bbox'])]
             for obj in target]
        boxes = torch.tensor(boxes)
        # Get labels
        labels = [obj['category_id'] for obj in target]
        labels = torch.tensor(labels)
    # Make a target dict whose keys are 'boxes' and 'labels'
    target = {'boxes': boxes, 'labels': labels}
    return target

def resize_target(target, src_image, resized_image):
    """
    Resize the target in accordance with the image Resize

    Parameters
    ----------
    target : Dict
        Target data (Torchvision object detection format)  
        
        {'boxes': Tensor([[xmin1, ymin1, xmax1, ymax1],..]), 'labels': Tensor([labelindex1, labelindex2,..])}

    src_image : Tensor
        An image before resize (h, w)
    
    resized_image : Tensor
        An image after resize (h, w)
    """
    src_size = src_image.size()
    resized_size = resized_image.size()
    h_ratio = resized_size[1] / src_size[1]
    w_ratio = resized_size[2] / src_size[2]

    targets_resize = {
        'boxes': target['boxes'] * torch.tensor([w_ratio, h_ratio, w_ratio, h_ratio], dtype=float),
        'labels': target['labels']
        }

    return targets_resize


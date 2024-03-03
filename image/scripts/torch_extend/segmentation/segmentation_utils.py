import torch

def target_transform_to_torchvision(target, in_format):
    """
    Transform target data to adopt to TorchVision semantic segmentation format

    {'boxes': Tensor([[xmin1, ymin1, xmax1, ymax1],..], 'labels': Tensor([labelindex1, labelindex2,..])}

    Parameters
    ----------
    target : {dict, list}
        Input target data (dict or list that includes bounding boxes and labels)

    in_format : {'pascal_voc', 'coco', 'yolo'}
        Annotation format of the input target data.

        'pascal_voc': {'annotation': {'object': [{'bndbox': {'xmin':xmin1, 'ymin':ymin1, 'xmax':xmax1, 'ymax':ymax1}, 'name': labelname1},..]}}

        'coco': [{'bbox': [xmin1, ymin1, w1, h1], 'category_id': labelindex1,..},..]

        'yolo':
    
    class_to_idx : {'coco', 'pascal_voc'}
        A dict which convert class name to class id. Only necessary for 'pascal_voc' format.
    """
    # From Pascal VOC Object detection format
    if in_format == 'pascal_voc':
        masks = target
    # From COCO format
    elif in_format == 'coco':
        pass
    # Make a target dict whose keys are 'masks'
    target = {'masks': masks}
    return target

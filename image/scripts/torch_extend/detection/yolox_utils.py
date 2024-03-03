from typing import Tuple, Dict, Any
import time
import numpy as np
import torch
from PIL.Image import Image

from yolox.data.data_augment import preproc
from yolox.utils import postprocess
from yolox.exp import Exp

def val_transform_to_yolox(img: Image, input_size: Tuple, swap=(2, 0, 1)):
    """
    Validation tranform from Torch Dataloader to YOLOX
    
    https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/data/data_augment.py#L213
    """
    img_dst = np.array(img)  # Convert to numpy.ndarray
    img_dst = img_dst[:,:,::-1]  # Convert RGB to BGR (Comply with cv2 format)
    img_dst, _ = preproc(img_dst, input_size, swap)  # 
    img_dst = torch.from_numpy(img_dst).unsqueeze(0)
    img_dst = img_dst.float()
    return img_dst

def inference(img: torch.Tensor, model: torch.nn.Module, experiment: Exp,
              conf_threshold=None, nms_threshold=None):
    """Run YOLOX Inference"""
    conf_thre = experiment.test_conf if conf_threshold is None else conf_threshold
    nms_thre = experiment.nmsthre if nms_threshold is None else nms_threshold
    with torch.no_grad():
        t0 = time.time()
        outputs = model(img)
        outputs = postprocess(
            outputs, experiment.num_classes, conf_thre,
            nms_thre, class_agnostic=True
        )
        if outputs[0] is None:
            outputs[0] = torch.zeros(size=(0, 7))
        print("Infer time: {:.4f}s".format(time.time() - t0))
    return outputs

def get_img_info(img: Image, input_size: Tuple):
    """Get YOLOX img_info"""
    img_info = {"id": 0}
    width, height = img.size
    img_info["height"] = height
    img_info["width"] = width
    img_info["raw_img"] = np.array(img)[:,:,::-1]
    img_info["ratio"] = min(input_size[0] / height, input_size[1] / width)
    return img_info

def convert_yolox_result_to_torchvision(yolox_results: Dict[str, Any], imgs_info: Dict[str, Any]):
    """
    Convert YOLOX demo results to TorchVision object detection prediction format

    from: [Tensor([[xmin1, ymin1, xmax1, ymax1, confidenceA1, confidenceB1, labelindex1], [xmin2,..],..]])]

    to: [{'boxes': Tensor([[xmin1, ymin1, xmax1, ymax1],..]), 'labels': Tensor([labelindex1,..]), 'scores': Tensor([confidence1,..])}]

    Parameters
    ----------
    yolox_results : dict
        YOLOX inference results (https://github.com/Megvii-BaseDetection/YOLOX/blob/main/tools/demo.py#L132)

        [Tensor([[xmin1, ymin1, xmax1, ymax1, confidenceA1, confidenceB1, labelindex1], [xmin2,..],..]])]
    
    imgs_info : dict
        Image information that is gathered by `get_img_info()` method.
    """
    tv_pred = [{
        'boxes': outputs[0][:,:4] / img_info["ratio"],
        'labels': outputs[0][:, 6],
        'scores': outputs[0][:, 4] * outputs[0][:, 5],
    } for outputs, img_info in zip(yolox_results, imgs_info)]
    return tv_pred

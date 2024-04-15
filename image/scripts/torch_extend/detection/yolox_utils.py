from typing import Tuple, Dict, Any, Literal
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL.Image import Image
import random
import warnings
from loguru import logger
from argparse import Namespace

# For inference
from yolox.data.data_augment import preproc
from yolox.utils import postprocess
from yolox.exp import Exp
# For training
from yolox.core import launch
from yolox.exp import check_exp_value, get_exp
from yolox.utils import configure_module, configure_nccl, configure_omp, get_num_devices

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


#@logger.catch
def _main(exp: Exp, args):
    if exp.seed is not None:
        random.seed(exp.seed)
        torch.manual_seed(exp.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! You may see unexpected behavior "
            "when restarting from checkpoints."
        )

    # set environment variables for distributed training
    configure_nccl()
    configure_omp()
    cudnn.benchmark = True

    trainer = exp.get_trainer(args)
    trainer.train()


def train(exp_file: str, devices: int, batch_size: int,
          fp16: bool, occupy: bool, ckpt: str,
          experiment_name:str=None, name:str=None, dist_backend:str='nccl',
          dist_url:str=None, resume:bool=False, start_epoch:int=None,
          num_machines:int=1, machine_rank:int=0, cache:str=None,
          logger:Literal['tensorboard', 'wandb']='tensorboard', opts:list=[]):
    """
    Run YOLOX training (https://github.com/Megvii-BaseDetection/YOLOX/blob/main/tools/train.py)

    Parameters
    ----------
    exp_file : str
        Path to the Exp file
    devices: int
        Number of the devices for the training
    batch_size: int
        Batch size
    fp16: bool
        Adopting mix precision training
    occupy: bool
        Occupy GPU memory first for training
    ckpt: str
        Path to the pretrained weight or the checkpoint file
    experiment_name: str
        Experiment name
    name: str
        Model name
    dist_backend: str
        Distributed backend
    dist_url: str
        Url used to set up distributed training
    resume: bool
        Resume training
    start_epoch: int
        Resume training start epoch
    num_machines: int
        Num of node for training
    machine_rank: int
        Node rank for multi-node training
    cache: str
        Caching imgs to ram/disk for fast training
    logger: {'tensorboard', 'wandb'}
        Logger to be used for metrics. Implemented loggers include `tensorboard` and `wandb`.
    opts: list
        Modify config options using the command-line
    """
    configure_module()
    args = Namespace(exp_file=exp_file, devices=devices, batch_size=batch_size, 
                     fp16=fp16, occupy=occupy, ckpt=ckpt,
                     experiment_name=experiment_name, name=name, dist_backend=dist_backend, 
                     dist_url=dist_url, resume=resume, start_epoch=start_epoch,
                     num_machines=num_machines, machine_rank=machine_rank, cache=cache, 
                     logger=logger, opts=opts)
    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)
    check_exp_value(exp)

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    num_gpu = get_num_devices() if args.devices is None else args.devices
    assert num_gpu <= get_num_devices()

    if args.cache is not None:
        exp.dataset = exp.get_dataset(cache=True, cache_type=args.cache)

    dist_url = "auto" if args.dist_url is None else args.dist_url
    launch(
        _main,
        num_gpu,
        args.num_machines,
        args.machine_rank,
        backend=args.dist_backend,
        dist_url=dist_url,
        args=(exp, args),
    )

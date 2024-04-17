from typing import Literal
from pathlib import Path
from argparse import Namespace

def train_detection(coco_path: str, device:Literal['cuda', 'cpu'],
                    lr:float=1e-4, lr_backbone:float=1e-5, batch_size:int=2,
                    weight_decay:float=1e-4, epochs:int=300, lr_drop:int=200,
                    clip_max_norm:float=0.1,
                    backbone:str='resnet50', dilation:bool=True, position_embedding:Literal['sine', 'learned']='sine',
                    enc_layers:int=6, dec_layers:int=6, dim_feedforward:int=2048,
                    hidden_dim:int=256, dropout:float=0.1, nheads:int=8,
                    num_queries:int=100,
                    aux_loss:bool=True,
                    set_cost_class:float=1.0, set_cost_bbox:float=5.0, set_cost_giou:float=2.0,
                    bbox_loss_coef:float=5.0,
                    giou_loss_coef:float=2.0, eos_coef:float=0.1,
                    dataset_file:str='coco', coco_panoptic_path:str=None, remove_difficult:bool=True,
                    output_dir:str='', seed:int=42, resume:str='',
                    start_epoch:int=0, num_workers:int=2,
                    world_size:int=1, dist_url:str='env://'):
    """
    Run DETR object detection training (https://github.com/facebookresearch/detr/blob/main/main.py)

    Parameters
    ----------
    coco_path : str
        Path to the Dataset that is used for the training
    device: {'cuda', 'cpu'}
        device to use for training / testing
    lr: float
        Learning rate
    lr_backbone: float
        Learning rate for the backbone layers
    batch_size: int
        Batch size
    weight_decay: float
        Weight decay
    epochs: int
        Number of epochs
    lr_drop: float
        Learning rate drop
    clip_max_norm: float
        Gradient clipping max norm
    
    backbone: str
        Name of the convolutional backbone to use
    dilation: bool
        If true, we replace stride with dilation in the last convolutional block (DC5)
    position_embedding: {'sine', 'learned'}
        Type of positional embedding to use on top of the image features
    
    enc_layers: int
        Number of encoding layers in the transformer
    dec_layers: int
        Number of decoding layers in the transformer
    dim_feedforward: int
        Intermediate size of the feedforward layers in the transformer blocks
    hidden_dim: int
        Size of the embeddings (dimension of the transformer)
    dropout: float
        Dropout applied in the transformer
    nheads: int
        Number of attention heads inside the transformer's attentions
    num_queries: int
        Number of query slots
    
    aux_loss: bool
        Enables auxiliary decoding losses (loss at each layer)
    
    set_cost_class: float
        Class coefficient in the matching cost
    set_cost_bbox: float
        L1 box coefficient in the matching cost
    set_cost_giou: float
        giou box coefficient in the matching cost
    
    mask_loss_coef: float
        ?
    dice_loss_coef: float
        ?
    bbox_loss_coef: float
        ?
    giou_loss_coef: float
        ?
    eos_coef: float
        Relative classification weight of the no-object class
    
    dataset_file: float
        ?
    coco_panoptic_path: float
        ?
    remove_difficult: bool
        ?
    
    output_dir: str
        path where to save, empty for no saving
    seed: int
        Random seed
    resume: str
        resume from checkpoint
    start_epoch: int
        start epoch
    num_workers: int
        ?
    
    world_size: int
        number of distributed processes
    dist_url: str
        url used to set up distributed training
    """
    # Load DETR package
    from main import main
    # Create arguments instance
    args = Namespace(coco_path=coco_path, frozen_weights=None, device=device,
                     lr=lr, lr_backbone=lr_backbone, batch_size=batch_size,
                     weight_decay=weight_decay, epochs=epochs, lr_drop=lr_drop,
                     clip_max_norm=clip_max_norm,
                     backbone=backbone, dilation=dilation, position_embedding=position_embedding,
                     enc_layers=enc_layers, dec_layers=dec_layers, dim_feedforward=dim_feedforward,
                     hidden_dim=hidden_dim, dropout=dropout, nheads=nheads,
                     num_queries=num_queries, pre_norm=False,
                     masks=False,
                     aux_loss=aux_loss,
                     set_cost_class=set_cost_class, set_cost_bbox=set_cost_bbox, set_cost_giou=set_cost_giou,
                     mask_loss_coef=1.0, dice_loss_coef=1.0, bbox_loss_coef=bbox_loss_coef,
                     giou_loss_coef=giou_loss_coef, eos_coef=eos_coef,
                     dataset_file=dataset_file, coco_panoptic_path=coco_panoptic_path, remove_difficult=remove_difficult,
                     output_dir=output_dir, seed=seed, resume=resume,
                     start_epoch=start_epoch, eval=False, num_workers=num_workers,
                     world_size=world_size, dist_url=dist_url)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

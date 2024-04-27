from typing import Any, Callable, List, Tuple, Optional
from typing import Dict, List, Literal
from torch import nn, Tensor, no_grad
from torch.utils.data import DataLoader
from sqlalchemy.orm import Session
from datetime import datetime
import os
import time
import streamlit as st
import numpy as np
import json

from torch_extend.segmentation.metrics import segmentation_ious_one_image
from torch_extend.segmentation.dataset import VOCSegmentationTV, CocoSegmentationTV
import sql.crud as crud
import preprocessing.seg_preprocessing as preproessing
from utils.segmentation.models import SEGMENTATION_MODELS

def segmentation_eval_batch(db: Session, evaluation_id: int, created_at: datetime,
                            image_fps: str, 
                            predictions: Dict[Literal['out', 'aux'], Tensor],
                            targets: List[Tensor],
                            idx_to_class: Dict[int, str], border_idx:int = None):
    """Calculate the average precision of each class label"""
    # List for storing scores
    tps_batch = []
    fps_batch = []
    fns_batch = []
    ###### Calculate IoUs of each image ######
    # Add the border label to idx_to_class
    idx_to_class_bd = {k: v for k, v in idx_to_class.items()}
    if border_idx is not None:
        idx_to_class_bd[border_idx] = 'border'
    # Loop of images
    for i, (prediction, target, image_fp) in enumerate(zip(predictions['out'], targets, image_fps)):
        # Get the predicted labels
        labels_pred = prediction.argmax(0)
        # Calculate the IoUs
        ious, tps, fps, fns = segmentation_ious_one_image(labels_pred, target, labels=list(idx_to_class_bd.keys()))
        ###### Post the image result on the DB ######
        unions = tps + fps + fns
        area_iou = np.nansum(tps) / np.nansum(unions)
        mean_iou = np.nanmean(ious)
        image_evaluation = {
            'img_path': image_fp,
            'img_width': target.size()[0],
            'img_height': target.size()[1],
            'area_iou': area_iou,
            'mean_iou': mean_iou
        }
        db_image_evaluation = crud.create_image_evaluations(db, image_evaluation, evaluation_id, created_at)
        ###### Post the image_label result on the DB ######
        label_image_evaluations = [
            {
                'label_id': k,
                'label_name': v,
                'tp': tps[i],
                'fp': fps[i],
                'fn': fns[i],
                'iou': ious[i]
            }
            for i, (union, (k, v)) in enumerate(zip(unions, idx_to_class.items()))
            if union > 0
        ]
        crud.create_label_image_evaluations(db, label_image_evaluations, db_image_evaluation.image_evaluation_id, created_at)
        # Store the metrics
        tps_batch.append(tps)
        fps_batch.append(fps)
        fns_batch.append(fns)
    ###### Accumulate IoUs ######
    tps_batch = np.array(tps_batch).sum(axis=0)
    fps_batch = np.array(fps_batch).sum(axis=0)
    fns_batch = np.array(fns_batch).sum(axis=0)
    return tps_batch, fps_batch, fns_batch

def segmentation_eval_torchvison(db: Session, evaluation_id, created_at,
                                 dataloader: DataLoader, model: nn.Module, device: Literal['cuda', 'cpu'],
                                 idx_to_class: Dict[int, str], border_idx:int = None):
    """Evaluate the TorchVision model and post the result on the DB"""
    progress_bar = st.progress(0, f'Inference progress')
    # Send the model to GPU
    model.to(device)
    # Lists for store the results
    tps_all = []
    fps_all = []
    fns_all = []
    elapsed_time = 0.0
    # Batch iteration
    for i, (imgs, targets, image_fps) in enumerate(dataloader):
        start = time.perf_counter()  # For elapsed time (Only prediction)
        imgs_gpu = imgs.to(device)
        model.eval()  # Set the evaluation mode
        with no_grad():  # Avoid memory overflow
            predictions = model(imgs_gpu)
        elapsed_time += time.perf_counter() - start
        # Calculate TP, FP, FN of the batch
        tps_batch, fps_batch, fns_batch = segmentation_eval_batch(
            db, evaluation_id, created_at, image_fps,
            predictions, targets, idx_to_class, border_idx)
        tps_all.append(tps_batch)
        fps_all.append(fps_batch)
        fns_all.append(fns_batch)
        if i%100 == 0:  # Show progress every 100 images
            progress_bar.progress(i/len(dataloader), f'Prediction for mIoU: {i}/{len(dataloader)} batches')
    progress_bar.empty()
    tps_all = np.array(tps_all).sum(axis=0)
    fps_all = np.array(fps_all).sum(axis=0)
    fns_all = np.array(fns_all).sum(axis=0)
    unions_all = tps_all + fps_all + fns_all
    ious_all = np.divide(tps_all, unions_all.astype(np.float32), out=np.full((len(tps_all),), np.nan), where=(unions_all!=0))
    mean_iou = np.nanmean(ious_all)

    # Store the result
    evaluation = {
        'area_iou': np.nansum(tps_all) / np.nansum(unions_all),
        'mean_iou': float(mean_iou),
        'tps': json.dumps(tps_all.tolist()),
        'fps': json.dumps(fps_all.tolist()),
        'fns': json.dumps(fns_all.tolist()),
        'unions': json.dumps(unions_all.tolist()),
        'ious': json.dumps(ious_all.tolist()),
        'elapsed_time': elapsed_time
    }
    st.write(evaluation)
    
    crud.update_evaluation(db, evaluation, evaluation_id)

    return evaluation

def get_evaluation_dataset(dataset_format, model_name, num_classes,
                           dataset_root, image_set):
    model_format = SEGMENTATION_MODELS[model_name]['format']
    transform = preproessing.get_transform(model_name)
    target_transform = preproessing.get_target_transform(model_name, num_classes)
    if model_format == 'TorchVision':
        if dataset_format == 'VOC':
            dataset = VOCSegmentationEval(dataset_root, image_set=image_set,
                                          transform=transform, target_transform=target_transform)
        elif dataset_format == 'COCO':
            annFile = f'{os.path.dirname(dataset_root)}/annotations/{image_set}.json'
            dataset = CocoSegmentationEval(root=dataset_root, annFile=annFile,
                                           transform=transform, target_transform=target_transform)
    return dataset, transform, target_transform

class VOCSegmentationEval(VOCSegmentationTV):
    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        image, target = super().__getitem__(index)
        image_path = self.images[index]
        return image, target, image_path

class CocoSegmentationEval(CocoSegmentationTV):
    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        image, target = super().__getitem__(index)
        image_path = self.coco.loadImgs(id)[0]["file_name"]
        return image, target, image_path
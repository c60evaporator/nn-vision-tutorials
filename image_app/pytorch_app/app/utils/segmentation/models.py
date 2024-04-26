from enum import Enum
import os
import re
import glob
import torch
from torchvision import models
import streamlit as st

from torch_extend.segmentation.metrics import segmentation_ious_one_image
import preprocessing.seg_preprocessing as preproessing

SEGMENTATION_MODELS = {
    'FCN': {'format': 'TorchVision', 'pattern': '.*?(fcn).*'},
    'DeepLabV3': {'format': 'TorchVision', 'pattern': '.*?(deeplabv3).*'},
    'LRASPP': {'format': 'TorchVision', 'pattern': '.*?(lraspp).*'},
}


def list_seg_models(modelformat):
    """List up the segmentation model names based on the designated format"""
    return [k for k, v in SEGMENTATION_MODELS.items() if v['format'] == modelformat]

def list_seg_weights(weight_dir, model_name):
    """List up the weights in the designated directory"""
    if model_name not in SEGMENTATION_MODELS.keys():
        raise Exception("The model name doesn't exist in TorchVision models")
    # List up weight files
    extensions = ['prm', 'pt', 'pth']
    weight_paths = []
    for extension in extensions:
        weight_paths.extend(glob.glob(f'{weight_dir}/*.{extension}'))
    # Search the weight based on the model name
    pattern = SEGMENTATION_MODELS[model_name]['pattern']
    model_weights = [weight for weight in weight_paths if re.match(pattern, os.path.basename(weight))]
    return model_weights

def load_seg_model(model_name, weight_path):
    """Load the model on the"""
    if model_name not in SEGMENTATION_MODELS.keys():
        raise Exception("The model name doesn't exist in TorchVision models")
    # Load TorchVision model
    if SEGMENTATION_MODELS[model_name]['format'] == 'TorchVision':
        params_load = torch.load(weight_path)
        if model_name == 'FCN':
            num_classes = len(params_load['classifier.4.bias'])
            model = models.segmentation.fcn_resnet50(aux_loss=True)
            model.aux_classifier = models.segmentation.fcn.FCNHead(1024, num_classes)
            model.classifier = models.segmentation.fcn.FCNHead(2048, num_classes)
        elif model_name == 'DeepLabV3':
            num_classes = len(params_load['classifier.4.bias'])
            model = models.segmentation.deeplabv3_resnet50()
            model.aux_classifier = models.segmentation.fcn.FCNHead(1024, num_classes)
            model.classifier = models.segmentation.deeplabv3.DeepLabHead(2048, num_classes)
            model.aux_classifier = None  # Disable aux_classifier to avoid an error
        elif model_name == 'LRASPP':
            num_classes = len(params_load['classifier.high_classifier.bias'])
            model = models.segmentation.lraspp_mobilenet_v3_large()
            low_channels = model.classifier.low_classifier.in_channels
            inter_channels = model.classifier.high_classifier.in_channels
            model.classifier.low_classifier = torch.nn.Conv2d(low_channels, num_classes, 1)
            model.classifier.high_classifier = torch.nn.Conv2d(inter_channels, num_classes, 1)
        model.load_state_dict(params_load)
        model.eval()
    return model, num_classes


def load_seg_default_models(weight_dir, weight_names):
    if 'seg_models' not in st.session_state:
        progress_bar = st.progress(0, f'Loading {len(weight_names)} models')
        models = []
        for i, weight_name in enumerate(weight_names):
            for model_idx, (k, v) in enumerate(SEGMENTATION_MODELS.items()):
                if re.match(v['pattern'], weight_name):
                    model_name = k
                    break
                if model_idx == len(SEGMENTATION_MODELS) - 1:
                    raise Exception(f"The weight {weight_name} doesn't match any model type")
            weight_path = f'{weight_dir}/{weight_name}'
            model, num_classes = load_seg_model(model_name, weight_path)
            models.append({
                'model_name': model_name,
                'num_classes': num_classes,
                'weight_name': weight_name,
                'model': model
            })
            progress_bar.progress((i + 1)/len(weight_names), text=f'Loading models {i}/{len(weight_names)}')
        st.session_state['seg_models'] = models
        progress_bar.empty()

def inference(image, model):
    img_transformed = preproessing.transform[model['model_name']](image)
    # Inference of TorchVision model
    if SEGMENTATION_MODELS[model['model_name']]['format'] == 'TorchVision':
        prediction = model['model'](img_transformed.unsqueeze(0))['out']
        predicted_labels = prediction.argmax(1).cpu().detach()
    return predicted_labels

def inference_and_score(image, model, idx_to_class, mask, border_idx):
    # Inference
    predicted_labels = inference(image, model)
    # Add the border label to idx_to_class
    idx_to_class_bd = {k: v for k, v in idx_to_class.items()}
    if border_idx is not None:
        idx_to_class_bd[border_idx] = 'border'
    # Calculate the metrics
    ious, tps, fps, fns = segmentation_ious_one_image(predicted_labels, mask, labels=list(idx_to_class_bd.keys()))
    iou_dict = {
        k: {
            'label_name': v,
            'tp': tps[i],
            'fp': fps[i],
            'fn': fns[i],
            'iou': ious[i]
        }
        for i, (k, v) in enumerate(idx_to_class.items()) 
    }
    return predicted_labels, iou_dict
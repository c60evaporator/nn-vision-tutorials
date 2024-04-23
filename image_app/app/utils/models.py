from enum import Enum
import os
import re
import glob
import torch
from torchvision import models
import streamlit as st

SEGMENTATION_MODELS = {
    'FCN': {'format': 'TorchVision', 'pattern': '.*?(fcn).*'},
    'DeepLabV3': {'format': 'TorchVision', 'pattern': '.*?(deeplabv3).*'},
    'LRASPP': {'format': 'TorchVision', 'pattern': '.*?(lraspp).*'},
}


def list_seg_models(modelformat):
    """List up the segmentation model names based on the designated format"""
    return [k for k, v in SEGMENTATION_MODELS.items() if v['format'] == modelformat]

def list_seg_weights(dir_path, model_name):
    """List up the weights in the designated directory"""
    if model_name not in SEGMENTATION_MODELS.keys():
        raise Exception("The model name doesn't exist in TorchVision models")
    # List up weight files
    extensions = ['prm', 'pt', 'pth']
    weight_paths = []
    for extension in extensions:
        weight_paths.extend(glob.glob(f'{dir_path}/*.{extension}'))
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
        if SEGMENTATION_MODELS[model_name] == 'FCN':
            model = models.segmentation.fcn_resnet50()
        elif SEGMENTATION_MODELS[model_name] == 'DeepLabV3':
            model = models.segmentation.deeplabv3_resnet50()
        elif SEGMENTATION_MODELS[model_name] == 'LRASPP':
            model = models.segmentation.lraspp_mobilenet_v3_large()
        params_load = torch.load(weight_path)
        model.load_state_dict(params_load)
    return model
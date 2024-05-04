from enum import Enum
import os
import re
import glob
import torch
from torchvision import models
from torchvision.transforms import Normalize
import albumentations as A
import streamlit as st
import numpy as np
from PIL import Image

from torch_extend.segmentation.metrics import segmentation_ious_one_image
import config.segmentation.preprocessing as preprocessing

SEGMENTATION_MODELS = {
    'FCN_ResNet50': {'format': 'TorchVision', 'group': 'FCN', 'pattern': '.*?(fcn|FCN).*?(resnet50|ResNet50).*'},
    'FCN_ResNet101': {'format': 'TorchVision', 'group': 'FCN', 'pattern': '.*?(fcn|FCN).*?(resnet101|ResNet101).*'},
    'DeepLabV3_MobileNet': {'format': 'TorchVision', 'group': 'DeepLabV3', 'pattern': '.*?(deeplabv3|DeepLabV3).*?(mobilenet|MobileNet).*'},
    'DeepLabV3_ResNet50': {'format': 'TorchVision', 'group': 'DeepLabV3', 'pattern': '.*?(deeplabv3|DeepLabV3).*?(resnet50|ResNet50).*'},
    'DeepLabV3_ResNet101': {'format': 'TorchVision', 'group': 'DeepLabV3', 'pattern': '.*?(deeplabv3|DeepLabV3).*?(resnet101|ResNet101).*'},
    'LRASPP_MobileNet': {'format': 'TorchVision', 'group': 'LRASPP', 'pattern': '.*?(lraspp|LRASPP).*?(mobilenet|MobileNet).*'},
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


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
        if SEGMENTATION_MODELS[model_name]['group'] == 'FCN':
            num_classes = len(params_load['classifier.4.bias'])
            if model_name == 'FCN_ResNet50':
                model = models.segmentation.fcn_resnet50(aux_loss=True)
            elif model_name == 'FCN_ResNet101':
                model = models.segmentation.fcn_resnet101(aux_loss=True)
            model.aux_classifier = models.segmentation.fcn.FCNHead(1024, num_classes)
            model.classifier = models.segmentation.fcn.FCNHead(2048, num_classes)
        elif SEGMENTATION_MODELS[model_name]['group'] == 'DeepLabV3':
            num_classes = len(params_load['classifier.4.bias'])
            if model_name == 'DeepLabV3_MobileNet':
                model = models.segmentation.deeplabv3_mobilenet_v3_large(aux_loss=True)
            elif model_name == 'DeepLabV3_ResNet50':
                model = models.segmentation.deeplabv3_resnet50(aux_loss=True)
            elif model_name == 'DeepLabV3_ResNet101':
                model = models.segmentation.fcn_resnet101(aux_loss=True)
            model.aux_classifier = models.segmentation.fcn.FCNHead(1024, num_classes)
            model.classifier = models.segmentation.deeplabv3.DeepLabHead(2048, num_classes)
            model.aux_classifier = None  # Disable aux_classifier to avoid an error
        elif SEGMENTATION_MODELS[model_name]['group'] == 'LRASPP':
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

def _reverse_transform_img(img, transform, albumentations_transform):
    # Denormalize
    trs = transform.transforms if transform is not None else albumentations_transform
    for tr in trs:
        if isinstance(tr, Normalize) or isinstance(tr, A.Normalize):
            img = Normalize(mean=[-mean/std for mean, std in zip(IMAGENET_MEAN, IMAGENET_STD)],
                            std=[1/std for std in IMAGENET_STD])(img)
    # To uint8
    img = (img*255).to(torch.uint8)
    return img

def inference(image, model):
    # Inference of TorchVision model
    if SEGMENTATION_MODELS[model['model_name']]['format'] == 'TorchVision':
        prediction = model['model'](image.unsqueeze(0))['out']
        predicted_labels = prediction.argmax(1).cpu().detach()

    return predicted_labels

def inference_and_score(img_transformed, mask_transformed, model, idx_to_class, border_idx):
    # Inference
    predicted_labels = inference(img_transformed, model)
    # Add the border label to idx_to_class
    idx_to_class_bd = {k: v for k, v in idx_to_class.items()}
    if border_idx is not None:
        idx_to_class_bd[border_idx] = 'border'
    # Calculate the metrics
    ious, tps, fps, fns = segmentation_ious_one_image(predicted_labels, mask_transformed, labels=list(idx_to_class_bd.keys()))
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

def _display_transform(img, transform, albumentations_transform):
    # Denormalize the image
    trs = transform.transforms if transform is not None else albumentations_transform
    for tr in trs:
        if isinstance(tr, Normalize) or isinstance(tr, A.Normalize):
            img_display = Normalize(mean=[-mean/std for mean, std in zip(IMAGENET_MEAN, IMAGENET_STD)],
                                    std=[1/std for std in IMAGENET_STD])(img)
    # Convert the image to PIL image
    img_display = (img_display*255).to(torch.uint8).permute(1, 2, 0).numpy()
    img_display = Image.fromarray(img_display).convert('RGB')
    
    return img_display

def transform_image_ann(image, ann_image, models, idx_to_class):
    imgs_transformed = []
    masks_transformed = []
    imgs_display = []
    for model in models:
        if model is None:
            imgs_transformed.append(None)
            masks_transformed.append(None)
            imgs_display.append(None)
        else:
            # Transform the image
            transform = preprocessing.get_transform(model['model_name'])
            target_transform = preprocessing.get_target_transform(model['model_name'])
            albumentations_transform = preprocessing.get_albumentations_transform(model['model_name'])
            if transform is not None and target_transform is not None:
                img_transformed = transform(image)
                mask_transformed = target_transform(ann_image).squeeze(0).long()
                mask_transformed[mask_transformed == 255] = len(idx_to_class)
            elif albumentations_transform is not None:
                img_transformed = albumentations_transform(image=np.array(image))['image']
                mask_transformed = albumentations_transform(image=np.array(image), mask=np.asarray(ann_image).copy())['mask'].squeeze(0).long()
            else:
                raise Exception('Please describe the transform either on `get_target_transform()` or on `get_albumentations_transform()`')
            # Denormalize the image for the display
            img_display = _display_transform(img_transformed, transform, albumentations_transform)
            imgs_transformed.append(img_transformed)
            masks_transformed.append(mask_transformed)
            imgs_display.append(img_display)
    
    return imgs_transformed, masks_transformed, imgs_display

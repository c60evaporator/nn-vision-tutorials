# %%
# https://blog.roboflow.com/how-to-use-segment-anything-model-sam/
# https://www.kaggle.com/code/mrinalmathur/segment-anything-model-tutorial
# Note: SAM model doesn't have classification function https://github.com/facebookresearch/segment-anything/issues/77

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import time
import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from torch_extend.segmentation.display import show_segmentations, show_predicted_segmentation_minibatch
from torch_extend.segmentation.metrics import segmentation_ious_torchvison
from torch_extend.segmentation.dataset import VOCSegmentationTV

SEED = 42
BATCH_SIZE = 2  # Batch size
NUM_EPOCHS = 10  # number of epochs
NUM_DISPLAYED_IMAGES = 10  # number of displayed images
NUM_LOAD_WORKERS = 4  # Number of workers for DataLoader (Multiple workers not work in original dataset)
DEVICE = 'cuda'  # 'cpu' or 'cuda'
DATA_SAVE_ROOT = '/workspace/datasets/object_detection'  # Directory for Saved dataset
PRETRAINED_WEIGHT = '/workspace/pretrained_weights/segmentation/sam_vit_h_4b8939.pth'  # Model Checkpoint download from (https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints)
MODEL_TYPE = 'vit_h'  # Selected from ['vit_h', 'vit_l', 'vit_b']

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CLASS_TO_IDX = {  # https://github.com/NVIDIA/DIGITS/blob/master/examples/semantic-segmentation/pascal-voc-classes.txt
    'background': 0,
    'aeroplane': 1,
    'bicycle': 2,
    'bird': 3,
    'boat': 4,
    'bottle': 5,
    'bus': 6,
    'car': 7,
    'cat': 8,
    'chair': 9,
    'cow': 10,
    'diningtable': 11,
    'dog': 12,
    'horse': 13,
    'motorbike': 14,
    'person': 15,
    'pottedplant': 16,
    'sheep': 17,
    'sofa': 18,
    'train': 19,
    'tvmonitor': 20
    }

# Confirm GPU availability
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if DEVICE == 'cpu':
    device = 'cpu'
# Set random seed
torch.manual_seed(SEED)

###### 1. Create dataset & Preprocessing ######
albumentations_transform = A.Compose([
    A.Resize(height=224, width=224),  # Applied to both image and mask
    #A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),  # Only applied to image
    ToTensorV2(),  # Applied to both image and mask
])
# Define class names
idx_to_class = {v: k for k, v in CLASS_TO_IDX.items()}
num_classes = len(idx_to_class) + 1  # Classification classes + 1 (border)
# Load validation dataset
val_dataset = VOCSegmentationTV(root = f'{DATA_SAVE_ROOT}/VOCdevkit/VOC2012',
                                idx_to_class=idx_to_class, image_set='val',
                                albumentations_transform=albumentations_transform)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_LOAD_WORKERS)


###### 2. Define Model ######
model = sam_model_registry[MODEL_TYPE](checkpoint=PRETRAINED_WEIGHT)
# Send the model to GPU
model.to(device)
mask_generator = SamAutomaticMaskGenerator(model)

# %%
import numpy as np
def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

###### Inference in the first mini-batch ######
# Inference
val_iter = iter(val_loader)
imgs, targets = next(val_iter)

for img in imgs:
    img_gpu = img.permute(1, 2, 0).detach().numpy()
    plt.figure(figsize=(20,20))
    plt.imshow(img_gpu)
    masks = mask_generator.generate(img_gpu)
    show_anns(masks)
    plt.axis('off')
    plt.show()
    plt.show()

# %%

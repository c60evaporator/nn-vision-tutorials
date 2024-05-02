# %% Pascal VOC Segmentation + FPN by smp (ResNext50) + Data Augumentation by Albumentations (https://github.com/qubvel/segmentation_models.pytorch/blob/master/examples/cars%20segmentation%20(camvid).ipynb)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.base import SegmentationModel, SegmentationHead
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
BATCH_SIZE = 16  # Batch size
NUM_EPOCHS = 10  # number of epochs
NUM_DISPLAYED_IMAGES = 10  # number of displayed images
NUM_LOAD_WORKERS = 4  # Number of workers for DataLoader (Multiple workers not work in original dataset)
DEVICE = 'cuda'  # 'cpu' or 'cuda'
DATA_SAVE_ROOT = '/scripts/datasets/object_detection'  # Directory for Saved dataset
PARAMS_SAVE_ROOT = '/scripts/params/segmentation'  # Directory for Saved parameters
FREEZE_PRETRAINED = False  # If True, Freeze pretrained parameters (Transfer learning)
ENCODER = 'resnet50'
ENCODER_WEIGHTS = 'imagenet'

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
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
# Recommended resize_size=520 but other size is available (https://pytorch.org/vision/main/models/generated/torchvision.models.segmentation.fcn_resnet50.html)
# Define preprocessing of the target for VOCSegmentation dataset (https://poutyne.org/examples/semantic_segmentation.html)
albumentations_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),
    A.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=0),
    A.RandomCrop(height=320, width=320, always_apply=True),
    A.GaussNoise(p=0.2),
    A.Perspective(p=0.5),
    A.OneOf(
        [
            A.CLAHE(p=1),
            A.RandomBrightnessContrast(p=1),
            A.RandomGamma(p=1),
        ],
        p=0.9,
    ),
    A.OneOf(
        [
            A.Sharpen(p=1),
            A.Blur(blur_limit=3, p=1),
            A.MotionBlur(blur_limit=3, p=1),
        ],
        p=0.9,
    ),
    A.OneOf(
        [
            A.RandomBrightnessContrast(p=1),
            A.HueSaturationValue(p=1),
        ],
        p=0.9,
    ),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2()
])
# Reverse normalization transform for showing the image
denormalize_transform = transforms.Compose([
    transforms.Normalize(mean=[-mean/std for mean, std in zip(IMAGENET_MEAN, IMAGENET_STD)],
                         std=[1/std for std in IMAGENET_STD])
])
# Define class names
idx_to_class = {v: k for k, v in CLASS_TO_IDX.items()}
num_classes = len(idx_to_class) + 1  # Classification classes + 1 (border)
# Load train dataset from image folder
train_dataset = VOCSegmentationTV(root = f'{DATA_SAVE_ROOT}/VOCdevkit/VOC2012',
                                  idx_to_class=idx_to_class, image_set='train',
                                  albumentations_transform = albumentations_transform)
# Define mini-batch DataLoader
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_LOAD_WORKERS)
# Display images in the first mini-batch
display_iter = iter(train_loader)
imgs, targets = next(display_iter)
for i, (img, target) in enumerate(zip(imgs, targets)):
    img = denormalize_transform(img)
    img = (img*255).to(torch.uint8)  # Change from float[0, 1] to uint[0, 255]
    show_segmentations(img, target, idx_to_class, bg_idx=0, border_idx=len(CLASS_TO_IDX))
# Load validation dataset
val_dataset = VOCSegmentationTV(root = f'{DATA_SAVE_ROOT}/VOCdevkit/VOC2012',
                                  idx_to_class=idx_to_class, image_set='val',
                                  albumentations_transform = albumentations_transform)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_LOAD_WORKERS)

###### 2. Define Model ######
# Load a pretrained network
model = smp.FPN(encoder_name=ENCODER, encoder_weights=ENCODER_WEIGHTS, classes=num_classes, activation='sigmoid')
# Freeze pretrained parameters
if FREEZE_PRETRAINED:
    for param in model.parameters():
        param.requires_grad = False
    in_channels_seghead = model.segmentation_head[0].in_channels
    model.segmentation_head = SegmentationHead(in_channels=in_channels_seghead, 
            out_channels=num_classes, activation='sigmoid', kernel_size=3)
print(model)
# Send the model to GPU
model.to(device)
# Choose parameters to be trained
#for p in model.parameters():
#    print(f'{p.shape} {p.requires_grad}')
params = [p for p in model.parameters() if p.requires_grad]

###### 3. Define Criterion & Optimizer ######
criterion = nn.CrossEntropyLoss()  # Criterion (Cross entropy loss)
optimizer = optim.Adam(model.parameters(), lr=0.0005)  # Optimizer (Adam). Only parameters in the final layer are set.

###### 4. Training ######
def train_batch(imgs: torch.Tensor, targets: torch.Tensor,
                model: SegmentationModel, optimizer: torch.optim.Optimizer, criterion,
                skip_single_batch: bool = False):
    """
    Validate one batch

    Parameters
    ----------
    skip_single_batch : bool
        Should be set to True if 1D BatchNormalization layer is included in the model (e.g. DeepLabV3). If True, a batch with single sample is ignored
    """
    # Skip the calculation if the number of images is 1
    if skip_single_batch and len(imgs) == 1:
        return torch.Tensor([0.0]), 1
    # Send images and labels to GPU
    imgs = imgs.to(device)
    targets = targets.to(device)
    # Calculate the loss
    output = model(imgs)  # Forward (Prediction)
    loss = criterion(output, targets)  # Calculate criterion
    # Update parameters
    optimizer.zero_grad()  # Initialize gradient
    loss.backward()  # Backpropagation (Calculate gradient)
    optimizer.step()  # Update parameters (Based on optimizer algorithm)
    return loss, 0

def validate_batch(val_imgs: torch.Tensor, val_targets: torch.Tensor,
                   model: SegmentationModel, criterion,
                   skip_single_batch: bool = False):
    """
    Validate one batch

    Parameters
    ----------
    skip_single_batch : str
        Should be set to True if 1D BatchNormalization layer is included in the model (e.g. DeepLabV3). If True, a batch with single sample is ignored
    """
    # Skip the calculation if the number of images is 1
    if skip_single_batch and len(val_imgs) == 1:
        return torch.Tensor([0.0]), 1
    # Calculate the loss
    val_imgs = val_imgs.to(device)
    val_targets = val_targets.to(device)
    val_output = model(val_imgs)  # Forward (Prediction)
    val_loss = criterion(val_output, val_targets)  # Calculate criterion
    return val_loss, 0

model.train()  # Set the training mode
losses = []  # Array for storing loss (criterion)
val_losses = []  # Array for validation loss
start = time.time()  # For elapsed time
# Epoch loop
for epoch in range(NUM_EPOCHS):
    # Initialize training metrics
    running_loss = 0.0  # Initialize running loss
    running_acc = 0.0  # Initialize running accuracy
    # Mini-batch loop
    for i, (imgs, targets) in enumerate(train_loader):
        # Training
        loss, skipped = train_batch(imgs, targets, model, optimizer, criterion)
        running_loss += loss.item()  # Update running loss
        if i%100 == 0:  # Show progress every 100 times
            print(f'minibatch index: {i}/{len(train_loader)}, elapsed_time: {time.time() - start}')
    # Calculate average of running losses and accs
    running_loss /= len(train_loader)
    losses.append(running_loss)

    # Calculate validation metrics
    val_running_loss = 0.0  # Initialize validation running loss
    for i, (val_imgs, val_targets) in enumerate(val_loader):
        val_loss, skipped = validate_batch(val_imgs, val_targets, model, criterion)
        val_running_loss += val_loss.item()  # Update running loss
        if i%100 == 0:  # Show progress every 100 times
            print(f'val minibatch index: {i}/{len(val_loader)}, elapsed_time: {time.time() - start}')
    val_running_loss /= len(val_loader)
    val_losses.append(val_running_loss)

    print(f'epoch: {epoch}, loss: {running_loss}, val_loss: {val_running_loss}, elapsed_time: {time.time() - start}')

###### 5. Model evaluation and visualization ######
# Plot loss history
plt.plot(losses, label='Train loss')
plt.plot(val_losses, label='Validation loss')
plt.title('Loss history')
plt.legend()
plt.show()

###### 6. Save the model ######
# Save parameters
params = model.state_dict()
if not os.path.exists(PARAMS_SAVE_ROOT):
    os.makedirs(PARAMS_SAVE_ROOT)
torch.save(params, f'{PARAMS_SAVE_ROOT}/vocseg_fpn_{ENCODER}_aug.prm')

###### Inference in the first mini-batch ######
# Reload parameters
params_load = torch.load(f'{PARAMS_SAVE_ROOT}/vocseg_fpn_{ENCODER}_aug.prm')
model.load_state_dict(params_load)
# Inference
val_iter = iter(val_loader)
imgs, targets = next(val_iter)
model.eval()  # Set the evaluation mode
imgs_gpu = imgs.to(device)
predictions = model(imgs_gpu)
# Reverse normalization for getting the raw image
imgs_display = [denormalize_transform(img) for img in imgs]
# Show the image
predictions = {'out': predictions}
show_predicted_segmentation_minibatch(imgs_display, predictions, targets, idx_to_class,
                                      bg_idx=0, border_idx=len(CLASS_TO_IDX), plot_raw_image=True,
                                      max_displayed_images=NUM_DISPLAYED_IMAGES)

# %% Calculate mean IoU
ious_all = segmentation_ious_torchvison(val_loader, model, device, idx_to_class, border_idx=len(CLASS_TO_IDX), prediction_as_dict=False)
print(pd.DataFrame([v for k, v in ious_all.items()]))

# %%

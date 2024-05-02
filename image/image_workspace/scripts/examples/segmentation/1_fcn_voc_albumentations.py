# %% Pascal VOC Segmentation + FCN (ResNet50) + Fine Tuning (tune=classification_head, transfer=classification_head.cls_logits)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.models.segmentation._utils import _SimpleSegmentationModel
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
BATCH_SIZE = 8  # Batch size
NUM_EPOCHS = 10  # number of epochs
NUM_DISPLAYED_IMAGES = 10  # number of displayed images
NUM_LOAD_WORKERS = 4  # Number of workers for DataLoader (Multiple workers not work in original dataset)
DEVICE = 'cuda'  # 'cpu' or 'cuda'
DATA_SAVE_ROOT = '/scripts/datasets/object_detection'  # Directory for Saved dataset
PARAMS_SAVE_ROOT = '/scripts/params/segmentation'  # Directory for Saved parameters
FREEZE_PRETRAINED = True  # If True, Freeze pretrained parameters (Transfer learning). If False, conduct fine tuning
SAME_IMG_SIZE = True  # Whether the resized image sizes are the same or not
SKIP_SINGLE_BATCH = False  # Should be set to True if 1D BatchNormalization layer is included in the model (e.g. DeepLabV3). If True, a batch with single sample is ignored
REPLACE_WHOLE_CLASSIFIER = False  # If True, all layers in the classifier is replaced for transefer tuning. If False, only the last layer is replaced

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
# Define preprocessing of the image
def collate_fn(batch):  # collate_fn is needed if the sizes of resized images are different
    return tuple(zip(*batch))
# Common transform that is applied to both the image and the mask
albumentations_transform = A.Compose([
    A.Resize(height=224, width=224),  # Applied to both image and mask
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),  # Only applied to image
    ToTensorV2(),  # Applied to both image and mask
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
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_LOAD_WORKERS,
                          collate_fn=None if SAME_IMG_SIZE else collate_fn)
# Display images in the first mini-batch
display_iter = iter(train_loader)
imgs, targets = next(display_iter)
for i, (img, target) in enumerate(zip(imgs, targets)):
    img = denormalize_transform(img)
    img = (img*255).to(torch.uint8)  # Change from float[0, 1] to uint[0, 255]
    show_segmentations(img, target, idx_to_class, bg_idx=0, border_idx=len(CLASS_TO_IDX))
# Load validation dataset
val_dataset = VOCSegmentationTV(root = f'{DATA_SAVE_ROOT}/VOCdevkit/VOC2012',
                                class_to_idx=CLASS_TO_IDX, image_set='val',
                                albumentations_transform = albumentations_transform)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_LOAD_WORKERS,
                        collate_fn=None if SAME_IMG_SIZE else collate_fn)

###### 2. Define Model ######
# Load a pretrained network (https://www.kaggle.com/code/dasmehdixtr/load-finetune-pretrained-model-in-pytorch)
weights = models.segmentation.FCN_ResNet50_Weights.DEFAULT
model = models.segmentation.fcn_resnet50(weights=weights)
# Freeze pretrained parameters
if FREEZE_PRETRAINED:
    for param in model.parameters():
        param.requires_grad = False
# Replace last layers for fine tuning
if REPLACE_WHOLE_CLASSIFIER:
    # Replace all layes in the classifier
    model.aux_classifier = models.segmentation.fcn.FCNHead(1024, num_classes)
    model.classifier = models.segmentation.fcn.FCNHead(2048, num_classes)
else:
    # Replace the last Conv layer of the classifier and the aux_classifier
    inter_channels_classifier = model.classifier[4].in_channels  # Input channels of the classifier
    inter_channels_aux = model.aux_classifier[4].in_channels  # Input channels of the aux_classifier
    model.classifier[4] = nn.Conv2d(inter_channels_classifier, num_classes, 1)  # Last Conv layer of the classifier
    model.aux_classifier[4] = nn.Conv2d(inter_channels_aux, num_classes, 1)  # Last Conv layer of the classifier 
print(model)
# Send the model to GPU
model.to(device)
# Choose parameters to be trained
#for p in model.parameters():
#    print(f'{p.shape} {p.requires_grad}')
params = [p for p in model.parameters() if p.requires_grad]

###### 3. Define Criterion & Optimizer ######
def criterion(inputs, target):
    losses = {}
    for name, x in inputs.items():
        losses[name] = nn.functional.cross_entropy(x, target, ignore_index=255)
    if len(losses) == 1:
        return losses["out"]
    return losses["out"] + 0.5 * losses["aux"]
optimizer = optim.Adam(params, lr=0.0005)  # Optimizer (Adam). Only parameters in the final layer are set.

###### 4. Training ######
def train_batch(imgs: torch.Tensor, targets: torch.Tensor,
                model: _SimpleSegmentationModel, optimizer: torch.optim.Optimizer, criterion,
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
                   model: _SimpleSegmentationModel, criterion,
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

# https://github.com/pytorch/vision/tree/main/references/segmentation
model.train()  # Set the training mode
losses = []  # Array for string loss (criterion)
val_losses = []  # Array for validation loss
start = time.time()  # For elapsed time
# Epoch loop
for epoch in range(NUM_EPOCHS):
    # Initialize training metrics
    model.train()  # Set the training mode
    running_loss = 0.0  # Initialize running loss
    n_skipped_batches = 0
    # Mini-batch loop
    for i, (imgs, targets) in enumerate(train_loader):
        # Training
        if SAME_IMG_SIZE:
            loss, skipped = train_batch(imgs, targets, model, optimizer, criterion, skip_single_batch=SKIP_SINGLE_BATCH)
            running_loss += loss.item()  # Update running loss
            n_skipped_batches += skipped
        else:  # Separate the batch into each sample if the image sizes are different
            n_skipped = 0
            for img, target in zip(imgs, targets):
                loss, skipped = train_batch(img.unsqueeze(0), target.unsqueeze(0), model, optimizer, criterion, skip_single_batch=SKIP_SINGLE_BATCH)
                running_loss += loss.item() / len(imgs)  # Update running loss
                n_skipped += skipped
            n_skipped_batches += n_skipped / len(imgs)
        if i%100 == 0:  # Show progress every 100 times
            print(f'minibatch index: {i}/{len(train_loader)}, elapsed_time: {time.time() - start}')
    # Raise exception if all the batches are skipped
    if len(train_loader) == n_skipped_batches:
        raise Exception('All the batches are skipped. Please make sure the batch size is more than 2')
    # Calculate average of running losses and accs
    running_loss /= len(train_loader) - n_skipped_batches
    losses.append(running_loss)

    # Calculate validation metrics (https://pytorch.org/tutorials/beginner/introyt/trainingyt.html#per-epoch-activity)
    # model.eval()  # Set the evaluation mode (Disabled to calculate the same loss as that of the training)
    val_running_loss = 0.0  # Initialize validation running loss
    n_skipped_batches = 0
    with torch.no_grad():
        for i, (val_imgs, val_targets) in enumerate(val_loader):
            if SAME_IMG_SIZE:
                val_loss, skipped = validate_batch(val_imgs, val_targets, model, criterion, skip_single_batch=SKIP_SINGLE_BATCH)
                val_running_loss += val_loss.item()  # Update running loss
                n_skipped_batches += skipped
            else:  # Separate the batch into each sample if the image sizes are different
                n_skipped = 0
                for val_img, val_target in zip(val_imgs, val_targets):
                    val_loss, skipped = validate_batch(val_img.unsqueeze(0), val_target.unsqueeze(0), model, criterion, skip_single_batch=SKIP_SINGLE_BATCH)
                    val_running_loss += val_loss.item() / len(imgs)  # Update running loss
                    n_skipped += skipped
                n_skipped_batches += n_skipped / len(val_imgs)
            if i%100 == 0:  # Show progress every 100 times
                print(f'val minibatch index: {i}/{len(val_loader)}, elapsed_time: {time.time() - start}')
    val_running_loss /= (len(val_loader) - n_skipped_batches)
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
torch.save(params, f'{PARAMS_SAVE_ROOT}/vocseg_fcn_resnet50.prm')


# %%
###### Inference in the first mini-batch ######
# Reload parameters
params_load = torch.load(f'{PARAMS_SAVE_ROOT}/vocseg_fcn_resnet50.prm')
model.load_state_dict(params_load)
# Inference
val_iter = iter(val_loader)
imgs, targets = next(val_iter)
model.eval()  # Set the evaluation mode
if SAME_IMG_SIZE: # If the image sizes are the same, inference can be conducted with the batch data
    imgs_gpu = imgs.to(device)
    predictions = model(imgs_gpu)
else: # if the image sizes are different, inference should be conducted with one sample
    imgs_gpu = [img.to(device) for img in imgs]
    predictions = [model(img.unsqueeze(0)) for img in imgs_gpu]
    predictions = {'out': [pred['out'][0] for pred in predictions]}
# Reverse normalization for getting the raw image
imgs_display = [denormalize_transform(img) for img in imgs]
# Show the image
show_predicted_segmentation_minibatch(imgs_display, predictions, targets, idx_to_class,
                                      bg_idx=0, border_idx=len(CLASS_TO_IDX), plot_raw_image=True,
                                      max_displayed_images=NUM_DISPLAYED_IMAGES)

# %% Calculate mean IoU
ious_all = segmentation_ious_torchvison(val_loader, model, device, idx_to_class, border_idx=len(CLASS_TO_IDX))
print(pd.DataFrame([v for k, v in ious_all.items()]))

# %%

# %% Pascal VOC + RetinaNet (ResNet50+FPN) + Fine Tuning (tune=classification_head, transfer=classification_head.cls_logits)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.models.detection.retinanet import RetinaNet
from torchvision.datasets import VOCDetection
import matplotlib.pyplot as plt
import time
import os
import math
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from torch_extend.detection.display import show_bounding_boxes, show_predicted_detection_minibatch, show_average_precisions
from torch_extend.detection.target_converter import target_transform_to_torchvision
from torch_extend.detection.metrics import average_precisions_torchvison

SEED = 42
BATCH_SIZE = 2  # Batch size
NUM_EPOCHS = 2  # number of epochs
NUM_DISPLAYED_IMAGES = 10  # number of displayed images
NUM_LOAD_WORKERS = 2  # Number of workers for DataLoader (Multiple workers not work in original dataset)
DEVICE = 'cuda'  # 'cpu' or 'cuda'
DATA_SAVE_ROOT = '/scripts/datasets/object_detection'  # Directory for Saved dataset
PARAMS_SAVE_ROOT = '/scripts/params/object_detection'  # Directory for Saved parameters
FREEZE_PRETRAINED = True  # If True, Freeze pretrained parameters (Transfer learning)
CLASS_TO_IDX = {  # https://github.com/matlab-deep-learning/Object-Detection-Using-Pretrained-YOLO-v2/blob/main/+helper/pascal-voc-classes.txt
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
# Define preprocessing for image
transform = transforms.Compose([
    transforms.ToTensor()  # Convert from range [0, 255] to a torch.FloatTensor in the range [0.0, 1.0]
])
# Define preprocessing for target
target_transform = transforms.Lambda(lambda x: target_transform_to_torchvision(x, in_format='pascal_voc', class_to_idx=CLASS_TO_IDX))
# Load train dataset from image folder
train_dataset = VOCDetection(root = DATA_SAVE_ROOT, year='2012',
                             image_set='train', download=True,
                             transform = transform, target_transform=target_transform)
# Define class names
idx_to_class = {v: k for k, v in CLASS_TO_IDX.items()}
num_classes = len(idx_to_class) + 1  # Classification classes + 1 (background)
# Define collate_fn (Default collate_fn is not available because the shape of `targets` is list[Dict[str, Tensor]]. See https://github.com/pytorch/vision/blob/main/references/detection/utils.py#L203C8-L203C8)
def collate_fn(batch):
    return tuple(zip(*batch))
# Define mini-batch DataLoader
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_LOAD_WORKERS, collate_fn=collate_fn)
# Display images in the first mini-batch
train_iter = iter(train_loader)
imgs, targets = next(train_iter)
for i, (img, target) in enumerate(zip(imgs, targets)):
    img = (img*255).to(torch.uint8)  # Change from float[0, 1] to uint[0, 255]
    boxes, labels = target['boxes'], target['labels']
    show_bounding_boxes(img, boxes, labels=labels, idx_to_class=idx_to_class)
    plt.show()
# Load validation dataset
val_dataset = VOCDetection(root = DATA_SAVE_ROOT, year='2012',
                           image_set='val', download=True,
                           transform = transform, target_transform=target_transform)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_LOAD_WORKERS, collate_fn=collate_fn)

###### 2. Define Model ######
# Load a pretrained model
model = models.detection.retinanet_resnet50_fpn(pretrained=True)
# Freeze pretrained parameters
if FREEZE_PRETRAINED:
    for param in model.parameters():
        param.requires_grad = False
# Modify the classification_head.cls_logits
num_anchors = model.head.classification_head.num_anchors  # Number of anchors
model.head.classification_head.num_classes = num_classes
cls_logits = nn.Conv2d(256, num_anchors*num_classes, kernel_size=3, stride=1, padding=1)
nn.init.normal_(cls_logits.weight, std=0.01)  # Initialize the weight
nn.init.constant_(cls_logits.bias, -math.log((1-0.01)/0.01))  # Initialize the bias
model.head.classification_head.cls_logits = cls_logits
print(model)
# Unfreeze parameters in classification_head
for param in model.head.classification_head.parameters():
    param.requires_grad = True
# Send the model to GPU
model.to(device)
# Choose parameters to be trained
params = [p for p in model.parameters() if p.requires_grad]

###### 3. Define Criterion & Optimizer ######
def criterion(loss_dict):  # Criterion (Sum of all the losses)
    return sum(loss for loss in loss_dict.values())  
optimizer = optim.SGD(params, lr=0.001, momentum=0.9)  # Optimizer (Adam). Only parameters in the final layer are set.

###### 4. Training ######
def train_one_epoch(model: RetinaNet, optimizer: torch.optim.Optimizer,
                    data_loader: DataLoader, device: str, epoch: int):
    """
    Train the model in a epoch

    https://github.com/pytorch/vision/blob/main/references/detection/engine.py
    """
    # Initialize training metrics
    model.train()  # Set the training mode
    running_loss = 0.0  # Initialize running loss
    # Dynamic learning rate reducing (https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate)
    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )
    # Mini-batch loop
    for i, (imgs, targets) in enumerate(data_loader):
        # Send images and labels to GPU
        imgs = [img.to(device) for img in imgs]
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()}
                   for t in targets]
        # Calculate the loss
        loss_dict = model(imgs, targets)  # Forward (Prediction)
        loss = criterion(loss_dict)  # Calculate criterion
        # Update parameters
        optimizer.zero_grad()  # Initialize gradient
        loss.backward()  # Backpropagation (Calculate gradient)
        optimizer.step()  # Update parameters (Based on optimizer algorithm)
        # Update learning rate
        if lr_scheduler is not None:
            lr_scheduler.step()
        # Store running losses
        running_loss += loss.item()  # Update running loss
        if i%100 == 0:  # Show progress every 100 times
            print(f'minibatch index: {i}/{len(data_loader)}, elapsed_time: {time.time() - start}')
    # Calculate average of running losses
    running_loss /= len(data_loader)
    return running_loss

def evaluate(model: RetinaNet, data_loader: DataLoader, device: str):
    """
    Calculate validation metrics in a epoch

    https://github.com/pytorch/vision/blob/main/references/detection/engine.py#L76
    """
    val_running_loss = 0.0  # Initialize validation running loss
    # Mini-batch loop
    with torch.no_grad():
        for i, (val_imgs, val_targets) in enumerate(data_loader):
            val_imgs = [img.to(device) for img in val_imgs]
            val_targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()}
                        for t in val_targets]
            val_loss_dict = model(val_imgs, val_targets)  # Forward (Prediction)
            val_loss = criterion(val_loss_dict)  # Calculate criterion
            val_running_loss += val_loss.item()   # Update running loss
            if i%100 == 0:  # Show progress every 100 times
                print(f'val minibatch index: {i}/{len(data_loader)}, elapsed_time: {time.time() - start}')
    # Calculate average of running losses
    val_running_loss /= len(data_loader)
    return val_running_loss

# Conduct training
losses = []  # Array for storing loss (criterion)
val_losses = []  # Array for validation loss
start = time.time()  # For elapsed time
# Epoch loop
for epoch in range(NUM_EPOCHS):
    # Train the model in a epoch
    running_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)
    losses.append(running_loss)
    # Calculate validation metrics (https://pytorch.org/tutorials/beginner/introyt/trainingyt.html#per-epoch-activity)
    val_running_loss = evaluate(model, val_loader, device)
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
torch.save(params, f'{PARAMS_SAVE_ROOT}/pascalvoc_retinanet_fine.prm')

###### Inference in the first mini-batch ######
# Reload parameters
params_load = torch.load(f'{PARAMS_SAVE_ROOT}/pascalvoc_retinanet_fine.prm')
model.load_state_dict(params_load)
# Inference
val_iter = iter(val_loader)
imgs, targets = next(val_iter)
imgs_gpu = [img.to(device) for img in imgs]
model.eval()  # Set the evaluation smode
predictions = model(imgs_gpu)
# Class names dict with background
idx_to_class_bg = {k: v for k, v in idx_to_class.items()}
idx_to_class_bg[-1] = 'background'

show_predicted_detection_minibatch(imgs, predictions, targets, idx_to_class_bg, max_displayed_images=NUM_DISPLAYED_IMAGES)

#%%
###### Calculate mAP ######
aps = average_precisions_torchvison(val_loader, model, device, idx_to_class_bg, conf_threshold=0.2)
show_average_precisions(aps)

# %%
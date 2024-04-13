# %% Pascal VOC + Faseter R-CNN (ResNet50+FPN) + Fine Tuning (tune=After 2nd ResNet Layer, transfer=box_predictor)
# https://pytorch.org/examples/intermediate/torchvision_tutorial.html
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.datasets import VOCDetection
import matplotlib.pyplot as plt
import time
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from torch_extend.detection.display import show_bounding_boxes, show_predicted_detection_minibatch
from torch_extend.detection.target_converter import target_transform_to_torchvision

SEED = 42
BATCH_SIZE = 2  # Batch size
NUM_EPOCHS = 2  # number of epochs
NUM_DISPLAYED_IMAGES = 10  # number of displayed images
NUM_LOAD_WORKERS = 2  # Number of workers for DataLoader (Multiple workers not work in original dataset)
DEVICE = 'cuda'  # 'cpu' or 'cuda'
DATA_SAVE_ROOT = '/scripts/examples/object_detection/datasets'  # Directory for Saved dataset
PARAMS_SAVE_ROOT = '/scripts/examples/object_detection/params'  # Directory for Saved parameters
FREEZE_PRETRAINED = False  # If True, Freeze pretrained parameters (Transfer learning)
CLASS_TO_IDX = {
    'person': 0,
    'bird': 1,
    'cat': 2,
    'cow': 3,
    'dog': 4,
    'horse': 5,
    'sheep': 6,
    'aeroplane': 7,
    'bicycle': 8,
    'boat': 9,
    'bus': 10,
    'car': 11,
    'motorbike': 12,
    'train': 13,
    'bottle': 14,
    'chair': 15,
    'diningtable': 16,
    'pottedplant': 17,
    'sofa': 18,
    'tvmonitor': 19
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
    # Show bounding boxes
    show_bounding_boxes(img, boxes, labels=labels, idx_to_class=idx_to_class)
    plt.show()
# Load validation dataset
val_dataset = VOCDetection(root = DATA_SAVE_ROOT, year='2012',
                           image_set='val', download=True,
                           transform = transform, target_transform=target_transform)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_LOAD_WORKERS, collate_fn=collate_fn)

###### 2. Define Model ######
# Load a pretrained model
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
#print(model)
# Freeze pretrained parameters
if FREEZE_PRETRAINED:
    for param in model.parameters():
        param.requires_grad = False
# Modify the box_predictor
in_features = model.roi_heads.box_predictor.cls_score.in_features  # Number of features of Box Predictor
model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
print(model)
# Send the model to GPU
model.to(device)
# Choose parameters to be trained
#for p in model.parameters():
#    print(f'{p.shape} {p.requires_grad}')
params = [p for p in model.parameters() if p.requires_grad]

###### 3. Define Criterion & Optimizer ######
def criterion(loss_dict):  # Criterion (Sum of all the losses)
    return sum(loss for loss in loss_dict.values())  
optimizer = optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)  # Optimizer (Adam). Only parameters in the final layer are set.

###### 4. Training ######
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
        # Send images and labels to GPU
        # https://github.com/pytorch/vision/blob/main/references/detection/engine.py
        imgs = list(img.to(device) for img in imgs)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()}
                   for t in targets]
        # Update parameters
        optimizer.zero_grad()  # Initialize gradient
        loss_dict = model(imgs, targets)  # Forward (Prediction)
        loss = criterion(loss_dict)  # Calculate criterion
        loss.backward()  # Backpropagation (Calculate gradient)
        optimizer.step()  # Update parameters (Based on optimizer algorithm)
        # Store running losses
        running_loss += loss.item()  # Update running loss
        if i%100 == 0:  # Show progress every 100 times
            print(f'minibatch index: {i}/{len(train_loader)}, elapsed_time: {time.time() - start}')
    # Calculate average of running losses and accs
    running_loss /= len(train_loader)
    losses.append(running_loss)

    # Calculate validation metrics
    val_running_loss = 0.0  # Initialize validation running loss
    for i, (val_imgs, val_targets) in enumerate(val_loader):
        val_imgs = [img.to(device) for img in val_imgs]
        val_targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()}
                       for t in val_targets]
        val_loss_dict = model(val_imgs, val_targets)  # Forward (Prediction)
        val_loss = criterion(val_loss_dict)  # Calculate criterion
        val_running_loss += val_loss.item()   # Update running loss
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
torch.save(params, f'{PARAMS_SAVE_ROOT}/pascalvoc_fasterrcnn_fine.prm')

###### Inference in the first mini-batch ######
# Reload parameters
params_load = torch.load(f'{PARAMS_SAVE_ROOT}/pascalvoc_fasterrcnn_fine.prm')
model.load_state_dict(params)
# Inference
val_iter = iter(val_loader)
imgs, targets = next(val_iter)
imgs_gpu = [img.to(device) for img in imgs]
model.eval()  # Set the evaluation mode
predictions = model(imgs_gpu)
# Class names dict with background
idx_to_class_bg = {k: v for k, v in idx_to_class.items()}
idx_to_class_bg[-1] = 'background'

show_predicted_detection_minibatch(imgs, predictions, targets, idx_to_class_bg, max_displayed_images=NUM_DISPLAYED_IMAGES)

# %%

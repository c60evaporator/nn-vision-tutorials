# %% Pascal VOC Segmentation(SMP DataLoader) + UNet by smp (ResNet34) + Fine Tuning (tune=classification_head, transfer=classification_head.cls_logits)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import VOCSegmentation
from torchvision.transforms.functional import InterpolationMode
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import time
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from torch_extend.segmentation.display import show_segmentations

SEED = 42
BATCH_SIZE = 2  # Batch size
NUM_EPOCHS = 3  # number of epochs
NUM_DISPLAYED_IMAGES = 10  # number of displayed images
NUM_LOAD_WORKERS = 2  # Number of workers for DataLoader (Multiple workers not work in original dataset)
DEVICE = 'cuda'  # 'cpu' or 'cuda'
DATA_SAVE_ROOT = '/scripts/examples/detection/datasets'  # Directory for Saved dataset
PARAMS_SAVE_ROOT = '/scripts/examples/segmentation/params'  # Directory for Saved parameters
FREEZE_PRETRAINED = True  # If True, Freeze pretrained parameters (Transfer learning)
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
# Define preprocessing for image (https://poutyne.org/examples/semantic_segmentation.html)
def replace_tensor_value_(tensor, a, border_class):
    tensor[tensor == a] = border_class
    return tensor
transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=InterpolationMode.NEAREST),  # Resize an image to 224px x 224px
    transforms.PILToTensor(),  # Convert from PIL Image to a torch.FloatTensor in the range [0.0, 1.0]
    transforms.Lambda(lambda x: replace_tensor_value_(x.squeeze(0).long(), 255, 21))  # Replace the border to the border class ID
])
# Define preprocessing for target
# Load train dataset from image folder
train_dataset = VOCSegmentation(root = DATA_SAVE_ROOT, year='2012',
                                image_set='train', download=True,
                                transform = transform)
# Define class names
idx_to_class = {v: k for k, v in CLASS_TO_IDX.items()}
num_classes = len(idx_to_class) + 2  # Classification classes + 2 (background + border)
# Define mini-batch DataLoader
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_LOAD_WORKERS)
# Display images in the first mini-batch
def collate_fn(batch):
    return tuple(zip(*batch))
display_dataset = VOCSegmentation(root = DATA_SAVE_ROOT, year='2012',
                                image_set='train', download=True,
                                transform = transforms.ToTensor())
display_loader = DataLoader(display_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_LOAD_WORKERS, collate_fn=collate_fn)
display_iter = iter(display_loader)
imgs, targets = next(display_iter)
for i, (img, segmentation_img) in enumerate(zip(imgs, targets)):
    img = (img*255).to(torch.uint8)  # Change from float[0, 1] to uint[0, 255]
    #labels = [idx_to_class[label.item()] for label in labels]  # Change labels from index to str
    show_segmentations(img, segmentation_img)
    plt.show()
# Load validation dataset
val_dataset = VOCSegmentation(root = DATA_SAVE_ROOT, year='2012',
                              image_set='val', download=True,
                              transform = transform)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_LOAD_WORKERS)

###### 2. Define Model ######
# Load a pretrained network
network = smp.Unet(encoder_name='resnet34', encoder_weights='imagenet', classes=num_classes)

###### 3. Define Criterion & Optimizer ######
criterion = nn.CrossEntropyLoss()  # Criterion (Cross entropy loss)
optimizer = optim.Adam(network.parameters(), lr=0.0005)  # Optimizer (Adam). Only parameters in the final layer are set.

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
        imgs = list(img.to(device) for img in imgs)
        targets = [{k: v.to(device) for k, v in t.items()}
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
        val_targets = [{k: v.to(device) for k, v in t.items()}
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
torch.save(params, f'{PARAMS_SAVE_ROOT}/pascalvoc_retinanet_fine.prm')
# Reload parameters
#params_load = torch.load(f'{PARAMS_SAVE_ROOT}/pascalvoc_retinanet_fine.prm')
#model.load_state_dict(params)
# %%

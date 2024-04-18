# %% Pascal VOC Segmentation + FCN (ResNet50) + Fine Tuning (tune=classification_head, transfer=classification_head.cls_logits)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.datasets import VOCSegmentation
from torchvision.transforms.functional import InterpolationMode
import matplotlib.pyplot as plt
import time
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from torch_extend.segmentation.display import show_segmentations, show_predicted_segmentation_minibatch

SEED = 42
BATCH_SIZE = 8  # Batch size
NUM_EPOCHS = 10  # number of epochs
NUM_DISPLAYED_IMAGES = 10  # number of displayed images
NUM_LOAD_WORKERS = 4  # Number of workers for DataLoader (Multiple workers not work in original dataset)
DEVICE = 'cuda'  # 'cpu' or 'cuda'
DATA_SAVE_ROOT = '/scripts/examples/segmentation/datasets'  # Directory for Saved dataset
PARAMS_SAVE_ROOT = '/scripts/examples/segmentation/params'  # Directory for Saved parameters
FREEZE_PRETRAINED = True  # If True, Freeze pretrained parameters (Transfer learning)

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
# Reference resize_size=520 (https://pytorch.org/vision/main/models/generated/torchvision.models.segmentation.fcn_resnet50.html)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image to 224px x 224px
    transforms.ToTensor(),  # Convert from range [0, 255] to a torch.FloatTensor in the range [0.0, 1.0]
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)  # Normalization (mean and std of the imagenet dataset for normalizing)
])
# Define preprocessing of the target for VOCSegmentation dataset (https://poutyne.org/examples/semantic_segmentation.html)
def replace_tensor_value_(tensor, a, border_class):
    tensor[tensor == a] = border_class
    return tensor
target_transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=InterpolationMode.NEAREST),  # Resize the image to 224px x 224px
    transforms.PILToTensor(),  # Convert from PIL Image to a torch.FloatTensor in the range [0.0, 1.0]
    transforms.Lambda(lambda x: replace_tensor_value_(x.squeeze(0).long(), 255, len(CLASS_TO_IDX)))  # Replace the border to the border class ID
])
# Define preprocessing for target
# Load train dataset from image folder
train_dataset = VOCSegmentation(root = DATA_SAVE_ROOT, year='2012',
                                image_set='train', download=True,
                                transform = transform, target_transform=target_transform)
# Define class names
idx_to_class = {v: k for k, v in CLASS_TO_IDX.items()}
num_classes = len(idx_to_class) + 1  # Classification classes + 1 (border)
# Define mini-batch DataLoader
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_LOAD_WORKERS)
# Display images in the first mini-batch
display_dataset = VOCSegmentation(root = DATA_SAVE_ROOT, year='2012',
                                  image_set='train', download=True,
                                  transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()]),
                                  target_transform=target_transform)
display_loader = DataLoader(display_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_LOAD_WORKERS)
display_iter = iter(display_loader)
imgs, targets = next(display_iter)
for i, (img, target) in enumerate(zip(imgs, targets)):
    img = (img*255).to(torch.uint8)  # Change from float[0, 1] to uint[0, 255]
    #labels = [idx_to_class[label.item()] for label in labels]  # Change labels from index to str
    show_segmentations(img, target, idx_to_class, bg_idx=0, border_idx=len(CLASS_TO_IDX))
# Load validation dataset
val_dataset = VOCSegmentation(root = DATA_SAVE_ROOT, year='2012',
                              image_set='val', download=True,
                              transform = transform, target_transform=target_transform)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_LOAD_WORKERS)
# Reverse transform for showing the image
val_reverse_transform = transforms.Compose([
    transforms.Normalize(mean=[-mean/std for mean, std in zip(IMAGENET_MEAN, IMAGENET_STD)],
                         std=[1/std for std in IMAGENET_STD])
])

###### 2. Define Model ######
# Load a pretrained network (https://www.kaggle.com/code/dasmehdixtr/load-finetune-pretrained-model-in-pytorch)
#weights = models.segmentation.FCN_ResNet50_Weights.DEFAULT
#model = models.segmentation.fcn_resnet50(weights=weights)
model = models.segmentation.fcn_resnet50(pretrained=True)
# Freeze pretrained parameters
if FREEZE_PRETRAINED:
    for param in model.parameters():
        param.requires_grad = False
# Modify the last Conv layer of the classifier and the aux_classifier 
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
# https://github.com/pytorch/vision/tree/main/references/segmentation
model.train()  # Set the training mode
losses = []  # Array for string loss (criterion)
val_losses = []  # Array for validation loss
start = time.time()  # For elapsed time
# Epoch loop
for epoch in range(NUM_EPOCHS):
    # Initialize training metrics
    running_loss = 0.0  # Initialize running loss
    running_acc = 0.0  # Initialize running accuracy
    # Mini-batch loop
    for i, (imgs, targets) in enumerate(train_loader):
        # Send images and labels to GPU ()
        imgs = imgs.to(device)
        targets = targets.to(device)
        # Update parameters
        optimizer.zero_grad()  # Initialize gradient
        output = model(imgs)  # Forward (Prediction)
        loss = criterion(output, targets)  # Calculate criterion
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
        val_imgs = val_imgs.to(device)
        val_targets = val_targets.to(device)
        val_output = model(val_imgs)  # Forward (Prediction)
        val_loss = criterion(val_output, val_targets)  # Calculate criterion
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
torch.save(params, f'{PARAMS_SAVE_ROOT}/vocsegmentation_fcn.prm')


# %%
###### Inference in the first mini-batch ######
# Reload parameters
params_load = torch.load(f'{PARAMS_SAVE_ROOT}/vocsegmentation_fcn.prm')
model.load_state_dict(params_load)
# Inference
val_iter = iter(val_loader)
imgs, targets = next(val_iter)
imgs_gpu = imgs.to(device)
model.eval()  # Set the evaluation mode
predictions = model(imgs_gpu)
# Reverse normalization for getting the raw image
imgs_display = [val_reverse_transform(img) for img in imgs]
# Show the image
show_predicted_segmentation_minibatch(imgs_display, predictions, targets, idx_to_class, 
                                      bg_idx=0, border_idx=len(CLASS_TO_IDX), plot_raw_image=True,
                                      max_displayed_images=NUM_DISPLAYED_IMAGES)

# %%

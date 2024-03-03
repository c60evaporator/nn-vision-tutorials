# %% Hymenoptera + ResNet18 + Transfer Learning
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models, datasets
import matplotlib.pyplot as plt
import time
import os
import requests
from zipfile import ZipFile

SEED = 42
BATCH_SIZE = 32
NUM_EPOCHS = 15  # number of epochs
NUM_DISPLAYED_IMAGES = 10  # number of displayed images
NUM_LOAD_WORKERS = 0  # Number of workers for DataLoader (Multiple workers not work in original dataset)
DEVICE = 'cuda'  # 'cpu' or 'cuda'
DATA_SAVE_ROOT = '/scripts/examples/cnn/datasets'  # Directory for Saved dataset
PARAMS_SAVE_ROOT = '/scripts/examples/cnn/params'  # Directory for Saved parameters
DATA_DOWNLOAD_URL = 'https://download.pytorch.org/tutorial/hymenoptera_data.zip'
TRAIN_DATA_DIR = 'train'
VAL_DATA_DIR = 'val'

# Confirm GPU availability
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if DEVICE == 'cpu':
    device = 'cpu'
# Set random seed
torch.manual_seed(SEED)

###### 1. Create dataset & Preprocessing ######
# Download dataset (https://www.tutorialspoint.com/downloading-files-from-web-using-python)
res = requests.get(DATA_DOWNLOAD_URL, allow_redirects=True)
zip_path = f'{DATA_SAVE_ROOT}/{DATA_DOWNLOAD_URL.split("/")[-1]}'
open(zip_path, 'wb').write(res.content)
data_dir = os.path.splitext(zip_path)[0]
# Unzip dataset
with ZipFile(zip_path, 'r') as z:
    z.extractall(path=DATA_SAVE_ROOT)
# Define preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images
    transforms.ToTensor(),  # Convert from range [0, 255] to a torch.FloatTensor in the range [0.0, 1.0]
    transforms.Normalize((0.5,), (0.5,))  # Normalization
])
# Load train dataset from image folder
train_dataset = datasets.ImageFolder(f'{data_dir}/{TRAIN_DATA_DIR}', transform=transform)
# Define class names
idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}
# Define mini-batch DataLoader
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_LOAD_WORKERS)
# Display images in the first mini-batch
train_iter = iter(train_loader)
imgs, labels = next(train_iter)
print(imgs.size())  # Print minibatch data shape
for i, (img, label) in enumerate(zip(imgs, labels)):
    img_permute = img.permute(1, 2, 0)  # Change axis order from (ch, x, y) to (x, y, ch)
    plt.imshow(img_permute)  # Display the image
    plt.title(f'idx: {i}, label: {idx_to_class[int(label)]}')
    plt.show()
    if i >= NUM_DISPLAYED_IMAGES - 1:
        break
# Load validation dataset
val_dataset = datasets.ImageFolder(f'{data_dir}/{VAL_DATA_DIR}', transform=transform)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_LOAD_WORKERS)

###### 2. Define Model ######
# Load a pretrained model
model = models.resnet18(pretrained=True)
print(model)
# Freeze all the parameters (https://pytorch.org/examples/beginner/transfer_learning_tutorial.html#convnet-as-fixed-feature-extractor)
for param in model.parameters():
    param.requires_grad = False
# Modify the final fully connected layer
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(idx_to_class.keys()))
# Send the model to GPU
model.to(device)

###### 3. Define Criterion & Optimizer ######
criterion = nn.CrossEntropyLoss()  # Criterion (Cross entropy loss)
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)  # Optimizer (Adam). Only parameters in the final layer are set.

###### 4. Training ######
losses = []  # Array for storing loss (criterion)
accs = []  # Array for storing accuracy
val_losses = []  # Array for validation loss
val_accs = []  # Array for validation accuracy
start = time.time()  # For elapsed time
# Epoch loop
for epoch in range(NUM_EPOCHS):
    # Initialize training metrics
    running_loss = 0.0  # Initialize running loss
    running_acc = 0.0  # Initialize running accuracy
    # Mini-batch loop
    for imgs, labels in train_loader:
        # Send images and labels to GPU
        imgs = imgs.to(device)  
        labels = labels.to(device)
        # Update parameters
        optimizer.zero_grad()  # Initialize gradient
        output = model(imgs)  # Forward (Prediction)
        loss = criterion(output, labels)  # Calculate criterion
        loss.backward()  # Backpropagation (Calculate gradient)
        optimizer.step()  # Update parameters (Based on optimizer algorithm)
        # Store running losses and accs
        running_loss += loss.item()  # Update running loss
        pred = torch.argmax(output, dim=1)  # Predicted labels
        running_acc += torch.mean(pred.eq(labels).float())  # Update running accuracy
    # Calculate average of running losses and accs
    running_loss /= len(train_loader)
    losses.append(running_loss)
    running_acc /= len(train_loader)
    accs.append(running_acc.cpu())

    # Calculate validation metrics
    val_running_loss = 0.0  # Initialize validation running loss
    val_running_acc = 0.0  # Initialize validation running accuracy
    for val_imgs, val_labels in val_loader:
        val_imgs = val_imgs.to(device)
        val_labels = val_labels.to(device)
        val_output = model(val_imgs)  # Forward (Prediction)
        val_loss = criterion(val_output, val_labels)  # Calculate criterion
        val_running_loss += val_loss.item()   # Update running loss
        val_pred = torch.argmax(val_output, dim=1)  # Predicted labels
        val_running_acc += torch.mean(val_pred.eq(val_labels).float())  # Update running accuracy
    val_running_loss /= len(val_loader)
    val_losses.append(val_running_loss)
    val_running_acc /= len(val_loader)
    val_accs.append(val_running_acc.cpu())

    print(f'epoch: {epoch}, loss: {running_loss}, acc: {running_acc},  val_loss: {val_running_loss}, val_acc: {val_running_acc}, elapsed_time: {time.time() - start}')

###### 5. Model evaluation and visualization ######
# Plot loss history
plt.plot(losses, label='Train loss')
plt.plot(val_losses, label='Validation loss')
plt.title('Loss history')
plt.legend()
plt.show()
# Plot accuracy history
plt.plot(accs, label='Train accuracy')
plt.plot(val_accs, label='Validation accuracy')
plt.title('Accuracy history')
plt.legend()
plt.show()

# Prediction after training
val_iter = iter(val_loader)
imgs, labels = next(val_iter)
imgs_gpu = imgs.to(device)
output = model(imgs_gpu)
preds = torch.argmax(output, dim=1)
# Display images
for i, (img, label, pred) in enumerate(zip(imgs, labels, preds)):
    img_permute = img.permute(1, 2, 0)
    plt.imshow(img_permute)  # Display the image
    plt.title(f'idx: {i}, true: {idx_to_class[int(label)]}, pred: {idx_to_class[int(pred)]}')
    plt.show()
    if i >= NUM_DISPLAYED_IMAGES - 1:
        break

###### 6. Save the model ######
# Save parameters
params = model.state_dict()
if not os.path.exists(PARAMS_SAVE_ROOT):
    os.makedirs(PARAMS_SAVE_ROOT)
torch.save(params, f'{PARAMS_SAVE_ROOT}/hymenoptera_resnet18_transfer.prm')
# Reload parameters
params_load = torch.load(f'{PARAMS_SAVE_ROOT}/hymenoptera_resnet18_transfer.prm')
model.load_state_dict(params)

# %%

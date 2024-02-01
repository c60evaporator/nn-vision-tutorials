"""
https://www.udemy.com/course/hands-on-pytorch/
"""

# %% Linear Regression
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, random_split
import matplotlib.pyplot as plt

SEED = 42
TEST_SIZE = 0.25
NUM_EPOCHS = 500  # number of epochs
plt.style.use('ggplot')

###### 1. Create dataset & Preprocessing ######
# Create tensors for dataset
torch.manual_seed(SEED)
a =3 
b = 2
x = torch.linspace(0, 5, 100).view(100, 1)
eps = torch.randn(100,1)
y = a * x + b + eps
plt.scatter(x, y)
plt.title('Dataset')
plt.show()
# Create dataset
# class TensorDatasetRegression(Dataset):
#     def __init__(self, x, y):
#         self.x = x
#         self.y = y
#     def __len__(self):
#         return len(self.x)
#     def __getitem__(self, idx):
#         sample = {
#             'feature': torch.tensor([self.x[idx]], dtype=torch.float32), 
#             'target': torch.tensor([self.y[idx]], dtype=torch.float32)
#         }
#         return sample
dataset = TensorDataset(x, y)
# Split dataset
train_set, test_set = random_split(dataset, [1-TEST_SIZE, TEST_SIZE])
x_train, x_test = x[train_set.indices], x[test_set.indices]
y_train, y_test = y[train_set.indices], y[test_set.indices]

###### 2. Define a model ######
# Define model
class LR(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(in_features=1, out_features=1)
    def forward(self, x):
        output = self.linear(x)
        return output
model = LR()
# Prediction after training
y_pred_before = model(x_test)
plt.plot(x_test, y_pred_before.detach(), label='prediction')
plt.scatter(x_test, y_test, label='data')
plt.legend()
plt.title('Prediction without training')
plt.show()

###### 3. Define Criterion & Optimizer ######
criterion = nn.MSELoss()  # Criterion (MSE)
optimizer = optim.SGD(model.parameters(), lr=0.001)  # Optimizer (SGD)

###### 4. Training ######
losses = []  # Array for storing loss (criterion)
# Epoch loop (Without minibatch selection)
for epoch in range(NUM_EPOCHS):
    # Update parameters
    optimizer.zero_grad()  # Initialize gradient
    y_pred = model(x_train)  # Forward (Prediction)
    loss = criterion(y_pred, y_train)  # Calculate criterion
    loss.backward()  # Backpropagation (Calculate gradient)
    optimizer.step()  # Update parameters (Based on optimizer algorithm)
    if epoch % 10 ==0:  # Store and print loss every 10 epochs
        print(f'epoch: {epoch}, loss: {loss.item()}')
        losses.append(loss.item())

###### 5. Model evaluation and visualization ######
# Plot loss history
plt.plot(losses)
plt.title('Loss history')
plt.show()
# Prediction after training
y_pred_after = model(x_test)
plt.plot(x_test, y_pred_after.detach(), label='prediction')
plt.scatter(x_test, y_test, label='data')
plt.legend()
plt.title('Prediction after training')
plt.show()

# %% MNIST + MLP
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import seaborn as sns
import matplotlib.pyplot as plt
import time
import os

SEED = 42
#TEST_SIZE = 0.25
BATCH_SIZE = 100
NUM_EPOCHS = 15  # number of epochs
NUM_DISPLAYED_IMAGES = 10  # number of displayed images
NUM_LOAD_WORKERS = 2  # Number of workers for DataLoader
DEVICE = 'cuda'  # 'cpu' or 'cuda'
DATA_SAVE_ROOT = '/scripts/tutorials/datasets'  # Directory for Saved dataset
PARAMS_SAVE_ROOT = '/scripts/tutorials/params'  # Directory for Saved parameters

# Confirm GPU availability
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if DEVICE == 'cpu':
    device = 'cpu'
# Set random seed
torch.manual_seed(SEED)

###### 1. Create dataset & Preprocessing ######
# Define preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert from range [0, 255] to a torch.FloatTensor in the range [0.0, 1.0]
    transforms.Normalize((0.5,), (0.5,))  # Normalization
])
# Load train dataset
train_dataset = MNIST(root = DATA_SAVE_ROOT, train = True, transform = transform, download=True)
# train_set, val_set = random_split(train_dataset, [1-TEST_SIZE, TEST_SIZE], generator=torch.Generator().manual_seed(42))
# Define class names
idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}
# Define mini-batch DataLoader
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_LOAD_WORKERS)
# Display images in the first mini-batch
train_iter = iter(train_loader)
imgs, labels = next(train_iter)
print(imgs.size())  # Print minibatch data shape
for i, (img, label) in enumerate(zip(imgs, labels)):
    sns.heatmap(img.numpy()[0, :, :])  # Display the image as heatmap
    plt.title(f'idx: {i}, label: {idx_to_class[int(label)]}')
    plt.show()
    if i >= NUM_DISPLAYED_IMAGES - 1:
        break
###### 2. Define Model ######
# Define Model
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(28 * 28, 400),
            nn.ReLU(inplace=True),
            nn.Linear(400, 200),
            nn.ReLU(inplace=True),
            nn.Linear(200, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 10)
        )
    def forward(self, x):
        output = self.classifier(x)
        return output
model = MLP()
# Send model to GPU
model.to(device)

###### 3. Define Criterion & Optimizer ######
criterion = nn.CrossEntropyLoss()  # Criterion (Cross entropy loss)
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Optimizer (Adam)

###### 4. Training ######
losses = []  # Array for storing loss (criterion)
accs = []  # Array for storing accuracy
start = time.time()  # For elapsed time
# Epoch loop
for epoch in range(NUM_EPOCHS):
    # Initialize training metrics
    running_loss = 0.0  # Initialize running loss
    running_acc = 0.0  # Initialize running accuracy
    # Mini-batch loop
    for imgs, labels in train_loader:
        # Reshape image to 1-D vector
        imgs = imgs.view(BATCH_SIZE, -1)
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
    print(f'epoch: {epoch}, loss: {running_loss}, acc: {running_acc}, elapsed_time: {time.time() - start}')

###### 5. Model evaluation and visualization ######
# Plot loss history
plt.plot(losses)
plt.title('Loss history')
plt.show()
# Plot accuracy history
plt.plot(accs)
plt.title('Accuracy history')
plt.show()

# Prediction after training
val_dataset = MNIST(root = DATA_SAVE_ROOT, train = False, transform = transform, download=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_LOAD_WORKERS)
val_iter = iter(val_loader)
imgs, labels = next(val_iter)
imgs_gpu = imgs.view(100, -1).to(device)
output = model(imgs_gpu)
preds = torch.argmax(output, dim=1)
# Display images
for i, (img, label, pred) in enumerate(zip(imgs, labels, preds)):
    sns.heatmap(img.numpy()[0, :, :])  # Display the image as heatmap
    plt.title(f'idx: {i}, true: {label}, pred: {pred}')
    plt.show()
    if i >= NUM_DISPLAYED_IMAGES - 1:
        break

###### 6. Save the model ######
# Save parameters
params = model.state_dict()
if not os.path.exists(PARAMS_SAVE_ROOT):
    os.makedirs(PARAMS_SAVE_ROOT)
torch.save(params, f'{PARAMS_SAVE_ROOT}/mnist_mlp.prm')
# Reload parameters
params_load = torch.load(f'{PARAMS_SAVE_ROOT}/mnist_mlp.prm')
model_load = MLP()
model_load.load_state_dict(params)

# %% CIFAR10 + CNN
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
import matplotlib.pyplot as plt
import time
import os

SEED = 42
BATCH_SIZE = 32
NUM_EPOCHS = 15  # number of epochs
NUM_DISPLAYED_IMAGES = 10  # number of displayed images
NUM_LOAD_WORKERS = 2  # Number of workers for DataLoader
DEVICE = 'cuda'  # 'cpu' or 'cuda'
DATA_SAVE_ROOT = '/scripts/tutorials/datasets'  # Directory for Saved dataset
PARAMS_SAVE_ROOT = '/scripts/tutorials/params'  # Directory for Saved parameters
CLASS_NAMES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Confirm GPU availability
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if DEVICE == 'cpu':
    device = 'cpu'
# Set random seed
torch.manual_seed(SEED)

###### 1. Create dataset & Preprocessing ######
# Define preprocessing
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # Augmentation by flipping
    transforms.ColorJitter(),  # Augmentation by randomly changing brightness, contrast, saturation, and hue.
    transforms.RandomRotation(degrees=10),  # Augmentation by rotation
    transforms.ToTensor(),  # Convert from range [0, 255] to a torch.FloatTensor in the range [0.0, 1.0]
    transforms.Normalize((0.5,), (0.5,))  # Normalization
])
# Load train dataset
train_dataset = CIFAR10(root = DATA_SAVE_ROOT, train = True, transform = transform, download=True)
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
    #img_permute = 0.5 * img_permute + 0.5  # Adjust image brightness
    #img_permute = torch.clip(img_permute, 0, 1)  # Clip image
    plt.imshow(img_permute)  # Display the image
    plt.title(f'idx: {i}, label: {idx_to_class[int(label)]}')
    plt.show()
    if i >= NUM_DISPLAYED_IMAGES - 1:
        break
# Load validation dataset
val_dataset = CIFAR10(root = DATA_SAVE_ROOT, train = False, transform = transform, download=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_LOAD_WORKERS)

###### 2. Define Model ######
# Define Model
class CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Linear(
            in_features = 4 * 4 * 128, out_features=num_classes
        )
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten to 1-D vector
        output = self.classifier(x)
        return output
model = CNN(num_classes=10)
# Send model to GPU
model.to(device)

###### 3. Define Criterion & Optimizer ######
criterion = nn.CrossEntropyLoss()  # Criterion (Cross entropy loss)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)  # Optimizer (Adam)

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
    #img_permute = 0.5 * img_permute + 0.5  # Adjust image brightness
    #img_permute = torch.clip(img_permute, 0, 1)  # Clip image
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
torch.save(params, f'{PARAMS_SAVE_ROOT}/cifar10_cnn.prm')
# Reload parameters
params_load = torch.load(f'{PARAMS_SAVE_ROOT}/cifar10_cnn.prm')
model_load = CNN(num_classes=10)
model_load.load_state_dict(params)

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
DATA_SAVE_ROOT = '/scripts/tutorials/datasets'  # Directory for Saved dataset
PARAMS_SAVE_ROOT = '/scripts/tutorials/params'  # Directory for Saved parameters
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
# Freeze all the parameters (https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html#convnet-as-fixed-feature-extractor)
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

# %% MNIST + CNN-Autoencoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os

SEED = 42
BATCH_SIZE = 100
NUM_EPOCHS = 15  # number of epochs
NUM_DISPLAYED_IMAGES = 10  # number of displayed images
MAX_DISPLAYED_CHANNEL = 8  # Maximum displayed channels of encoded image
NUM_LOAD_WORKERS = 2  # Number of workers for DataLoader
DEVICE = 'cuda'  # 'cpu' or 'cuda'
DATA_SAVE_ROOT = '/scripts/tutorials/datasets'  # Directory for Saved dataset
PARAMS_SAVE_ROOT = '/scripts/tutorials/params'  # Directory for Saved parameters

# Confirm GPU availability
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if DEVICE == 'cpu':
    device = 'cpu'
# Set random seed
torch.manual_seed(SEED)

###### 1. Create dataset & Preprocessing ######
# Define preprocessing
transform = transforms.Compose([
    transforms.ToTensor()  # Convert from range [0, 255] to a torch.FloatTensor in the range [0.0, 1.0]
])
# Load train dataset
train_dataset = MNIST(root = DATA_SAVE_ROOT, train = True, transform = transform, download=True)
# Define mini-batch DataLoader
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# Display images in the first mini-batch
train_iter = iter(train_loader)
imgs, _ = next(train_iter)
print(imgs.size())  # Print minibatch data shape
for i, img in enumerate(imgs):
    img_permute = img.permute(1, 2, 0)  # Change axis order from (ch, x, y) to (x, y, ch)
    plt.imshow(img_permute, cmap='gray')  # Display the image
    plt.title(f'idx: {i}')
    plt.show()
    if i >= NUM_DISPLAYED_IMAGES - 1:
        break
# Load validation dataset
val_dataset = MNIST(root = DATA_SAVE_ROOT, train = False, transform = transform, download=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_LOAD_WORKERS)

###### 2. Define Model ######
# Define Model
class ConvAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=8, out_channels=1, kernel_size=3, padding=1),
            nn.Tanh()  # Tanh and Sigmoid tend to be better than ReLU as the activation function in the final layer of the decoder
        )
    def forward(self, x):
        x = self.encoder(x)
        output = self.decoder(x)
        return output
model = ConvAE()
# Send model to GPU
model.to(device)

###### 3. Define Criterion & Optimizer ######
criterion = nn.MSELoss()  # Criterion (MSE)
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Optimizer (Adam)

###### 4. Training ######
losses = []  # Array for storing loss (criterion)
val_losses = []  # Array for validation loss
start = time.time()  # For elapsed time
# Epoch loop
for epoch in range(NUM_EPOCHS):
    # Initialize training metrics
    running_loss = 0.0  # Initialize running loss
    # Mini-batch loop
    for imgs, _ in train_loader:
        # Send images to GPU
        imgs = imgs.to(device)
        # Update parameters
        optimizer.zero_grad()  # Initialize gradient
        output = model(imgs)  # Forward (Prediction)
        loss = criterion(output, imgs)  # Calculate criterion (MSE between input and output images)
        loss.backward()  # Backpropagation (Calculate gradient)
        optimizer.step()  # Update parameters (Based on optimizer algorithm)
        # Store running losses
        running_loss += loss.item()  # Update running loss
    # Calculate average of running losses
    running_loss /= len(train_loader)
    losses.append(running_loss)

    # Calculate validation metrics
    val_running_loss = 0.0  # Initialize validation running loss
    for val_imgs, _ in val_loader:
        val_imgs = val_imgs.to(device)
        val_output = model(val_imgs)  # Forward (Prediction)
        val_loss = criterion(val_output, val_imgs)  # Calculate criterion
        val_running_loss += val_loss.item()   # Update running loss
    val_running_loss /= len(val_loader)
    val_losses.append(val_running_loss)

    print(f'epoch: {epoch}, loss: {running_loss},  val_loss: {val_running_loss}, elapsed_time: {time.time() - start}')

###### 5. Model evaluation and visualization ######
# Plot loss history
plt.plot(losses, label='Train loss')
plt.plot(val_losses, label='Validation loss')
plt.title('Loss history')
plt.legend()
plt.show()

# Display encoded and decoded image
val_dataset = MNIST(root = DATA_SAVE_ROOT, train = False, transform = transform, download=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_LOAD_WORKERS)
val_iter = iter(val_loader)
imgs, _ = next(val_iter)
imgs_gpu = imgs.to(device)
imgs_encoded = model.encoder(imgs_gpu)
imgs_decoded = model(imgs_gpu)
# Display images
for i, (img, img_encoded, img_decoded) in enumerate(zip(imgs, imgs_encoded, imgs_decoded)):
    fig = plt.figure(figsize=(14, 6))
    # Create canvas
    num_channels = img_encoded.size()[0]
    num_disp_channel = min(num_channels, MAX_DISPLAYED_CHANNEL)
    gs = fig.add_gridspec(num_disp_channel, 3, width_ratios=[num_disp_channel, 1, num_disp_channel])
    # Display the input image as heatmap
    ax_input = fig.add_subplot(gs[:, 0])
    ax_input.imshow(img.permute(1, 2, 0), cmap='gray')  # Display the image
    ax_input.set_title(f'idx: {i}, Input image, size={list(img.size())}')
    # Display the encoded image as heatmap
    for channel in range(num_disp_channel):
        ax_encoded = fig.add_subplot(gs[channel, 1])
        ax_encoded.imshow(img_encoded.detach().to('cpu')[channel, :, :], cmap='gray')  
        ax_encoded.set_ylabel(f'ch{channel+1}', fontsize=8)
        if channel == 0:
            ax_encoded.set_title(f'Encoded {list(img_encoded.size())}', fontsize=12)
    # Display the decoded image as heatmap
    ax_decoded = fig.add_subplot(gs[:, 2])
    ax_decoded.imshow(img_decoded.permute(1, 2, 0).detach().to('cpu'), cmap='gray')  # Display the image
    ax_decoded.set_title(f'idx: {i}, Input image, size={list(img.size())}')
    plt.show()
    if i >= NUM_DISPLAYED_IMAGES - 1:
        break

###### 6. Save the model ######
# Save parameters
params = model.state_dict()
if not os.path.exists(PARAMS_SAVE_ROOT):
    os.makedirs(PARAMS_SAVE_ROOT)
torch.save(params, f'{PARAMS_SAVE_ROOT}/mnist_cnn_autoencoder.prm')
# Reload parameters
params_load = torch.load(f'{PARAMS_SAVE_ROOT}/mnist_cnn_autoencoder.prm')
model_load = ConvAE()
model_load.load_state_dict(params)

# %% CIFAR10 + ResNet10
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
import matplotlib.pyplot as plt
import time
import os

SEED = 42
BATCH_SIZE = 32
NUM_EPOCHS = 15  # number of epochs
NUM_DISPLAYED_IMAGES = 10  # number of displayed images
NUM_LOAD_WORKERS = 2  # Number of workers for DataLoader
DEVICE = 'cuda'  # 'cpu' or 'cuda'
DATA_SAVE_ROOT = '/scripts/tutorials/datasets'  # Directory for Saved dataset
PARAMS_SAVE_ROOT = '/scripts/tutorials/params'  # Directory for Saved parameters

# Confirm GPU availability
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if DEVICE == 'cpu':
    device = 'cpu'
# Set random seed
torch.manual_seed(SEED)

###### 1. Create dataset & Preprocessing ######
# Define preprocessing
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # Augmentation by flipping
    transforms.ColorJitter(),  # Augmentation by randomly changing brightness, contrast, saturation, and hue.
    transforms.RandomRotation(degrees=10),  # Augmentation by rotation
    transforms.ToTensor(),  # Convert from range [0, 255] to a torch.FloatTensor in the range [0.0, 1.0]
    transforms.Normalize((0.5,), (0.5,))  # Normalization
])
# Load train dataset
train_dataset = CIFAR10(root = DATA_SAVE_ROOT, train = True, transform = transform, download=True)
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
    #img_permute = 0.5 * img_permute + 0.5  # Adjust image brightness
    #img_permute = torch.clip(img_permute, 0, 1)  # Clip image
    plt.imshow(img_permute)  # Display the image
    plt.title(f'idx: {i}, label: {idx_to_class[int(label)]}')
    plt.show()
    if i >= NUM_DISPLAYED_IMAGES - 1:
        break
# Load validation dataset
val_dataset = CIFAR10(root = DATA_SAVE_ROOT, train = False, transform = transform, download=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_LOAD_WORKERS)

###### 2. Define Model ######
# Define BasicBlock
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.stride = stride
        # Main CNN (bias should be set to False https://qiita.com/sugulu_Ogawa_ISID/items/62f5f7adee083d96a587#62-%E3%83%90%E3%83%83%E3%83%81%E3%83%8E%E3%83%BC%E3%83%9E%E3%83%A9%E3%82%A4%E3%82%BC%E3%83%BC%E3%82%B7%E3%83%A7%E3%83%B3%E3%81%AE%E5%89%8D%E3%81%AE%E5%B1%A4%E3%81%AF%E3%83%90%E3%82%A4%E3%82%A2%E3%82%B9%E3%83%91%E3%83%A9%E3%83%A1%E3%83%BC%E3%82%BF%E3%82%92false%E3%81%AB)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # Downsampling (Skip connection)
        if self.stride != 1:
            self.skipconv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=0, bias=False)
            self.skipbn = nn.BatchNorm2d(out_channels)
    def downsample(self, x):
        x = self.skipconv(x)
        x = self.skipbn(x)
        return x
    def forward(self, x):
        identity = x
        # Main CNN
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        # Downsampling (Skip connection)
        if self.stride != 1:
            x += self.downsample(identity)
        return x
# Define Model (https://debuggercafe.com/implementing-resnet18-in-pytorch-from-scratch/)
class ResNet(nn.Module):
    def __init__(self, block, num_blocks=[1, 1, 1, 1],
                 img_channels=3, inner_channels=[64, 64, 128, 192, 256],
                 num_classes=10):
        super().__init__()
        self.in_channels = 64
        # First convolution layer
        self.conv1 = nn.Conv2d(in_channels=img_channels, out_channels=self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # Basic blocks
        self.layer1 = self._make_layer(block, num_blocks=num_blocks[0], in_channels=inner_channels[0], out_channels=inner_channels[1], stride=1)
        self.layer2 = self._make_layer(block, num_blocks=num_blocks[1], in_channels=inner_channels[1], out_channels=inner_channels[2], stride=2)
        self.layer3 = self._make_layer(block, num_blocks=num_blocks[2], in_channels=inner_channels[2], out_channels=inner_channels[3], stride=2)
        self.layer4 = self._make_layer(block, num_blocks=num_blocks[3], in_channels=inner_channels[3], out_channels=inner_channels[4], stride=2)
        # AvgPool and Affine layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_features = 1 * 1 * inner_channels[-1], out_features=num_classes)
    def _make_layer(self, block, num_blocks, in_channels, out_channels, stride=1):
        layers = []
        for i in range(num_blocks):
            if i == 0:
                layers.append(block(in_channels, out_channels, stride=stride))
            else:
                layers.append(block(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)
    def forward(self, x):
        # First convolution layer
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # Basic blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # AvgPool and Affine layer
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # Flatten to 1-D vector
        output = self.fc(x)
        return output

model = ResNet(block=BasicBlock, num_classes=10)
print(model)
# Send model to GPU
model.to(device)

###### 3. Define Criterion & Optimizer ######
criterion = nn.CrossEntropyLoss()  # Criterion (Cross entropy loss)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)  # Optimizer (Adam)

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
torch.save(params, f'{PARAMS_SAVE_ROOT}/cifar10_resnet10.prm')
# Reload parameters
params_load = torch.load(f'{PARAMS_SAVE_ROOT}/cifar10_resnet10.prm')
model_load = ResNet(block=BasicBlock, num_classes=10)
model_load.load_state_dict(params)

# %% Original dataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset, random_split
from torchvision import transforms
import matplotlib.pyplot as plt
import time
import os
from zipfile import ZipFile
from PIL import Image
import numpy as np
import copy

SEED = 42
TEST_SIZE = 0.25
BATCH_SIZE = 32
NUM_EPOCHS = 15  # number of epochs
NUM_DISPLAYED_IMAGES = 10  # number of displayed images
NUM_LOAD_WORKERS = 0  # Number of workers for DataLoader (Multiple workers not work in original dataset)
DEVICE = 'cuda'  # 'cpu' or 'cuda'
DATA_SAVE_ROOT = '/scripts/tutorials/datasets'  # Directory for Saved dataset
PARAMS_SAVE_ROOT = '/scripts/tutorials/params'  # Directory for Saved parameters
DATA_FILE_NAME = 'train.zip'
TRAIN_DATA_DIR = 'train'
VAL_DATA_DIR = 'val'
CLASS_TO_IDX = {'cat': 0, 'dog': 1}

# Confirm GPU availability
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if DEVICE == 'cpu':
    device = 'cpu'
# Set random seed
torch.manual_seed(SEED)

###### 1. Create dataset & Preprocessing ######
# Unzip dataset
zip_path = f'{DATA_SAVE_ROOT}/{DATA_FILE_NAME}'
data_dir = os.path.splitext(zip_path)[0]
with ZipFile(zip_path, 'r') as z:
    z.extractall(path=DATA_SAVE_ROOT)
# Classify labels from file names
file_list = os.listdir(data_dir)
cat_files = [file_name for file_name in file_list if 'cat' in file_name]
dog_files = [file_name for file_name in file_list if 'dog' in file_name]
# Define preprocessing
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize images
    transforms.ToTensor(),  # Convert from range [0, 255] to a torch.FloatTensor in the range [0.0, 1.0]
    transforms.Normalize((0.5,), (0.5,))  # Normalization
])
# Define Dataset
class CatDogDataset(Dataset):
    def __init__(self, file_list, dir, transform=None):
        self.file_list = file_list
        self.dir = dir
        self.transform = transform
        if 'dog' in self.file_list[0]:
            self.label = 1
        else:
            self.label = 0
    def __len__(self):
        return len(self.file_list)
    def __getitem__(self, index):
        file_path = os.path.join(self.dir, self.file_list[index])
        img = Image.open(file_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, self.label

# Load train dataset from image folder
cat_dataset = CatDogDataset(cat_files, data_dir, transform=transform)
dog_dataset = CatDogDataset(dog_files, data_dir, transform=transform)
cat_dog_dataset = ConcatDataset([cat_dataset, dog_dataset])
# Split dataset
train_dataset, val_dataset = random_split(cat_dog_dataset, [1-TEST_SIZE, TEST_SIZE])
# Define class names
idx_to_class = {v: k for k, v in CLASS_TO_IDX.items()}
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
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_LOAD_WORKERS)

# %% DummySin + LSTM
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
import matplotlib.pyplot as plt
import time
import os
import numpy as np

SEED = 42
TEST_SIZE = 0.1
BATCH_SIZE = 32
NUM_EPOCHS = 80  # number of epochs
DEVICE = 'cuda'  # 'cpu' or 'cuda'
DATA_SAVE_ROOT = '/scripts/tutorials/datasets'  # Directory for Saved dataset
PARAMS_SAVE_ROOT = '/scripts/tutorials/params'  # Directory for Saved parameters
SEQ_LENGTH = 40

# Confirm GPU availability
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if DEVICE == 'cpu':
    device = 'cpu'
# Set random seed
torch.manual_seed(SEED)
np.random.seed(SEED)

###### 1. Create dataset & Preprocessing ######
# Create tensors for dataset
x = np.linspace(0, 499, 500)
eps = np.random.randn(500) * 0.2
y = np.sin(x * 2 * np.pi / 50) + eps
plt.plot(x, y)
plt.title('Dataset')
plt.show()
# Create Sequence data
def make_sequence_data(y, num_sequence):
    num_data = len(y)
    seq_data = []
    target_data = []
    for i in range(num_data - num_sequence):
        seq_data.append(y[i:i+num_sequence])
        target_data.append(y[i+num_sequence:i+num_sequence+1])
    seq_arr = np.array(seq_data)
    target_arr = np.array(target_data)
    return seq_arr, target_arr
y_seq, y_target = make_sequence_data(y, SEQ_LENGTH)
print(y_seq.shape)
print(y_target.shape)
# Separate Train and test data
num_val = int(TEST_SIZE * y_seq.shape[0])
y_seq_train, y_target_train = y_seq[:-num_val], y_target[:-num_val]
y_seq_val, y_target_val = y_seq[-num_val:], y_target[-num_val:]
a = x[SEQ_LENGTH:-num_val]
b = y_target_train[:,0]
plt.plot(x[SEQ_LENGTH:-num_val], y_target_train[:,0], label='Train')
plt.plot(x[-num_val:], y_target_val[:,0], label='Test')
plt.title('Train and Test target data')
plt.legend()
plt.show()
# Convert to Tensor (Should reshape from (sample, seq) to (seq, sample, input_size))
y_seq_t = torch.FloatTensor(y_seq_train).permute(1, 0).unsqueeze(dim=-1)
y_target_t = torch.FloatTensor(y_target_train).permute(1, 0).unsqueeze(dim=-1)
y_seq_v = torch.FloatTensor(y_seq_val).permute(1, 0).unsqueeze(dim=-1)
y_target_v = torch.FloatTensor(y_target_val).permute(1, 0).unsqueeze(dim=-1)

###### 2. Define Model ######
# Define Model (https://debuggercafe.com/implementing-resnet18-in-pytorch-from-scratch/)
class LSTM(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=1, hidden_size=self.hidden_size)
        self.linear = nn.Linear(in_features=self.hidden_size, out_features=1)
    def forward(self, x):
        x, _ = self.lstm(x)
        x_last = x[-1]
        output = self.linear(x_last)
        return output

model = LSTM(100)
print(model)
# Send model to GPU
model.to(device)

###### 3. Define Criterion & Optimizer ######
criterion = nn.MSELoss()  # Criterion (MSE)
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Optimizer (Adam)

###### 4. Training ######
losses = []  # Array for storing loss (criterion)
val_losses = []  # Array for validation loss
start = time.time()  # For elapsed time
# Epoch loop
for epoch in range(NUM_EPOCHS):
    # Send data to GPU
    y_seq_t_minibatch = y_seq_t.to(device)  
    y_target_t_minibatch = y_target_t.to(device)
    # Update parameters (No minibatch)
    optimizer.zero_grad()  # Initialize gradient
    output = model(y_seq_t_minibatch)  # Forward (Prediction)
    loss = criterion(output, y_target_t_minibatch)  # Calculate criterion
    loss.backward()  # Backpropagation (Calculate gradient)
    optimizer.step()  # Update parameters (Based on optimizer algorithm)
    # Calculate running losses
    losses.append(loss.item())

    # Calculate validation metrics
    y_seq_v_minibatch = y_seq_v.to(device)
    y_target_v_minibatch = y_target_v.to(device)
    val_output = model(y_seq_v_minibatch)  # Forward (Prediction)
    val_loss = criterion(val_output, y_target_v_minibatch)  # Calculate criterion
    val_losses.append(val_loss.item())

    print(f'epoch: {epoch}, loss: {loss}, val_loss: {val_loss}, elapsed_time: {time.time() - start}')

###### 5. Model evaluation and visualization ######
# Plot loss history
plt.plot(losses, label='Train loss')
plt.plot(val_losses, label='Validation loss')
plt.title('Loss history')
plt.legend()
plt.show()
# Prediction of test data
y_pred = model(y_seq_v.to(device))
plt.plot(x, y, label='true')
plt.plot(np.arange(500-num_val, 500), y_pred.detach().to('cpu'), label='pred')
plt.xlim([400, 500])
plt.legend()
plt.show()

###### 6. Save the model ######
# Save parameters
params = model.state_dict()
if not os.path.exists(PARAMS_SAVE_ROOT):
    os.makedirs(PARAMS_SAVE_ROOT)
torch.save(params, f'{PARAMS_SAVE_ROOT}/dummysin_lstm.prm')
# Reload parameters
params_load = torch.load(f'{PARAMS_SAVE_ROOT}/dummysin_lstm.prm')
model_load = LSTM(100)
model_load.load_state_dict(params)

# %%

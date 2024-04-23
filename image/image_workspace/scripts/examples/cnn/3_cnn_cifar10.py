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
DATA_SAVE_ROOT = '/scripts/datasets/cnn'  # Directory for Saved dataset
PARAMS_SAVE_ROOT = '/scripts/params/cnn'  # Directory for Saved parameters
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

# %%

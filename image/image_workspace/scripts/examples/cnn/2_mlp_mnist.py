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
DATA_SAVE_ROOT = '/scripts/datasets/cnn'  # Directory for Saved dataset
PARAMS_SAVE_ROOT = '/scripts/params/cnn'  # Directory for Saved parameters

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

# %%

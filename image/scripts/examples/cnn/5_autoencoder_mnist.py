# %% MNIST + CNN-Autoencoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
import time
import os

SEED = 42
BATCH_SIZE = 100
NUM_EPOCHS = 15  # number of epochs
NUM_DISPLAYED_IMAGES = 10  # number of displayed images
MAX_DISPLAYED_CHANNEL = 8  # Maximum displayed channels of encoded image
NUM_LOAD_WORKERS = 2  # Number of workers for DataLoader
DEVICE = 'cuda'  # 'cpu' or 'cuda'
DATA_SAVE_ROOT = '/scripts/examples/cnn/datasets'  # Directory for Saved dataset
PARAMS_SAVE_ROOT = '/scripts/examples/cnn/params'  # Directory for Saved parameters

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

# %%

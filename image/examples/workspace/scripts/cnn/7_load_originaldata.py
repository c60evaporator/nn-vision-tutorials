# %% Original dataset
import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset, random_split
from torchvision import transforms
import matplotlib.pyplot as plt
import os
from zipfile import ZipFile
from PIL import Image

SEED = 42
TEST_SIZE = 0.25
BATCH_SIZE = 32
NUM_EPOCHS = 15  # number of epochs
NUM_DISPLAYED_IMAGES = 10  # number of displayed images
NUM_LOAD_WORKERS = 0  # Number of workers for DataLoader (Multiple workers not work in original dataset)
DEVICE = 'cuda'  # 'cpu' or 'cuda'
DATA_SAVE_ROOT = '/workspace/datasets/classification'  # Directory for Saved dataset
PARAMS_SAVE_ROOT = '/workspace/params/classification'  # Directory for Saved parameters
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

# %%

#%% coco128(YOLO format) + YOLOv5
# https://docs.ultralytics.com/yolov5/tutorials/pytorch_hub_model_loading/

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from IPython import get_ipython
import requests
import os
from zipfile import ZipFile
import yaml
import subprocess
import shutil

from cv_utils.detection_datasets import YoloDetectionTV
from cv_utils.detection_result_convs import convert_yolov5_result_to_torchvision
from cv_utils.show_torchvision import show_bounding_boxes, show_predicted_detection_minibatch

SEED = 42
BATCH_SIZE = 16  # Batch size
NUM_EPOCHS = 20  # number of epochs
NUM_DISPLAYED_IMAGES = 10  # number of displayed images
NUM_LOAD_WORKERS = 2  # Number of workers for DataLoader (Multiple workers not work in original dataset)
DEVICE = 'cuda'  # 'cpu' or 'cuda'
DATA_SAVE_ROOT = '/scripts/tutorials/datasets'  # Directory for Saved dataset
RESULTS_SAVE_ROOT = '/scripts/tutorials/results'
PARAMS_SAVE_ROOT = '/scripts/tutorials/params'  # Directory for Saved parameters
DATA_YAML_URL = 'https://raw.githubusercontent.com/ultralytics/yolov5/master/data/coco128.yaml'  # coco128 data with YOLO format. The train data and the val data are the same.
TRAIN_SCRIPT_PATH = '/repos/yolov5/train.py'  # Training script (Clone from https://github.com/ultralytics/yolov5)
PRETRAINED_WEIGHT = '/scripts/tutorials/pretrained_weights/yolov5m.pt'  # Pretrained weight for YOLOv5 (Download from https://github.com/ultralytics/yolov5#pretrained-checkpoints)

# Confirm GPU availability
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if DEVICE == 'cpu':
    device = 'cpu'
# Set random seed
torch.manual_seed(SEED)

###### 1. Create dataset & Preprocessing ######
# Download dataset yaml file
res = requests.get(DATA_YAML_URL, allow_redirects=True)
yaml_path = f'{DATA_SAVE_ROOT}/{DATA_YAML_URL.split("/")[-1]}'
open(yaml_path, 'wb').write(res.content)
with open(yaml_path, 'r') as file:
    dataset_yml = yaml.safe_load(file)
dataset_url = dataset_yml['download']
# Download dataset (https://www.tutorialspoint.com/downloading-files-from-web-using-python)
res = requests.get(dataset_url, allow_redirects=True)
zip_path = f'{DATA_SAVE_ROOT}/{dataset_url.split("/")[-1]}'
open(zip_path, 'wb').write(res.content)
data_dir = os.path.splitext(zip_path)[0]
# Unzip dataset
with ZipFile(zip_path, 'r') as z:
    z.extractall(path=DATA_SAVE_ROOT)

# Define class names
idx_to_class = dataset_yml['names']
# Display images in the first mini-batch
display_transform = transforms.Compose([
    transforms.ToTensor()  # Convert from range [0, 255] to a torch.FloatTensor in the range [0.0, 1.0]
])
def collate_fn(batch):
    return tuple(zip(*batch))
display_dataset = YoloDetectionTV(root = f'{DATA_SAVE_ROOT}/coco128/images/train2017',
                                  ann_dir = f'{DATA_SAVE_ROOT}/coco128/labels/train2017',
                                  transform=display_transform)
display_loader = DataLoader(display_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_LOAD_WORKERS, collate_fn=collate_fn)
display_iter = iter(display_loader)
imgs, targets = next(display_iter)
for i, (img, target) in enumerate(zip(imgs, targets)):
    img = (img*255).to(torch.uint8)  # Change from float[0, 1] to uint[0, 255]
    boxes, labels = target['boxes'], target['labels']
    show_bounding_boxes(img, boxes, labels=labels, idx_to_class=idx_to_class)
    plt.show()

# Load validation dataset
val_dataset = YoloDetectionTV(root = f'{DATA_SAVE_ROOT}/coco128/images/train2017',
                              ann_dir = f'{DATA_SAVE_ROOT}/coco128/labels/train2017',
                              transform=display_transform)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_LOAD_WORKERS, collate_fn=collate_fn)

###### 2. Define Model ######
###### 3. Define Criterion & Optimizer ######
###### 4. Training ######
# Note: the format of the training dataset should be YOLO format
# Train (Options: https://github.com/ultralytics/yolov5/blob/master/train.py#L442)
train_command = f'python3 {TRAIN_SCRIPT_PATH} --data {yaml_path} --epochs {NUM_EPOCHS} --weights {PRETRAINED_WEIGHT} --batch-size {BATCH_SIZE} --seed {SEED} --project {RESULTS_SAVE_ROOT}/yolov5'
subprocess.run(train_command, shell=True)
# Save the weights
result_dir = f'{RESULTS_SAVE_ROOT}/yolov5/{sorted(os.listdir(f"{RESULTS_SAVE_ROOT}/yolov5"))[-1]}'
os.makedirs(f'{PARAMS_SAVE_ROOT}/yolov5', exist_ok=True)
model_weight_name = f'{os.path.splitext(os.path.basename(DATA_YAML_URL))[0]}_{os.path.splitext(os.path.basename(PRETRAINED_WEIGHT))[0]}.pt'
shutil.copy(f'{result_dir}/weights/best.pt', f'{PARAMS_SAVE_ROOT}/yolov5/{model_weight_name}')

###### 5. Model evaluation and visualization ######


###### Inference in the first mini-batch ######
# Load a model with the trained weight
model = torch.hub.load('ultralytics/yolov5', 'custom', path=f'{PARAMS_SAVE_ROOT}/yolov5/{model_weight_name}')
# Send the model to GPU
model.to(device)
val_iter = iter(val_loader)
imgs, targets = next(val_iter)  # Load the first batch
# Inference
images_fps = [target['image_path'] for target in targets]  # Get the image pathes
results = model(images_fps)
# Show Results
results.show()
get_ipython().magic('matplotlib inline')  # Matplotlib inline should be enabled to show plots after commiting YOLO inference
# Convert the Results to Torchvision object detection prediction format
predictions = convert_yolov5_result_to_torchvision(results)

# Show predicted images
show_predicted_detection_minibatch(imgs, predictions, targets, idx_to_class, max_displayed_images=NUM_DISPLAYED_IMAGES)

#%% coco128(YOLO format) + YOLOv8
from ultralytics import YOLO
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from IPython import get_ipython
import requests
import os
from zipfile import ZipFile
import yaml
import shutil
from datetime import datetime

from cv_utils.detection_datasets import YoloDetectionTV
from cv_utils.detection_result_convs import convert_yolov8_result_to_torchvision
from cv_utils.show_torchvision import show_bounding_boxes, show_predicted_detection_minibatch

SEED = 42
BATCH_SIZE = 16  # Batch size
NUM_EPOCHS = 10  # number of epochs
NUM_DISPLAYED_IMAGES = 10  # number of displayed images
NUM_LOAD_WORKERS = 2  # Number of workers for DataLoader (Multiple workers not work in original dataset)
DEVICE = 'cuda'  # 'cpu' or 'cuda'
DATA_SAVE_ROOT = '/scripts/tutorials/datasets'  # Directory for Saved dataset
RESULTS_SAVE_ROOT = '/scripts/tutorials/results'
PARAMS_SAVE_ROOT = '/scripts/tutorials/params'  # Directory for Saved parameters
MODEL_YAML_URL = '/scripts/tutorials/configs/yolov8.yaml'  # YOLOv8 Model yaml file (Download from https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models/v8)
DATA_YAML_URL = 'https://raw.githubusercontent.com/ultralytics/yolov5/master/data/coco128.yaml'  # coco128 data with YOLO format. The train data and the val data are the same.
PRETRAINED_WEIGHT = 'yolov8n.pt'  # Pretrained weight for YOLOv8 (Select from https://github.com/ultralytics/ultralytics#models)

###### 1. Create dataset & Preprocessing (The same as YOLOv5) ######
# Download dataset yaml file
res = requests.get(DATA_YAML_URL, allow_redirects=True)
yaml_path = f'{DATA_SAVE_ROOT}/{DATA_YAML_URL.split("/")[-1]}'
open(yaml_path, 'wb').write(res.content)
with open(yaml_path, 'r') as file:
    dataset_yml = yaml.safe_load(file)
dataset_url = dataset_yml['download']
# Download dataset (https://www.tutorialspoint.com/downloading-files-from-web-using-python)
res = requests.get(dataset_url, allow_redirects=True)
zip_path = f'{DATA_SAVE_ROOT}/{dataset_url.split("/")[-1]}'
open(zip_path, 'wb').write(res.content)
data_dir = os.path.splitext(zip_path)[0]
# Unzip dataset
with ZipFile(zip_path, 'r') as z:
    z.extractall(path=DATA_SAVE_ROOT)

# Define class names
idx_to_class = dataset_yml['names']
# Display images in the first mini-batch
display_transform = transforms.Compose([
    transforms.ToTensor()  # Convert from range [0, 255] to a torch.FloatTensor in the range [0.0, 1.0]
])
def collate_fn(batch):
    return tuple(zip(*batch))
display_dataset = YoloDetectionTV(root = f'{DATA_SAVE_ROOT}/coco128/images/train2017',
                                  ann_dir = f'{DATA_SAVE_ROOT}/coco128/labels/train2017',
                                  transform=display_transform)
display_loader = DataLoader(display_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_LOAD_WORKERS, collate_fn=collate_fn)
display_iter = iter(display_loader)
imgs, targets = next(display_iter)
for i, (img, target) in enumerate(zip(imgs, targets)):
    img = (img*255).to(torch.uint8)  # Change from float[0, 1] to uint[0, 255]
    boxes, labels = target['boxes'], target['labels']
    show_bounding_boxes(img, boxes, labels=labels, idx_to_class=idx_to_class)
    plt.show()

# Load validation dataset
val_dataset = YoloDetectionTV(root = f'{DATA_SAVE_ROOT}/coco128/images/train2017',
                              ann_dir = f'{DATA_SAVE_ROOT}/coco128/labels/train2017',
                              transform=display_transform)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_LOAD_WORKERS, collate_fn=collate_fn)

###### 2. Define Model ######
###### 3. Define Criterion & Optimizer ######
###### 4. Training ######
# Note: the format of the training dataset should be YOLO format
# https://docs.ultralytics.com/modes/train/
result_dir = f'{RESULTS_SAVE_ROOT}/yolov8/{datetime.now().strftime("%Y%m%d%H%M%S")}'
#model = YOLO(MODEL_YAML_URL).load(PRETRAINED_WEIGHT)
model = YOLO(PRETRAINED_WEIGHT)
model.train(data="coco128.yaml", epochs=NUM_EPOCHS, batch=BATCH_SIZE, seed=SEED, project=result_dir)  # train the model
metrics = model.val()  # evaluate model performance on the validation set
# Save the weights
os.makedirs(f'{PARAMS_SAVE_ROOT}/yolov8', exist_ok=True)
model_weight_name = f'{os.path.splitext(os.path.basename(DATA_YAML_URL))[0]}_{os.path.splitext(os.path.basename(PRETRAINED_WEIGHT))[0]}.pt'
shutil.copy(f'{result_dir}/train/weights/best.pt', f'{PARAMS_SAVE_ROOT}/yolov8/{model_weight_name}')

###### Inference in the first mini-batch ######
# Load a model with the trained weight
model_trained = YOLO(f'{PARAMS_SAVE_ROOT}/yolov8/{model_weight_name}')
# Load the first mini-batch
val_iter = iter(val_loader)
imgs, targets = next(val_iter)
# Inference
images_fps = [target['image_path'] for target in targets]  # Get the image pathes
results = model_trained(images_fps)

# Show Results
results[0].show()
get_ipython().magic('matplotlib inline')  # Matplotlib inline should be enabled to show plots after commiting YOLO inference
# Convert the Results to Torchvision object detection prediction format
predictions = convert_yolov8_result_to_torchvision(results)

# Show predicted images
show_predicted_detection_minibatch(imgs, predictions, targets, idx_to_class, max_displayed_images=NUM_DISPLAYED_IMAGES)

#%% coco128(YOLO format) + YOLOX
# https://docs.ultralytics.com/yolov5/tutorials/pytorch_hub_model_loading/

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from IPython import get_ipython
import requests
import os
from zipfile import ZipFile
import yaml
import subprocess
import shutil

from cv_utils.detection_datasets import YoloDetectionTV
from cv_utils.detection_result_convs import convert_yolov5_result_to_torchvision
from cv_utils.show_torchvision import show_bounding_boxes, show_predicted_detection_minibatch

SEED = 42
BATCH_SIZE = 16  # Batch size
NUM_EPOCHS = 20  # number of epochs
NUM_DISPLAYED_IMAGES = 10  # number of displayed images
NUM_LOAD_WORKERS = 2  # Number of workers for DataLoader (Multiple workers not work in original dataset)
DEVICE = 'cuda'  # 'cpu' or 'cuda'
DATA_SAVE_ROOT = '/scripts/tutorials/datasets'  # Directory for Saved dataset
RESULTS_SAVE_ROOT = '/scripts/tutorials/results'
PARAMS_SAVE_ROOT = '/scripts/tutorials/params'  # Directory for Saved parameters
DATA_YAML_URL = 'https://raw.githubusercontent.com/ultralytics/yolov5/master/data/coco128.yaml'  # coco128 data with YOLO format. The train data and the val data are the same.
TRAIN_SCRIPT_PATH = '/repos/yolov5/train.py'  # Training script (Clone from https://github.com/ultralytics/yolov5)
PRETRAINED_WEIGHT = '/scripts/tutorials/pretrained_weights/yolov5m.pt'  # Pretrained weight for YOLOv5 (Download from https://github.com/ultralytics/yolov5#pretrained-checkpoints)

# Confirm GPU availability
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if DEVICE == 'cpu':
    device = 'cpu'
# Set random seed
torch.manual_seed(SEED)

###### 1. Create dataset & Preprocessing ######
# Download dataset yaml file
res = requests.get(DATA_YAML_URL, allow_redirects=True)
yaml_path = f'{DATA_SAVE_ROOT}/{DATA_YAML_URL.split("/")[-1]}'
open(yaml_path, 'wb').write(res.content)
with open(yaml_path, 'r') as file:
    dataset_yml = yaml.safe_load(file)
dataset_url = dataset_yml['download']
# Download dataset (https://www.tutorialspoint.com/downloading-files-from-web-using-python)
res = requests.get(dataset_url, allow_redirects=True)
zip_path = f'{DATA_SAVE_ROOT}/{dataset_url.split("/")[-1]}'
open(zip_path, 'wb').write(res.content)
data_dir = os.path.splitext(zip_path)[0]
# Unzip dataset
with ZipFile(zip_path, 'r') as z:
    z.extractall(path=DATA_SAVE_ROOT)

# Define class names
idx_to_class = dataset_yml['names']
# Display images in the first mini-batch
display_transform = transforms.Compose([
    transforms.ToTensor()  # Convert from range [0, 255] to a torch.FloatTensor in the range [0.0, 1.0]
])
def collate_fn(batch):
    return tuple(zip(*batch))
display_dataset = YoloDetectionTV(root = f'{DATA_SAVE_ROOT}/coco128/images/train2017',
                                  ann_dir = f'{DATA_SAVE_ROOT}/coco128/labels/train2017',
                                  transform=display_transform)
display_loader = DataLoader(display_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_LOAD_WORKERS, collate_fn=collate_fn)
display_iter = iter(display_loader)
imgs, targets = next(display_iter)
for i, (img, target) in enumerate(zip(imgs, targets)):
    img = (img*255).to(torch.uint8)  # Change from float[0, 1] to uint[0, 255]
    boxes, labels = target['boxes'], target['labels']
    show_bounding_boxes(img, boxes, labels=labels, idx_to_class=idx_to_class)
    plt.show()

# Load validation dataset
val_dataset = YoloDetectionTV(root = f'{DATA_SAVE_ROOT}/coco128/images/train2017',
                              ann_dir = f'{DATA_SAVE_ROOT}/coco128/labels/train2017',
                              transform=display_transform)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_LOAD_WORKERS, collate_fn=collate_fn)

###### 2. Define Model ######
###### 3. Define Criterion & Optimizer ######
###### 4. Training ######
# Note: the format of the training dataset should be YOLO format
# Train (Options: https://github.com/ultralytics/yolov5/blob/master/train.py#L442)
train_command = f'python3 {TRAIN_SCRIPT_PATH} --data {yaml_path} --epochs {NUM_EPOCHS} --weights {PRETRAINED_WEIGHT} --batch-size {BATCH_SIZE} --seed {SEED} --project {RESULTS_SAVE_ROOT}/yolov5'
subprocess.run(train_command, shell=True)
# Save the weights
result_dir = f'{RESULTS_SAVE_ROOT}/yolov5/{sorted(os.listdir(f"{RESULTS_SAVE_ROOT}/yolov5"))[-1]}'
os.makedirs(f'{PARAMS_SAVE_ROOT}/yolov5', exist_ok=True)
model_weight_name = f'{os.path.splitext(os.path.basename(DATA_YAML_URL))[0]}_{os.path.splitext(os.path.basename(PRETRAINED_WEIGHT))[0]}.pt'
shutil.copy(f'{result_dir}/weights/best.pt', f'{PARAMS_SAVE_ROOT}/yolov5/{model_weight_name}')

###### 5. Model evaluation and visualization ######


###### Inference in the first mini-batch ######
# Load a model with the trained weight
model = torch.hub.load('ultralytics/yolov5', 'custom', path=f'{PARAMS_SAVE_ROOT}/yolov5/{model_weight_name}')
# Send the model to GPU
model.to(device)
val_iter = iter(val_loader)
imgs, targets = next(val_iter)  # Load the first batch
# Inference
images_fps = [target['image_path'] for target in targets]  # Get the image pathes
results = model(images_fps)
# Show Results
results.show()
get_ipython().magic('matplotlib inline')  # Matplotlib inline should be enabled to show plots after commiting YOLO inference
# Convert the Results to Torchvision object detection prediction format
predictions = convert_yolov5_result_to_torchvision(results)

# Show predicted images
show_predicted_detection_minibatch(imgs, predictions, targets, idx_to_class, max_displayed_images=NUM_DISPLAYED_IMAGES)
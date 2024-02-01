#%% coco128 + YOLOv5
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
from cv_utils.torchhub_utils import convert_yolo_result_to_torchvision
from cv_utils.show_torchvision import show_bounding_boxes, show_predicted_detection_minibatch

SEED = 42
BATCH_SIZE = 16  # Batch size
NUM_EPOCHS = 20  # number of epochs
NUM_DISPLAYED_IMAGES = 10  # number of displayed images
NUM_LOAD_WORKERS = 2  # Number of workers for DataLoader (Multiple workers not work in original dataset)
DEVICE = 'cuda'  # 'cpu' or 'cuda'
DATA_SAVE_ROOT = '/scripts/tutorials/datasets'  # Directory for Saved dataset
YAML_DOWNLOAD_URL = 'https://raw.githubusercontent.com/ultralytics/yolov5/master/data/coco128.yaml'  # Note: The train data and the val data are the same in coco128
RESULTS_SAVE_ROOT = '/scripts/tutorials/results'
PARAMS_SAVE_ROOT = '/scripts/tutorials/params'  # Directory for Saved parameters
TRAIN_SCRIPT_PATH = '/repos/yolov5/train.py'
PRETRAINED_WEIGHT = 'yolov5m.pt'  # (https://github.com/ultralytics/yolov5#documentation)

# Confirm GPU availability
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if DEVICE == 'cpu':
    device = 'cpu'
# Set random seed
torch.manual_seed(SEED)

###### 1. Create dataset & Preprocessing ######
# Download dataset yaml file
res = requests.get(YAML_DOWNLOAD_URL, allow_redirects=True)
yaml_path = f'{DATA_SAVE_ROOT}/{YAML_DOWNLOAD_URL.split("/")[-1]}'
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
# Train (Options: https://github.com/ultralytics/yolov5/blob/master/train.py#L442)
train_command = f'python3 {TRAIN_SCRIPT_PATH} --data {yaml_path} --epochs {NUM_EPOCHS} --weights {PRETRAINED_WEIGHT} --batch-size {BATCH_SIZE} --project {RESULTS_SAVE_ROOT}'
subprocess.run(train_command, shell=True)
# Save the weights
result_dir = sorted(os.listdir(RESULTS_SAVE_ROOT))[-1]
shutil.copy(f'{RESULTS_SAVE_ROOT}/{result_dir}/weights/best.pt', f'{PARAMS_SAVE_ROOT}/coco128_yolov5.pt')

###### 5. Model evaluation and visualization ######


###### Inference in the first mini-batch ######
# Load a model with the trained weight
model = torch.hub.load('ultralytics/yolov5', 'custom', path=f'{PARAMS_SAVE_ROOT}/coco128_yolov5.pt')
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
predictions = convert_yolo_result_to_torchvision(results)

# Show predicted images
show_predicted_detection_minibatch(imgs, predictions, targets, idx_to_class, max_displayed_images=NUM_DISPLAYED_IMAGES)

#%% YOLOv8
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

model.train(data="coco128.yaml", epochs=3)  # train the model
metrics = model.val()  # evaluate model performance on the validation set
# Images
img = 'https://ultralytics.com/images/bus.jpg'  # or file, Path, PIL, OpenCV, numpy, list

# Inference
results = model(img)

# Results
results.print()  # or .show(), .save(), .crop(), .pandas(), etc.
# %%

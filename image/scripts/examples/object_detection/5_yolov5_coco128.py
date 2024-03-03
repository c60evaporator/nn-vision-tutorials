#%% coco128(YOLO format) + YOLOv5
# https://docs.ultralytics.com/yolov5/examples/pytorch_hub_model_loading/
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from IPython import get_ipython
import requests
import time
import os
from zipfile import ZipFile
import yaml
import subprocess
import shutil
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from torch_extend.detection.dataset import YoloDetectionTV
from torch_extend.detection.result_converter import convert_yolov5_hub_result
from torch_extend.detection.display import show_bounding_boxes, show_predicted_detection_minibatch

SEED = 42
BATCH_SIZE = 16  # Batch size
NUM_EPOCHS = 20  # number of epochs
NUM_DISPLAYED_IMAGES = 10  # number of displayed images
NUM_LOAD_WORKERS = 2  # Number of workers for DataLoader (Multiple workers not work in original dataset)
DEVICE = 'cuda'  # 'cpu' or 'cuda'
DATA_SAVE_ROOT = '/scripts/examples/object_detection/datasets'  # Directory for Saved dataset
RESULTS_SAVE_ROOT = '/scripts/examples/object_detection/results'
PARAMS_SAVE_ROOT = '/scripts/examples/object_detection/params'  # Directory for Saved parameters
DATA_YAML_URL = 'https://raw.githubusercontent.com/ultralytics/yolov5/master/data/coco128.yaml'  # coco128 data with YOLO format. The train data and the val data are the same.
TRAIN_SCRIPT_PATH = '/repos/yolov5/train.py'  # Training script (Clone from https://github.com/ultralytics/yolov5)
PRETRAINED_WEIGHT = '/scripts/examples/object_detection/pretrained_weights/yolov5m.pt'  # Pretrained weight for YOLOv5 (Download from https://github.com/ultralytics/yolov5#pretrained-checkpoints)

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

# Define display loader
display_transform = transforms.Compose([
    transforms.ToTensor()  # Convert from range [0, 255] to a torch.FloatTensor in the range [0.0, 1.0]
])
def collate_fn(batch):
    return tuple(zip(*batch))
display_dataset = YoloDetectionTV(root = f'{DATA_SAVE_ROOT}/coco128/images/train2017',
                                  ann_dir = f'{DATA_SAVE_ROOT}/coco128/labels/train2017',
                                  transform=display_transform)
display_loader = DataLoader(display_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_LOAD_WORKERS, collate_fn=collate_fn)
# Define class names
idx_to_class = dataset_yml['names']
# Display images in the first mini-batch
display_iter = iter(display_loader)
imgs, targets = next(display_iter)
for i, (img, target) in enumerate(zip(imgs, targets)):
    img = (img*255).to(torch.uint8)  # Change from float[0, 1] to uint[0, 255]
    boxes, labels = target['boxes'], target['labels']
    show_bounding_boxes(img, boxes, labels=labels, idx_to_class=idx_to_class)
    plt.show()

# Load validation dataset
val_transform = transforms.Compose([
    transforms.ToTensor()  # Convert from range [0, 255] to a torch.FloatTensor in the range [0.0, 1.0]
])
val_dataset = YoloDetectionTV(root = f'{DATA_SAVE_ROOT}/coco128/images/train2017',
                              ann_dir = f'{DATA_SAVE_ROOT}/coco128/labels/train2017',
                              transform=val_transform)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_LOAD_WORKERS, collate_fn=collate_fn)

###### 2. Define Model ######
###### 3. Define Criterion & Optimizer ######
###### 4. Training ######
# Note: the format of the training dataset should be YOLO format
start = time.time()  # For elapsed time
# Train (Options: https://github.com/ultralytics/yolov5/blob/master/train.py#L442)
train_command = f'python3 {TRAIN_SCRIPT_PATH} --data {yaml_path} --epochs {NUM_EPOCHS} --weights {PRETRAINED_WEIGHT} --batch-size {BATCH_SIZE} --seed {SEED} --project {RESULTS_SAVE_ROOT}/yolov5'
subprocess.run(train_command, shell=True)
print(f'Training complete, elapsed_time={time.time() - start}')
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
image_fps = [target['image_path'] for target in targets]  # Get the image pathes
results = model(image_fps)
# Show Results
results.show()
get_ipython().magic('matplotlib inline')  # Matplotlib inline should be enabled to show plots after commiting YOLO inference
# Convert the Results to Torchvision object detection prediction format
predictions = convert_yolov5_hub_result(results)

# Show predicted images
show_predicted_detection_minibatch(imgs, predictions, targets, idx_to_class, max_displayed_images=NUM_DISPLAYED_IMAGES)

# %%

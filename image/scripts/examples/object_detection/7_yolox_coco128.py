#%% mini-coco128 + YOLOX
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import time
import os
import subprocess
import shutil
import sys

from configs.yolox_exp_train import Exp

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from torch_extend.detection.yolox_utils import val_transform_to_yolox, convert_yolox_result_to_torchvision, inference, get_img_info, train
from torch_extend.detection.dataset import CocoDetectionTV
from torch_extend.detection.data_converter import convert_yolo2coco
from torch_extend.detection.display import show_bounding_boxes, show_predicted_detection_minibatch, show_average_precisions
from torch_extend.detection.metrics import average_precisions

TRAIN_BY_COMMAND = False  # If True, use the console command for the training. If False use the function for the training

SEED = 42
BATCH_SIZE = 16  # Batch size
NUM_DISPLAYED_IMAGES = 10  # number of displayed images
NUM_LOAD_WORKERS = 2  # Number of workers for DataLoader (Only used for displaying images)
DEVICE = 'cuda'  # 'cpu' or 'cuda'
DATA_SAVE_ROOT = '/scripts/examples/object_detection/datasets'  # Directory for Saved dataset
RESULTS_SAVE_ROOT = '/scripts/examples/object_detection/results'
PARAMS_SAVE_ROOT = '/scripts/examples/object_detection/params'  # Directory for Saved parameters
DATA_YAML_URL = 'https://raw.githubusercontent.com/ultralytics/yolov5/master/data/coco128.yaml'  # coco128 data with YOLO format. The train data and the val data are the same.
YOLOX_ROOT = '/repos/YOLOX'  # YOLOX (Clone from https://github.com/Megvii-BaseDetection/YOLOX)
EXP_SCRIPT_PATH = '/scripts/examples/object_detection/configs/yolox_exp_train.py'  # Exp file path (https://github.com/Megvii-BaseDetection/YOLOX/blob/main/docs/train_custom_data.md#2-create-your-exp-file-to-control-everything)
TRAIN_SCRIPT_PATH = 'tools/train.py'  # Training script path (relative path from YOLOX_ROOT)
PRETRAINED_WEIGHT = 'pretrained_weights/yolox_s.pth'  # Pretrained weight for YOLOX (Download from https://github.com/Megvii-BaseDetection/YOLOX/tree/main?tab=readme-ov-file#benchmark)

# Confirm GPU availability
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if DEVICE == 'cpu':
    device = 'cpu'
# Set random seed
torch.manual_seed(SEED)
# Print device count
print(f'device_count={torch.cuda.device_count()}')

###### 1. Create dataset & Preprocessing######
# # Download dataset yaml file
# res = requests.get(DATA_YAML_URL, allow_redirects=True)
# yaml_path = f'{DATA_SAVE_ROOT}/{DATA_YAML_URL.split("/")[-1]}'
# open(yaml_path, 'wb').write(res.content)
# with open(yaml_path, 'r') as file:
#     dataset_yml = yaml.safe_load(file)
# dataset_url = dataset_yml['download']
# # Download dataset (https://www.tutorialspoint.com/downloading-files-from-web-using-python)
# res = requests.get(dataset_url, allow_redirects=True)
# zip_path = f'{DATA_SAVE_ROOT}/{dataset_url.split("/")[-1]}'
# open(zip_path, 'wb').write(res.content)
# data_dir = os.path.splitext(zip_path)[0]
# # Unzip dataset
# with ZipFile(zip_path, 'r') as z:
#     z.extractall(path=DATA_SAVE_ROOT)

# # Convert the dataset to COCO format
# convert_yolo2coco(yolo_yaml=yaml_path, yolo_root_dir=f'{DATA_SAVE_ROOT}/coco128', output_dir=f'{DATA_SAVE_ROOT}/coco128_coco')

# Define display loader
display_transform = transforms.Compose([
    transforms.ToTensor()  # Convert from range [0, 255] to a torch.FloatTensor in the range [0.0, 1.0]
])
def collate_fn(batch):
    return tuple(zip(*batch))
display_dataset = CocoDetectionTV(root = f'{DATA_SAVE_ROOT}/mini-coco128/train2017',
                                  annFile = f'{DATA_SAVE_ROOT}/mini-coco128/annotations/instances_train2017.json',
                                  transform=display_transform)
display_loader = DataLoader(display_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_LOAD_WORKERS, collate_fn=collate_fn)
# Define class names
idx_to_class = {
    v['id']: v['name']
    for k, v in display_dataset.coco.cats.items()
}
indices = [idx for idx in idx_to_class.keys()]
na_cnt = 0
for i in range(max(indices)):
    if i not in indices:
        na_cnt += 1
        idx_to_class[i] = f'NA{"{:02}".format(na_cnt)}'
# Display images in the first mini-batch
display_iter = iter(display_loader)
imgs, targets = next(display_iter)
for i, (img, target) in enumerate(zip(imgs, targets)):
    img = (img*255).to(torch.uint8)  # Change from float[0, 1] to uint[0, 255]
    boxes, labels = target['boxes'], target['labels']
    show_bounding_boxes(img, boxes, labels=labels, idx_to_class=idx_to_class)
    plt.show()

###### 2. Define Model ######
###### 3. Define Criterion & Optimizer ######
###### 4. Training ######
# Reference (https://github.com/Megvii-BaseDetection/YOLOX/blob/main/docs/train_custom_data.md#3-train)
# Note: the format of the training dataset should be YOLO format
start = time.time()  # For elapsed time

if TRAIN_BY_COMMAND:
    # Train by console command
    train_command = f'python3 {YOLOX_ROOT}/{TRAIN_SCRIPT_PATH} -f {EXP_SCRIPT_PATH} -d 1 -b {BATCH_SIZE} --fp16 -o -c {PRETRAINED_WEIGHT}'
    print(train_command)
    subprocess.run(train_command, shell=True)
else:
    # Train by the function
    train(exp_file=EXP_SCRIPT_PATH, devices=1, batch_size=BATCH_SIZE, fp16=True, occupy=True, ckpt=PRETRAINED_WEIGHT)

print(f'Training complete, elapsed_time={time.time() - start}')
# Save the weights
result_dir = f'{RESULTS_SAVE_ROOT}/yolox/{sorted(os.listdir(f"{RESULTS_SAVE_ROOT}/yolox"))[-1]}'
os.makedirs(f'{PARAMS_SAVE_ROOT}/yolox', exist_ok=True)
model_weight_name = f'{os.path.basename(result_dir).split("_")[2]}_{os.path.basename(result_dir).split("_")[1]}.pth'
print(model_weight_name)
shutil.copy(f'{result_dir}/best_ckpt.pth', f'{PARAMS_SAVE_ROOT}/yolox/{model_weight_name}')

###### 5. Model evaluation and visualization ######
# Get YOLOX experiment
from configs.yolox_exp_inference import Exp
experiment = Exp()
# Load validation dataset
val_transform = transforms.Lambda(lambda x: val_transform_to_yolox(x, experiment.test_size))
val_dataset = CocoDetectionTV(root = f'{DATA_SAVE_ROOT}/mini-coco128/val2017',
                              annFile = f'{DATA_SAVE_ROOT}/mini-coco128/annotations/instances_val2017.json')
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_LOAD_WORKERS, collate_fn=collate_fn)

###### Inference in the first mini-batch ######
# Load the model (https://www.kaggle.com/code/max237/getting-started-with-yolox-inference-only)
model_trained = experiment.get_model()
# Send the model to GPU
model_trained.to(device)
# Turn off training mode so the model so it won't try to calculate loss
model_trained.eval()
model_trained.head.training=False
model_trained.training=False
# Load the weights from training
best_weights = torch.load(f'{PARAMS_SAVE_ROOT}/yolox/{model_weight_name}')
# best_weights = torch.load(f'/scripts/examples/pretrained_weights/yolox_s.pth')
model_trained.load_state_dict(best_weights['model'])
# Load the first mini-batch
val_iter = iter(val_loader)
imgs, targets = next(val_iter)
imgs_transformed = [val_transform(img) for img in imgs]
imgs_gpu = [img.to(device) for img in imgs_transformed]
# Inference (https://github.com/Megvii-BaseDetection/YOLOX/blob/main/tools/demo.py#L132)
results = []
for img in imgs_gpu:
    outputs = inference(img, model_trained, experiment)
    results.append(outputs)
# Convert the Results to Torchvision object detection prediction format
imgs_info = [get_img_info(img, experiment.test_size) for img in imgs]
predictions = convert_yolox_result_to_torchvision(results, imgs_info)
# Class names dict with background
idx_to_class_bg = {k: v for k, v in idx_to_class.items()}
idx_to_class_bg[-1] = 'background'
# Show predicted images
imgs_display = [display_transform(img) for img in imgs]
show_predicted_detection_minibatch(imgs_display, predictions, targets, idx_to_class, max_displayed_images=NUM_DISPLAYED_IMAGES)

#%%
###### Calculate mAP ######
targets_list = []
predictions_list = []
start = time.time()  # For elapsed time
for i, (imgs, targets) in enumerate(val_loader):
    imgs_transformed = [val_transform(img) for img in imgs]
    imgs_gpu = [img.to(device) for img in imgs_transformed]
    # Inference
    results = []
    for img in imgs_gpu:
        outputs = inference(img, model_trained, experiment)
        results.append(outputs)
    # Convert the Results to Torchvision object detection prediction format
    predictions = convert_yolox_result_to_torchvision(results, imgs_info)
    # Store the result
    targets_list.extend(targets)
    predictions_list.extend(predictions)
    if i%100 == 0:  # Show progress every 100 images
        print(f'Prediction for mAP: {i}/{len(val_loader)} batches, elapsed_time: {time.time() - start}')
aps = average_precisions(predictions_list, targets_list, idx_to_class_bg, iou_threshold=0.5, conf_threshold=0.2)
# Show mAP
show_average_precisions(aps)

# %%

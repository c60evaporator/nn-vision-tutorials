# %% COCO Detection + DETR (ResNet50) + No training
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import to_tensor
import matplotlib.pyplot as plt
import time
import os
import shutil
from datetime import datetime
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from torch_extend.detection.display import show_bounding_boxes, show_predicted_detection_minibatch, show_average_precisions
from torch_extend.detection.target_converter import resize_target
from torch_extend.detection.dataset import CocoDetectionTV
from torch_extend.detection.torchhub_utils import convert_detr_hub_result
from torch_extend.detection.metrics import average_precisions

SEED = 42
BATCH_SIZE = 1  # Batch size
NUM_EPOCHS = 10  # number of epochs
NUM_DISPLAYED_IMAGES = 10  # number of displayed images
NUM_LOAD_WORKERS = 2  # Number of workers for DataLoader (Multiple workers need much memory, so if the error "RuntimeError: DataLoader worker (pid ) is killed by signal" occurs, you should set it 0)
DEVICE = 'cuda'  # 'cpu' or 'cuda'
DATA_SAVE_ROOT = '/workspace/datasets/object_detection'  # Directory for Saved dataset
DATASET_NAME = 'mini-coco128'
RESULTS_SAVE_ROOT = '/workspace/scripts/object_detection/results'
PARAMS_SAVE_ROOT = '/workspace/params/object_detection'  # Directory for Saved parameters
DETR_ROOT = '/repos/DETR/detr'  # YOLOX (Clone from https://github.com/Megvii-BaseDetection/YOLOX)
PRETRAINED_WEIGHT = '/workspace/pretrained_weights/object_detection/detr-r50-e632da11.pth'  # Pretrained weight for DETR (Download from https://github.com/facebookresearch/detr/tree/main?tab=readme-ov-file#model-zoo)
SAME_IMG_SIZE = False  # Whether the resized image sizes are the same or not
PROB_THRESHOLD = 0.8  # Threshold for the class probability

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Confirm GPU availability
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if DEVICE == 'cpu':
    device = 'cpu'
# Set random seed
torch.manual_seed(SEED)

###### 1. Showing dataset ######
# Define display loader
display_transform = transforms.Compose([
    transforms.ToTensor()  # Convert from range [0, 255] to a torch.FloatTensor in the range [0.0, 1.0]
])
def collate_fn(batch):
    return tuple(zip(*batch))
display_dataset = CocoDetectionTV(root = f'{DATA_SAVE_ROOT}/COCO/val2017',
                                  annFile = f'{DATA_SAVE_ROOT}/COCO/annotations/instances_val2017.json',
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
    labels = [idx_to_class[label.item()] for label in labels]  # Change labels from index to str
    show_bounding_boxes(img, boxes, labels=labels)
    plt.show()
# Load validation dataset
val_transform = transforms.Compose([
    transforms.Resize(800),  # Resize an image to fit the short side to 800px
    transforms.ToTensor(),  # Convert from range [0, 255] to a torch.FloatTensor in the range [0.0, 1.0]
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)  # Normalization (mean and std of the imagenet dataset for normalizing)
])
val_dataset = CocoDetectionTV(root = f'{DATA_SAVE_ROOT}/COCO/val2017',
                              annFile = f'{DATA_SAVE_ROOT}/COCO/annotations/instances_val2017.json')
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_LOAD_WORKERS, collate_fn=collate_fn)
# Reverse transform for showing the image
val_reverse_transform = transforms.Compose([
    transforms.Normalize(mean=[-mean/std for mean, std in zip(IMAGENET_MEAN, IMAGENET_STD)],
                         std=[1/std for std in IMAGENET_STD])
])

###### 2. Define Model ######

###### 3. Define Criterion & Optimizer ######

###### 4. Training ######
# Import DETR package
sys.path.append(DETR_ROOT)
from torch_extend.detection.detr_utils import train_detection
# Train by the function
start = time.time()  # For elapsed time
result_dir = f'{RESULTS_SAVE_ROOT}/detr/{datetime.now().strftime("%Y%m%d%H%M%S")}_{DATASET_NAME}_{os.path.splitext(os.path.basename(PRETRAINED_WEIGHT))[0]}'
os.makedirs(result_dir, exist_ok=True)
train_data_path = f'{DATA_SAVE_ROOT}/{DATASET_NAME}'
train_detection(coco_path=train_data_path, device=device, 
                batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, num_workers=NUM_LOAD_WORKERS,
                output_dir=result_dir)
print(f'Training complete, elapsed_time={time.time() - start}')
# Save the weights
os.makedirs(f'{PARAMS_SAVE_ROOT}/detr', exist_ok=True)
model_weight_name = f'{os.path.basename(result_dir).split("_")[1]}_{os.path.basename(result_dir).split("_")[2]}.pth'
shutil.copy(f'{result_dir}/checkpoint.pth', f'{PARAMS_SAVE_ROOT}/detr/{model_weight_name}')

###### 5. Model evaluation and visualization ######

###### 6. Save the model ######


###### Inference in the first mini-batch ######
# Load a model with the trained weight
model = torch.hub.load("facebookresearch/detr", "detr_resnet50", pretrained=False)
print(model)
# Send the model to GPU
model.to(device)
# Turn off training mode so the model so it won't try to calculate loss
model.eval()
# Load the weights from the training result
best_weight = torch.load(f'{PARAMS_SAVE_ROOT}/detr/{model_weight_name}')
model.load_state_dict(best_weight['model'])

# Load a minibatch data
val_iter = iter(val_loader)
imgs, targets = next(val_iter)  # Load the first batch
imgs_transformed = [val_transform(img) for img in imgs]
imgs_gpu = [img.to(device) for img in imgs_transformed]
# Inference
if SAME_IMG_SIZE: # If the image sizes are the same, inference can be conducted with the batch data
    results = model(imgs_gpu)
    img_sizes = imgs_transformed.size()
else: # if the image sizes are different, inference should be conducted with one sample
    results = [model(img.unsqueeze(0)) for img in imgs_gpu]
    img_sizes = [img.size()[1:3] for img in imgs_transformed]
# Convert the Results to Torchvision object detection prediction format
predictions = convert_detr_hub_result(
    results, img_sizes=img_sizes,
    same_img_size=SAME_IMG_SIZE, prob_threshold=PROB_THRESHOLD
)
# Convert the Target bounding box positions in accordance with the resize
targets_resize = [resize_target(target, to_tensor(img), resized_img) for target, img, resized_img in zip(targets, imgs, imgs_transformed)]
# Class names dict with background
idx_to_class_bg = {k: v for k, v in idx_to_class.items()}
idx_to_class_bg[-1] = 'background'
# Show predicted images
imgs_display = [val_reverse_transform(img) for img in imgs_transformed]  # Reverse normalization
show_predicted_detection_minibatch(imgs_display, predictions, targets_resize, idx_to_class_bg, max_displayed_images=NUM_DISPLAYED_IMAGES)

#%%
###### Calculate mAP ######
targets_list = []
predictions_list = []
start = time.time()  # For elapsed time
for i, (imgs, targets) in enumerate(val_loader):
    imgs_transformed = [val_transform(img) for img in imgs]
    imgs_gpu = [img.to(device) for img in imgs_transformed]
    # Inference
    with torch.no_grad():  # Avoid memory overflow
        if SAME_IMG_SIZE: # If the image sizes are the same, inference can be conducted with the batch data
            results = model(imgs_gpu)
            img_sizes = imgs_transformed.size()
        else: # if the image sizes are different, inference should be conducted with one sample
            results = [model(img.unsqueeze(0)) for img in imgs_gpu]
            img_sizes = [img.size()[1:3] for img in imgs_transformed]
    # Convert the Results to Torchvision object detection prediction format
    predictions = convert_detr_hub_result(
        results, img_sizes=img_sizes,
        same_img_size=SAME_IMG_SIZE, prob_threshold=PROB_THRESHOLD
    )
    # Convert the Target bounding box positions in accordance with the resize
    targets_resize = [resize_target(target, to_tensor(img), resized_img) for target, img, resized_img in zip(targets, imgs, imgs_transformed)]
    # Store the result
    targets_list.extend(targets_resize)
    predictions_list.extend(predictions)
    if i%100 == 0:  # Show progress every 100 images
        print(f'Prediction for mAP: {i}/{len(val_loader)} batches, elapsed_time: {time.time() - start}')
aps = average_precisions(predictions_list, targets_list, idx_to_class_bg, iou_threshold=0.5, conf_threshold=0.2)
# Show mAP
show_average_precisions(aps)

# %%

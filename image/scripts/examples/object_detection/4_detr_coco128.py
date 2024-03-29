# %% COCO Detection + DETR (ResNet50) + No training
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import to_tensor
import matplotlib.pyplot as plt
from IPython import get_ipython
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from torch_extend.detection.display import show_bounding_boxes, show_predicted_detection_minibatch
from torch_extend.detection.target_converter import resize_target
from torch_extend.detection.dataset import CocoDetectionTV
from torch_extend.detection.result_converter import convert_detr_hub_result

SEED = 42
BATCH_SIZE = 2  # Batch size
NUM_EPOCHS = 3  # number of epochs
NUM_DISPLAYED_IMAGES = 10  # number of displayed images
NUM_LOAD_WORKERS = 0  # Number of workers for DataLoader (Multiple workers need much memory, so if the error "RuntimeError: DataLoader worker (pid ) is killed by signal" occurs, you should set it 0)
DEVICE = 'cuda'  # 'cpu' or 'cuda'
DATA_SAVE_ROOT = '/scripts/examples/object_detection/datasets/COCO/'  # Directory for Saved dataset
PARAMS_SAVE_ROOT = '/scripts/examples/object_detection/params'  # Directory for Saved parameters
FREEZE_PRETRAINED = True  # If True, Freeze pretrained parameters (Transfer learning)
SAME_IMG_SIZE = False  # Whether the image sizes are the same or not
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
# Load train dataset from image folder (https://medium.com/howtoai/pytorch-torchvision-coco-dataset-b7f5e8cad82)
# train_dataset = CocoDetection(root = f'{DATA_SAVE_ROOT}/train2017',
#                               annFile = f'{DATA_SAVE_ROOT}/annotations/instances_train2017.json',
#                               transform=transform, target_transform=target_transform)
# Define display loader
display_transform = transforms.Compose([
    transforms.ToTensor()  # Convert from range [0, 255] to a torch.FloatTensor in the range [0.0, 1.0]
])
def collate_fn(batch):
    return tuple(zip(*batch))
display_dataset = CocoDetectionTV(root = f'{DATA_SAVE_ROOT}/val2017',
                                  annFile = f'{DATA_SAVE_ROOT}/annotations/instances_val2017.json',
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
val_dataset = CocoDetectionTV(root = f'{DATA_SAVE_ROOT}/val2017',
                              annFile = f'{DATA_SAVE_ROOT}/annotations/instances_val2017.json')
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_LOAD_WORKERS, collate_fn=collate_fn)
# Reverse transform for showing the image
val_reverse_transform = transforms.Compose([
    transforms.Normalize(mean=[-mean/std for mean, std in zip(IMAGENET_MEAN, IMAGENET_STD)],
                         std=[1/std for std in IMAGENET_STD])
])

###### 2. Define Model ######

###### 3. Define Criterion & Optimizer ######

###### 4. Training ######

###### 5. Model evaluation and visualization ######

###### 6. Save the model ######


###### Inference in the first mini-batch ######
# Load a model with the trained weight
model = torch.hub.load("facebookresearch/detr", "detr_resnet50", pretrained=True)
print(model)
# Send the model to GPU
model.to(device)
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
get_ipython().magic('matplotlib inline')  # Matplotlib inline should be enabled to show plots after commiting YOLO inference
# Convert the Results to Torchvision object detection prediction format
predictions = convert_detr_hub_result(
    results, img_sizes=img_sizes,
    same_img_size=SAME_IMG_SIZE, prob_threshold=PROB_THRESHOLD
)
# Convert the Target bounding box positions in accordance with the resize
targets_resize = [resize_target(target, to_tensor(img), resized_img) for target, img, resized_img in zip(targets, imgs, imgs_transformed)]
# Show predicted images
imgs_display = [val_reverse_transform(img) for img in imgs_transformed]  # Reverse normalization
show_predicted_detection_minibatch(imgs_display, predictions, targets_resize, idx_to_class, max_displayed_images=NUM_DISPLAYED_IMAGES)

# %%

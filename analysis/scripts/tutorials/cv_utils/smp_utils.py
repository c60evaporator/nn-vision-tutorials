from torch.utils.data import DataLoader, Dataset
import cv2
import numpy as np
import os

VOC_CLASS_PALETTE = [
    [0, 0, 0],
    [128, 0, 0],
    [0, 128, 0],
    [128, 128, 0],
    [0, 0, 128],
    [128, 0, 128],
    [0, 128, 128],
    [128, 128, 128],
    [64, 0, 0],
    [192, 0, 0],
    [64, 128, 0],
    [192, 128, 0],
    [64, 0, 128],
    [192, 0, 128],
    [64, 128, 128],
    [192, 128, 128],
    [0, 64, 0],
    [128, 64, 0],
    [0, 192, 0],
    [128, 192, 0],
    [0, 64, 128],
]


class VOCSegmentationSmp(Dataset):
    """
    Dataset for segmentation_models_pytorch from Pascal VOC Segmentation format

    https://github.com/qubvel/segmentation_models.pytorch/blob/master/examples/cars%20segmentation%20(camvid).ipynb

    Parameters
    ----------
    images_dir : str
        Path to images folder
    masks_dir : str
        Path to segmentation masks folder
    class_values: list
        Values of classes to extract from segmentation mask
    augmentation : albumentations.Compose
        Data transfromation pipeline (e.g. flip, scale, etc.)
    preprocessing : albumentations.Compose
        Data preprocessing (e.g. noralization, shape manipulation, etc.)
    """
    def __init__(
            self, 
            images_dir, 
            masks_dir,
            classes=None,
            augmentation=None, 
            preprocessing=None,
            class_palette=None
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        
        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        if class_palette is None:
            self.class_palette = VOC_CLASS_PALETTE
        else:
            self.class_palette = class_palette
    
    def __getitem__(self, i):
        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)
        
        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask
        
    def __len__(self):
        return len(self.ids)
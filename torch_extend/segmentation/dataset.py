import torchvision.transforms as v2
import torchvision.transforms.functional as F2
from torchvision.datasets import CocoDetection, VisionDataset
import streamlit as st
import os
from typing import Any, Callable, List, Tuple, Optional
from PIL import Image
from pycocotools import mask as coco_mask
import numpy as np
from xml.etree.ElementTree import Element as ET_Element
try:
    from defusedxml.ElementTree import parse as ET_parse
except ImportError:
    from xml.etree.ElementTree import parse as ET_parse

from ..detection.dataset import VODBaseTV

class SegmentationOutput():
    def _get_images_targets(self):
        sample_size = self.__len__()
        images_targets = [self.__getitem__(idx) for idx in range(sample_size)]
        images = [image for image, target in images_targets]
        targets = [target for image, target in images_targets]
        return images, targets

def _convert_polygon_to_mask(segmentations, height, width):
    """
    Convert the polygons to masks
    reference https://github.com/facebookresearch/detr/blob/main/datasets/coco.py
    """
    masks = []
    np_masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = np.any(mask, axis=2).astype(np.uint8)
        masks.append(mask)
    if masks:
        masks = np.stack(masks, axis=0)
    else:
        masks = np.zeros((0, height, width), dtype=np.uint8)
    return masks

def _merge_masks(labels: np.ndarray, masks: np.ndarray):
    """
    Merge multiple masks int a mask
    """
    dst_mask = np.zeros((masks.shape[1], masks.shape[2]), dtype=np.uint8)
    for label, src_mask in zip(labels, masks):
        dst_mask = np.where((dst_mask == 0) & (src_mask > 0), label, dst_mask)
    return dst_mask

class CocoSegmentationTV(CocoDetection, SegmentationOutput):
    """
    Dataset from COCO format to Torchvision format with image path

    Parameters
    ----------
    root : str
        Path to images folder
    annFile : str
        Path to annotation text file folder
    transform : callable, optional
        A function/transform that takes in a PIL image and returns a transformed version. E.g, ``transforms.PILToTensor``
    target_transform : callable, optional
        A function/transform that takes in the target and transforms it.
    transforms : callable, optional
        A function/transform that takes input sample and its target as entry and returns a transformed version.
    random_crop : tuple, optional
        If not None, `torchvision.transforms.RandomCrop` with the specified size is applied to both the PIL image and the target. This transform is applied in advance of `transform` and `target_transform`. (https://stackoverflow.com/questions/58215056/how-to-use-torchvision-transforms-for-data-augmentation-of-segmentation-task-in)
    """
    def __init__(
        self,
        root: str,
        annFile: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
        random_crop: Optional[Tuple] = None,
    ) -> None:
        super().__init__(root, annFile, transform, target_transform, transforms)
        self.class_to_idx = {
            v['name']: v['id']
            for k, v in self.coco.cats.items()
        }
        self.random_crop = random_crop

    def _load_target(self, id: int, height: int, width: int) -> List[Any]:
        target_src = self.coco.loadAnns(self.coco.getAnnIds(id))
        # Get the labels
        labels = [obj['category_id'] for obj in target_src]
        labels = np.array(labels)
        # Get the segmentation polygons
        segmentations = [obj['segmentation'] for obj in target_src]
        # Convert the polygons to masks
        masks = _convert_polygon_to_mask(segmentations, height, width)
        # Merge the masks
        target = _merge_masks(labels, masks)
        return target
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id, image.size[1], image.size[0])
        target = Image.fromarray(target)

        if self.random_crop:
            i, j, h, w = v2.RandomCrop.get_params(image, self.random_crop)
            image = F2.crop(image, i, j, h, w)
            target = F2.crop(target, i, j, h, w)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target
    
class VOCSegmentationTV(VODBaseTV, SegmentationOutput):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Segmentation Dataset.

    Parameters
    ----------
    root : str
        Root directory of the VOC Dataset.
    image_set : str
        Select the image_set to use, ``"train"``, ``"trainval"`` or ``"val"``.
    transform : callable, optional
        A function/transform that  takes in an PIL image and returns a transformed version. E.g, ``transforms.PILToTensor``
    target_transform : callable, optional
        A function/transform that takes in the target and transforms it.
    transforms : callable, optional
        A function/transform that takes input sample and its target as entry and returns a transformed version.
    random_crop : tuple, optional
        If not None, `torchvision.transforms.RandomCrop` with the specified size is applied to both the PIL image and the target. This transform is applied in advance of `transform` and `target_transform`. (https://stackoverflow.com/questions/58215056/how-to-use-torchvision-transforms-for-data-augmentation-of-segmentation-task-in)
    """
    _SPLITS_DIR = "Segmentation"
    _TARGET_DIR = "SegmentationClass"
    _TARGET_FILE_EXT = ".png"

    def __init__(
        self,
        root: str,
        image_set: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
        random_crop: Optional[Tuple] = None,
    ):
        super().__init__(root, image_set, transform, target_transform, transforms)

        # Additional
        self.random_crop = random_crop

    @property
    def masks(self) -> List[str]:
        return self.targets

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        image = Image.open(self.images[index]).convert("RGB")
        target = Image.open(self.masks[index])

        if self.random_crop:
            i, j, h, w = v2.RandomCrop.get_params(image, self.random_crop)
            image = F2.crop(image, i, j, h, w)
            target = F2.crop(target, i, j, h, w)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target
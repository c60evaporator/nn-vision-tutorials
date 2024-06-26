import torch
from torchvision.datasets import VisionDataset, CocoDetection
from typing import Any, Callable, List, Dict, Optional, Tuple
from PIL import Image
import collections
import os
from xml.etree.ElementTree import Element as ET_Element
try:
    from defusedxml.ElementTree import parse as ET_parse
except ImportError:
    from xml.etree.ElementTree import parse as ET_parse

from .target_converter import convert_bbox_centerxywh_to_xyxy, convert_bbox_xywh_to_xyxy, target_transform_from_torchvision

class DetectionOutput():
    def _get_images_targets(self):
        sample_size = self.__len__()
        images_targets = [self.__getitem__(idx) for idx in range(sample_size)]
        images = [image for image, target in images_targets]
        targets = [target for image, target in images_targets]
        return images, targets

    def output_coco_annotation(self, info_dict=None, licenses_list=None):
        if self.idx_to_class is None:
            raise AttributeError('The "idx_to_class" attribute should not be None if the output format is COCO')
        # Get target and images as the TorchVision format
        images, targets = self._get_images_targets()
        # Convert the target to COCO format
        targets = [
            target_transform_from_torchvision(target, out_format='coco')
            for target in targets
        ]
        # Create "images" field in the annotation file
        ann_images = [
            {
                'license': 4,
                'file_name': os.path.basename(image_fp),
                'coco_url': '',
                'height': image.height,
                'width': image.width,
                'date_captured': '',
                'flickr_url': '',
                'id': i_img,
            }
            for i_img, (image, image_fp) in enumerate(zip(images, self.images))
        ]
        # Create "annotations" field in the annotation file
        ann_annotations = []
        obj_id_cnt = 0
        for i_img, target in enumerate(targets):
            for obj in target:
                obj_ann = {
                    'segmentation': [],
                    'area': 0,
                    'iscrowd': 0,
                    'image_id': i_img,
                    'bbox': obj['bbox'],
                    'category_id': obj['category_id'],
                    'id': obj_id_cnt
                }
                obj_id_cnt += 1
                ann_annotations.append(obj_ann)
        # Create "categories" field in the annotation file
        ann_categories = [
            {
                'supercategory': label_name,
                'id': idx,
                'name': label_name
            }
            for idx, label_name in self.idx_to_class.items()
        ]
        # Output the annotation json
        ann_dict = {
            'info': info_dict if info_dict is not None else {},
            'licenses': licenses_list if licenses_list is not None else {},
            'images': ann_images,
            'annotations': ann_annotations,
            'categories': ann_categories
        }
        return ann_dict

class CocoDetectionTV(CocoDetection, DetectionOutput):
    """
    Dataset from COCO format to Torchvision format with image path

    Parameters
    ----------
    root : str
        Path to images folder
    annFile : str
        Path to annotation text file folder
    transform : callable, optional
        A function/transform that  takes in an PIL image and returns a transformed version. E.g, ``transforms.PILToTensor``
    target_transform : callable, optional
        A function/transform that takes in the target and transforms it.
    transforms : callable, optional
        A function/transform that takes input sample and its target as entry and returns a transformed version.
    """
    def __init__(
        self,
        root: str,
        annFile: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, annFile, transform, target_transform, transforms)
        self.idx_to_class = {
            v['id']: v['name']
            for k, v in self.coco.cats.items()
        }

    def _load_target(self, id: int) -> List[Any]:
        target_src = self.coco.loadAnns(self.coco.getAnnIds(id))
        # Get the labels
        labels = [obj['category_id'] for obj in target_src]
        labels = torch.tensor(labels)
        # Get the bounding boxes
        boxes = [[int(k) for k in convert_bbox_xywh_to_xyxy(*obj['bbox'])]
             for obj in target_src]
        boxes = torch.tensor(boxes) if len(boxes) > 0 else torch.zeros(size=(0, 4))
        # Get the image path
        image_path = self.coco.loadImgs(id)[0]["file_name"]
        target = {'boxes': boxes, 'labels': labels, 'image_path': os.path.join(self.root, image_path)}
        return target
    
class YoloDetectionTV(VisionDataset, DetectionOutput):
    """
    Dataset from YOLO format to Torchvision format with image path

    Parameters
    ----------
    root : str
        Path to images folder
    ann_dir : str
        Path to annotation text file folder
    idx_to_class : Dict[int, str]
        A dict which indicates the conversion from the label indices to the label names
    transform : callable, optional
        A function/transform that  takes in an PIL image and returns a transformed version. E.g, ``transforms.PILToTensor``
    target_transform : callable, optional
        A function/transform that takes in the target and transforms it.
    transforms : callable, optional
        A function/transform that takes input sample and its target as entry and returns a transformed version.
    """
    def __init__(
        self, 
        root: str, 
        ann_dir: str,
        idx_to_class: Dict[int, str] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None
    ):
        super().__init__(root, transforms, transform, target_transform)
        self.idx_to_class = idx_to_class
        self.ids = os.listdir(root)
        self.images = [os.path.join(root, image_id) for image_id in self.ids]
        self.targets = [os.path.join(ann_dir, image_id.replace('png', 'txt').replace('jpg', 'txt')) for image_id in self.ids]
        assert len(self.images) == len(self.targets)
    
    def _load_image(self, index: int) -> Image.Image:
        return Image.open(self.images[index]).convert("RGB")

    def _load_target(self, index: int, w: int, h: int) -> List[Any]:
        ann_path = self.targets[index]
        # If annotation text file doesn't exist
        if not os.path.exists(ann_path):
            boxes = torch.zeros(size=(0, 4))
            labels = torch.tensor([])
        else:  # If annotation text file exists
            # Read text file 
            with open(ann_path) as f:
                lines = f.readlines()
            # Get the labels
            labels = [int(line.split(' ')[0]) for line in lines]
            labels = torch.tensor(labels)
            # Get the bounding boxes
            rect_list = [line.split(' ')[1:] for line in lines]
            rect_list = [[float(cell.replace('\n','')) for cell in rect] for rect in rect_list]
            boxes = [convert_bbox_centerxywh_to_xyxy(*rect) for rect in rect_list]  # Convert [x_c, y_c, w, h] to [xmin, ymin, xmax, ymax]
            boxes = torch.tensor(boxes) * torch.tensor([w, h, w, h], dtype=float)  # Convert normalized coordinates to raw coordinates
        target = {'boxes': boxes, 'labels': labels, 'image_path': self.images[index]}
        return target

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        image = self._load_image(index)
        w, h = image.size
        target = self._load_target(index, w, h)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        return len(self.ids)

class VODBaseTV(VisionDataset, DetectionOutput):
    _SPLITS_DIR: str
    _TARGET_DIR: str
    _TARGET_FILE_EXT: str

    def __init__(
        self,
        voc_root: str,
        image_set: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None
    ):
        super().__init__(voc_root, transforms, transform, target_transform)
        self.image_set = image_set
        if not os.path.isdir(voc_root):
            raise RuntimeError("Dataset not found or corrupted")
        
        splits_dir = os.path.join(voc_root, "ImageSets", self._SPLITS_DIR)
        split_f = os.path.join(splits_dir, image_set.rstrip("\n") + ".txt")
        with open(os.path.join(split_f)) as f:
            file_names = [x.strip() for x in f.readlines()]
        
        image_dir = os.path.join(voc_root, "JPEGImages")
        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]

        target_dir = os.path.join(voc_root, self._TARGET_DIR)
        self.targets = [os.path.join(target_dir, x + self._TARGET_FILE_EXT) for x in file_names]

        assert len(self.images) == len(self.targets)

    def __len__(self) -> int:
        return len(self.images)

class VOCDetectionTV(VODBaseTV, DetectionOutput):
    """
    Dataset from Pascal VOC format to Torchvision format with image path

    Parameters
    ----------
    root : str
        Root directory of the VOC Dataset.
    idx_to_class : Dict[int, str]
        A dict which indicates the conversion from the label indices to the label names
    image_set : str
        Select the image_set to use, ``"train"``, ``"trainval"`` or ``"val"``.
    transform : callable, optional
        A function/transform that  takes in an PIL image and returns a transformed version. E.g, ``transforms.PILToTensor``
    target_transform : callable, optional
        A function/transform that takes in the target and transforms it.
    transforms : callable, optional
        A function/transform that takes input sample and its target as entry and returns a transformed version.
    """
    _SPLITS_DIR = "Main"
    _TARGET_DIR = "Annotations"
    _TARGET_FILE_EXT = ".xml"

    def __init__(
        self, 
        root: str,
        idx_to_class : Dict[int, str],
        image_set: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None
    ):
        super().__init__(root, image_set, transform, target_transform, transforms)
        self.idx_to_class = idx_to_class
        self.class_to_idx = {v: k for k, v in idx_to_class.items()}
        self.ids = os.listdir(root)
    
    def _load_image(self, index: int) -> Image.Image:
        return Image.open(self.images[index]).convert("RGB")

    def _load_target(self, index: int, w: int, h: int) -> List[Any]:
        ann_path = self.targets[index]
        # Read XML file 
        target = self.parse_voc_xml(ET_parse(self.annotations[index]).getroot())
        objects = target['annotation']['object']
        # Get the labels
        labels = [self.class_to_idx[obj['name']] for obj in objects]
        labels = torch.tensor(labels)
        # Get the bounding boxes
        box_keys = ['xmin', 'ymin', 'xmax', 'ymax']
        boxes = [[int(obj['bndbox'][k]) for k in box_keys] for obj in objects]
        boxes = torch.tensor(boxes)
        target = {'boxes': boxes, 'labels': labels, 'image_path': self.images[index]}
        return target

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        image = self._load_image(index)
        w, h = image.size
        target = self._load_target(index, w, h)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target
    
    @staticmethod
    def parse_voc_xml(node: ET_Element) -> Dict[str, Any]:
        """
        From torchvision.datasets.VOCDetection.parse_voc_xml
        """
        voc_dict: Dict[str, Any] = {}
        children = list(node)
        if children:
            def_dic: Dict[str, Any] = collections.defaultdict(list)
            for dc in map(VOCDetectionTV.parse_voc_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            if node.tag == "annotation":
                def_dic["object"] = [def_dic["object"]]
            voc_dict = {node.tag: {ind: v[0] if len(v) == 1 else v for ind, v in def_dic.items()}}
        if node.text:
            text = node.text.strip()
            if not children:
                voc_dict[node.tag] = text
        return voc_dict

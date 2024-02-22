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

from cv_utils.detection_utils import convert_bbox_centerxywh_to_xyxy, convert_bbox_xywh_to_xyxy


class CocoDetectionTV(CocoDetection):
    """
    Dataset from YOLO format to Torchvision format with image path

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
    def _load_target(self, id: int) -> List[Any]:
        target_src = self.coco.loadAnns(self.coco.getAnnIds(id))
        # Get the labels
        labels = [obj['category_id'] for obj in target_src]
        labels = torch.tensor(labels)
        # Get the bounding boxes
        boxes = [[int(k) for k in convert_bbox_xywh_to_xyxy(*obj['bbox'])]
             for obj in target_src]
        boxes = torch.tensor(boxes)
        # Get the image path
        image_path = self.coco.loadImgs(id)[0]["file_name"]
        target = {'boxes': boxes, 'labels': labels, 'image_path': os.path.join(self.root, image_path)}
        return target
    
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

class YoloDetectionTV(VisionDataset):
    """
    Dataset from YOLO format to Torchvision format with image path

    Parameters
    ----------
    root : str
        Path to images folder
    ann_dir : str
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
        ann_dir: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None
    ):
        super().__init__(root, transforms, transform, target_transform)
        self.ids = os.listdir(root)
        self.images_fps = [os.path.join(root, image_id) for image_id in self.ids]
        self.ann_fps = [os.path.join(ann_dir, image_id.replace('png', 'txt').replace('jpg', 'txt')) for image_id in self.ids]
        assert len(self.images_fps) == len(self.ann_fps)
    
    def _load_image(self, index: int) -> Image.Image:
        return Image.open(self.images_fps[index]).convert("RGB")

    def _load_target(self, index: int, w: int, h: int) -> List[Any]:
        ann_path = self.ann_fps[index]
        # If annotation text file doesn't exist
        if not os.path.exists(ann_path):
            boxes = torch.zeros(size=(0, 4))
            labels = torch.tensor([])
        else:  # If annotation text file exists
            # Read text file 
            with open(ann_path) as f:
                lines = f.readlines()
            # Get the labels
            classes = [int(line.split(' ')[0]) for line in lines]
            labels = torch.tensor(classes)
            # Get the bounding boxes
            rect_list = [line.split(' ')[1:] for line in lines]
            rect_list = [[float(cell.replace('\n','')) for cell in rect] for rect in rect_list]
            boxes = [convert_bbox_centerxywh_to_xyxy(*rect) for rect in rect_list]  # Convert [x_c, y_c, w, h] to [xmin, ymin, xmax, ymax]
            boxes = torch.tensor(boxes) * torch.tensor([w, h, w, h], dtype=float)  # Convert normalized coordinates to raw coordinates
        target = {'boxes': boxes, 'labels': labels, 'image_path': self.images_fps[index]}
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
    
class VOCDetectionTV(VisionDataset):
    """
    Dataset from Pascal VOC format to Torchvision format with image path

    Parameters
    ----------
    root : str
        Path to images folder
    ann_dir : str
        Path to annotation XML file folder
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
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None
    ):
        super().__init__(root, transforms, transform, target_transform)
        self.ids = os.listdir(root)
        self.images_fps = [os.path.join(root, image_id) for image_id in self.ids]
        self.ann_fps = [os.path.join(ann_dir, image_id.replace('png', 'xml').replace('jpg', 'xml')) for image_id in self.ids]
        assert len(self.images_fps) == len(self.ann_fps)
    
    def _load_image(self, index: int) -> Image.Image:
        return Image.open(self.images_fps[index]).convert("RGB")

    def _load_target(self, index: int, w: int, h: int) -> List[Any]:
        ann_path = self.ann_fps[index]
        # Read XML file 
        target = self.parse_voc_xml(ET_parse(self.annotations[index]).getroot())
        # Get the labels
        classes = [int(line.split(' ')[0]) for line in lines]
        labels = torch.tensor(classes)
        # Get the bounding boxes
        rect_list = [line.split(' ')[1:] for line in lines]
        rect_list = [[float(cell.replace('\n','')) for cell in rect] for rect in rect_list]
        boxes = [convert_bbox_centerxywh_to_xyxy(*rect) for rect in rect_list]  # Convert [x_c, y_c, w, h] to [xmin, ymin, xmax, ymax]
        boxes = torch.tensor(boxes) * torch.tensor([w, h, w, h], dtype=float)  # Convert normalized coordinates to raw coordinates
        target = {'boxes': boxes, 'labels': labels, 'image_path': self.images_fps[index]}
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
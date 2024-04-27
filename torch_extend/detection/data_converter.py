import yaml
import os
import shutil
import json

from .dataset import YoloDetectionTV, DetectionOutput

def _output_images_from_dataset(dataset, out_dir, copy_metadata=False):
    os.makedirs(out_dir, exist_ok=True)
    for image_fp in dataset.images:
        if copy_metadata:
            shutil.copy2(image_fp, out_dir)
        else:
            shutil.copy(image_fp, out_dir)

def _output_dataset_as_voc(train_dataset: DetectionOutput, val_dataset: DetectionOutput, test_dataset: DetectionOutput, 
                           output_dir, out_train_dir, out_val_dir, out_test_dir):
    """Output the dataset as Pascal VOC format"""
    pass

def _save_coco_annotation(ann_dict, out_dir, ann_filename):
    if ann_dict is not None:
        os.makedirs(out_dir, exist_ok=True)
        with open(f'{out_dir}/{ann_filename}', 'w') as f:
            json.dump(ann_dict, f)

def _output_dataset_as_coco(train_dataset: DetectionOutput, val_dataset: DetectionOutput, test_dataset: DetectionOutput,
                            output_dir, out_train_dir, out_val_dir, out_test_dir,
                            train_ann_name, val_ann_name, test_ann_name):
    """Output the dataset as COCO format"""
    # Convert annotation data to coco format
    train_ann = train_dataset.output_coco_annotation()
    val_ann = val_dataset.output_coco_annotation()
    test_ann = test_dataset.output_coco_annotation() if test_dataset is not None else None
    # Save images
    train_dir_name = out_train_dir if out_train_dir is not None else 'train'
    val_dir_name = out_val_dir if out_val_dir is not None else 'val'
    test_dir_name = out_test_dir if out_test_dir is not None else 'test'
    _output_images_from_dataset(train_dataset, f'{output_dir}/{train_dir_name}')
    if val_dataset is not None:
        _output_images_from_dataset(val_dataset, f'{output_dir}/{val_dir_name}')
    if test_dataset is not None:
        _output_images_from_dataset(test_dataset, f'{output_dir}/{test_dir_name}')
    # Save annotation files
    train_ann_filename = train_ann_name if train_ann_name is not None else 'train.json'
    val_ann_filename = val_ann_name if val_ann_name is not None else 'val.json'
    test_ann_filename = test_ann_name if test_ann_name is not None else 'test.json'
    _save_coco_annotation(train_ann, f'{output_dir}/annotations/', train_ann_filename)
    _save_coco_annotation(val_ann, f'{output_dir}/annotations/', val_ann_filename)
    _save_coco_annotation(test_ann, f'{output_dir}/annotations/', test_ann_filename)

def _output_dataset_as_yolo(train_dataset: DetectionOutput, val_dataset: DetectionOutput, test_dataset: DetectionOutput, 
                            output_dir, out_train_dir, out_val_dir, out_test_dir):
    """Output the dataset as YOLO format"""
    pass

def _convert_from_yolo(yolo_yaml, yolo_root_dir, output_dir, 
                       out_train_dir, out_val_dir, out_test_dir,
                       train_ann_name, val_ann_name, test_ann_name,
                       output_format):
    """
    Convert dataset from YOLO format to another format.

    Parameters
    ----------
    yolo_yaml : str
        A path of the YOLO dataset YAML file

    yolo_root_dir : str
        A path of the YOLO dataset root directory
    
    output_dir : str
        A path of where the converted dataset will be output

    out_train_dir : str
        A direcotry name where the converted train images will be saved. If None, A name "train" is used

    out_val_dir : str
        A direcotry name where the converted validation images will be saved. If None, A name "val" is used

    out_test_dir : str
        A direcotry name where the converted validation images will be saved. If None, A name "test" is used
    
    train_ann_name : str
        A name of saved annotation JSON file of the train data. If None, A name "train.json" is used

    val_ann_name : str
        A name of saved annotation JSON file of the validation data. If None, A name "val.json" is used

    test_ann_name : str
        A name of saved annotation JSON file of the test data. If None, A name "test.json" is used
    
    output_format : {'pascal_voc', 'coco'}
        An output format
    """
    with open(yolo_yaml, 'r') as file:
        yolo_conf = yaml.safe_load(file)
    # Get image folders
    train_img_dir = yolo_conf['train']
    val_img_dir = yolo_conf['val']
    test_img_dir = yolo_conf.get('test', None)
    # Validate image folders whether the first folder name is "images"
    if train_img_dir[:6] != 'images':
        raise RuntimeError('The first folder name of "train" in YAML file should be "images"')
    if val_img_dir[:6] != 'images':
        raise RuntimeError('The first folder name of "val" in YAML file should be "images"')
    if test_img_dir is not None and test_img_dir[:6] != 'images':
        raise RuntimeError('The first folder name of "test" in YAML file should be "images"')
    # Assume annotation folders
    train_ann_dir = 'labels' + train_img_dir[6:]
    val_ann_dir = 'labels' + val_img_dir[6:]
    test_ann_dir = 'labels' + test_img_dir[6:] if test_img_dir is not None else None
    # Get label names
    idx_to_class = yolo_conf['names']
    class_to_idx = {v: k for k, v in idx_to_class.items()}
    nc = len(idx_to_class)
    # Load datasets
    train_dataset = YoloDetectionTV(
        root = f'{yolo_root_dir}/{train_img_dir}',
        ann_dir = f'{yolo_root_dir}/{train_ann_dir}',
        class_to_idx = class_to_idx
    )
    val_dataset = YoloDetectionTV(
        root = f'{yolo_root_dir}/{val_img_dir}',
        ann_dir = f'{yolo_root_dir}/{val_ann_dir}',
        class_to_idx = class_to_idx
    )
    if test_img_dir is not None:
        test_dataset = YoloDetectionTV(
            root = f'{yolo_root_dir}/{test_img_dir}',
            ann_dir = f'{yolo_root_dir}/{test_ann_dir}',
            class_to_idx = class_to_idx
        )
    else:
        test_dataset = None
    # Output the converted dataset based on the specified format
    if output_format == 'pascal_voc':
        _output_dataset_as_voc(train_dataset, val_dataset, test_dataset, output_dir)
    elif output_format == 'coco':
        _output_dataset_as_coco(train_dataset, val_dataset, test_dataset, output_dir,
                                out_train_dir, out_val_dir, out_test_dir,
                                train_ann_name, val_ann_name, test_ann_name)
    else:
        raise ValueError('The "output_format" argument should be "pascal_voc" or "coco"')

def convert_yolo2coco(yolo_yaml, yolo_root_dir, output_dir, 
                      out_train_dir=None, out_val_dir=None, out_test_dir=None,
                      train_ann_name=None, val_ann_name=None, test_ann_name=None):
    """
    Convert YOLO format dataset to COCO format

    Parameters
    ----------
    yolo_yaml : str
        A path of the YOLO dataset YAML file

    yolo_root_dir : str
        A path of the YOLO dataset root directory
    
    output_dir : str
        A path of the output root directory where the converted dataset will be saved

    out_train_dir : str
        A direcotry name where the converted train images will be saved. If None, "train" is used

    out_val_dir : str
        A direcotry name where the converted validation images will be saved. If None, "val" is used

    out_test_dir : str
        A direcotry name where the converted validation images will be saved. If None, "test" is used
    
    train_ann_name : str
        A name of saved annotation JSON file of the train data. If None, "train.json" is used

    val_ann_name : str
        A name of saved annotation JSON file of the validation data. If None, "val.json" is used

    test_ann_name : str
        A name of saved annotation JSON file of the test data. If None, "test.json" is used
    """
    _convert_from_yolo(yolo_yaml, yolo_root_dir, output_dir,
                       out_train_dir, out_val_dir, out_test_dir,
                       train_ann_name, val_ann_name, test_ann_name,
                       output_format='coco')

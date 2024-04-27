from enum import Enum
import os
import streamlit as st
import glob
import yaml
from PIL import Image
import torchvision.transforms.functional as F

class VocSegFormat(Enum):
    ImageSets = 'ImageSets'
    JPEGImages = 'JPEGImages'
    Annotations = 'SegmentationClass'

def list_datasets(dir_path):
    "List up the directories in the desiginated directory"
    if isinstance(dir_path, str):
        dir_paths = [dir_paths]
    else:
        dir_paths = [l for l in dir_path]
    dataset_paths = []
    for dir in dir_paths:
        dataset_paths.extend([
            f'{dir}/{f}'
            for f in os.listdir(dir) if os.path.isdir(os.path.join(dir, f))
        ])
    return {os.path.basename(path): path for path in dataset_paths}

def get_seg_voc_dataset(dataset_root, class_names_yaml):
    # Check whether the necessary folders exist
    dataset_subfolders = [f for f in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, f))]
    if VocSegFormat.Annotations.value not in dataset_subfolders:
        st.error(f'There is no "{VocSegFormat.Annotations.value}" folder in the dataset')
        return None
    elif VocSegFormat.ImageSets.value not in dataset_subfolders:
        st.error(f'There is no "{VocSegFormat.ImageSets.value}" folder in the dataset')
        return None
    elif VocSegFormat.JPEGImages.value not in dataset_subfolders:
        st.error(f'There is no "{VocSegFormat.JPEGImages.value}" folder in the dataset')
        return None
    imgsets_dir = f'{dataset_root}/{VocSegFormat.ImageSets.value}'
    imgsets_subfolders = [f for f in os.listdir(imgsets_dir) if os.path.isdir(os.path.join(imgsets_dir, f))]
    if 'Segmentation' not in imgsets_subfolders:
        st.error('There is no "Segmentation" folder in the dataset')
        return None
    # Annotation files
    ann_dir = f'{dataset_root}/{VocSegFormat.Annotations.value}'
    #ann_paths = glob.glob(f'{ann_dir}/*.xml')
    # ImageSets files
    imgsets_segdir = f'{imgsets_dir}/Segmentation'
    imgsets_paths = glob.glob(f'{imgsets_segdir}/*.txt')
    # Image files
    image_dir = f'{dataset_root}/{VocSegFormat.JPEGImages.value}'

    # Select the image set
    image_sets = [os.path.splitext(os.path.basename(imgset_file))[0]
                    for imgset_file in imgsets_paths]
    image_set = st.selectbox('Select image set', image_sets)
    imgsets_file = f'{imgsets_segdir}/{image_set}.txt'

    # Load the ImageSets file
    with open(imgsets_file) as f:
        lines = f.read().splitlines()
    # Load index to class dict
    if not os.path.isfile(f'/{dataset_root}/{class_names_yaml}'):
        st.error('Please put the idx_to_class.yml file on the root folder of the dataset')
    with open(f'/{dataset_root}/{class_names_yaml}', 'r') as yml:
        idx_to_class = yaml.safe_load(yml)['names']
    # Accumulate dataset info
    dataset_info = {
        'idx_to_class': idx_to_class,
        'image_set': image_set,
        'images': [
        {'name': line, 'image': f'{image_dir}/{line}.jpg', 'annotation': f'{ann_dir}/{line}.png'}
        for line in lines
    ]}
    return dataset_info

def convert_seg_voc_mask(annotation_img, idx_to_class):
    # Convert
    mask = F.pil_to_tensor(annotation_img)
    # Replace the border
    mask[mask == 255] = len(idx_to_class)
    return mask

def read_image_metadata(img_path):
    img = Image.open(img_path)
    return list(img.size) + [str(img.mode)]
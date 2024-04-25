from enum import Enum
import os
import streamlit as st
import glob

class VocFormat(Enum):
    Annotations = 'Annotations'
    ImageSets = 'ImageSets'
    JPEGImages = 'JPEGImages'

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

def get_seg_voc_dataset(dataset_root):
    # Check whether the necessary folders exist
    dataset_subfolders = [f for f in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, f))]
    if VocFormat.Annotations.name not in dataset_subfolders:
        st.error('There is no "Annotations" folder in the dataset')
        return
    elif VocFormat.ImageSets.name not in dataset_subfolders:
        st.error('There is no "ImageSets" folder in the dataset')
        return
    elif VocFormat.JPEGImages.name not in dataset_subfolders:
        st.error('There is no "JPEGImages" folder in the dataset')
        return
    imgsets_dir = f'{dataset_root}/{VocFormat.ImageSets.name}'
    imgsets_subfolders = [f for f in os.listdir(imgsets_dir) if os.path.isdir(os.path.join(imgsets_dir, f))]
    if 'Segmentation' not in imgsets_subfolders:
        st.error('There is no "Segmentation" folder in the dataset')
        return
    # Annotation files
    ann_dir = f'{dataset_root}/{VocFormat.Annotations.name}'
    #ann_paths = glob.glob(f'{ann_dir}/*.xml')
    # ImageSets files
    imgsets_segdir = f'{imgsets_dir}/Segmentation'
    imgsets_paths = glob.glob(f'{imgsets_segdir}/*.txt')
    # Image files
    image_dir = f'{dataset_root}/{VocFormat.JPEGImages.name}'
    #image_paths = glob.glob(f'{ann_dir}/*.jpg')

    # Select the dataset type
    dataset_types = [os.path.splitext(os.path.basename(imgset_file))[0]
                    for imgset_file in imgsets_paths]
    dataset_type = st.selectbox('Select dataset type', dataset_types)
    imgsets_file = f'{imgsets_segdir}/{dataset_type}.txt'

    # Load the ImageSets file
    with open(imgsets_file) as f:
        lines = f.readlines()
    # Create 
    dataset_info = [
        {'image': f'{image_dir}/{line}.jpg', 'annotation': f'{image_dir}/{line}.xml'}
        for line in lines
    ]
    return dataset_info
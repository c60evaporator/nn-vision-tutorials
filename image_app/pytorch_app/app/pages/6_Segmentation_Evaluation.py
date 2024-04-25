import os
from enum import Enum
import yaml
import streamlit as st
from st_aggrid import AgGrid, GridUpdateMode, GridOptionsBuilder

import torch_extend.segmentation.metrics as metrics
from utils.dataset import list_datasets, get_seg_voc_dataset
from utils.models import list_seg_models, list_seg_weights, load_seg_model

class SegmentationDataFormat(Enum):
    VOC = 'VOC'
    COCO = 'COCO'

class SegmentationModelFormat(Enum):
    TorchVision = 'TorchVision'
    SMP = 'SMP'

# Read the config file
with open('./config/config.yml', 'r') as yml:
    config = yaml.safe_load(yml)

col_dataset, col_model = st.columns(2)

###### Select the Dataset ######
with col_dataset:
    st.markdown('**Select the dataset**')
    # Select the dataset
    dataset_dict = list_datasets(config['directories']['dataset_dir']['semantic_segmentation'])
    selected_dataset = st.selectbox('Select dataset', dataset_dict.keys())
    selected_dataset_path = dataset_dict[selected_dataset]
    # Select the format
    selected_format = st.selectbox('Select format', SegmentationDataFormat.__members__.keys())
    # If the format i
    if selected_format == SegmentationDataFormat.VOC.name:
        # Select subfolder (Option)
        subfolders = [f for f in os.listdir(selected_dataset_path) if os.path.isdir(os.path.join(selected_dataset_path, f))]
        subfolder = st.selectbox('Select subfolder', [None] + subfolders)
        if subfolder is None:
            dataset_root = selected_dataset_path
        else:
            dataset_root = f'{selected_dataset_path}/{subfolder}'
        dataset_info = get_seg_voc_dataset(dataset_root)
        st.write(dataset_info)

###### Load the Models ######
with col_model:
    st.markdown('**Select the model**')
    # Select the model format
    selected_modelformat = st.selectbox('Select model format', SegmentationModelFormat.__members__.keys())
    # Select the model name
    selected_modelname = st.selectbox('Select model name', list_seg_models(selected_modelformat))
    # List up the weights
    weight_dir = config['directories']['weight_dir']['semantic_segmentation']
    weight_paths = list_seg_weights(weight_dir, selected_modelname)
    # Select the weight
    weight_names = [os.path.basename(weight_path) for weight_path in weight_paths]
    selected_weight_name = st.selectbox('Select model weight', [None] + weight_names)
    if selected_weight_name is not None:
        selected_weight = f'{weight_dir}/{selected_weight_name}'
        # Load the Model
        model, num_classes = load_seg_model(selected_modelname, selected_weight)
        st.write(f'Loaded model={model.__class__.__name__}, num_classes={num_classes}')

###### Display data in the Dataset ######


###### Display evaluation history ######

###### Batch evaluation ######
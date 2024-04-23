import os
from enum import Enum
import yaml
import streamlit as st
from st_aggrid import AgGrid, GridUpdateMode, GridOptionsBuilder

import torch_extend.segmentation.metrics as metrics
from utils.dataset import list_datasets, load_segmentation_voc
from utils.models import list_seg_models, list_seg_weights, load_seg_model

class SegmentationDataFormat(Enum):
    VOC = 'VOC'
    COCO = 'COCO'

class SegmentationModelFormat(Enum):
    TorchVision = 'TorchVision'
    SMP = 'SMP'

# Read the config file
with open('./config/data_config.yml', 'r') as yml:
    data_config = yaml.safe_load(yml)

col_dataset, col_model = st.columns(2)

###### Select the Dataset ######
with col_dataset:
    st.markdown('**Select the dataset**')
    # Select the dataset
    dataset_dict = list_datasets(data_config['directories']['dataset_dir']['semantic_segmentation'])
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
        load_segmentation_voc(dataset_root)

###### Select the Model ######
with col_model:
    st.markdown('**Select the model**')
    # Select the model format
    selected_modelformat = st.selectbox('Select model format', SegmentationModelFormat.__members__.keys())
    # Select the model name
    selected_modelname = st.selectbox('Select model name', list_seg_models(selected_modelformat))
    # List up the weights
    weight_dir = data_config['directories']['weight_dir']['semantic_segmentation']
    weight_paths = list_seg_weights(weight_dir, selected_modelname)
    # Select the weight
    weight_names = [os.path.basename(weight_path) for weight_path in weight_paths]
    selected_weight_name = st.selectbox('Select model weight', weight_names)
    selected_weight = f'{weight_dir}/{selected_weight_name}'
    # Load the Model
    model = load_seg_model(selected_modelname, selected_weight)
    st.write(model)

###### Display data in the Dataset ######


######  ######
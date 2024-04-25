import os
from enum import Enum
import yaml
import streamlit as st
from st_aggrid import AgGrid, GridUpdateMode, GridOptionsBuilder
import pandas as pd

import torch_extend.segmentation.metrics as metrics
from utils.dataset import list_datasets, get_seg_voc_dataset
from utils.models import list_seg_models, list_seg_weights, load_seg_model, load_seg_default_models

class SegmentationDataFormat(Enum):
    VOC = 'VOC'
    COCO = 'COCO'

class SegmentationModelFormat(Enum):
    TorchVision = 'TorchVision'
    SMP = 'SMP'

# Read the config file
with open('./config/config.yml', 'r') as yml:
    config = yaml.safe_load(yml)

col_dataset, col_model = st.columns([1, 2])

###### Load default Models ######
load_seg_default_models(weight_dir=config['directories']['weight_dir']['semantic_segmentation'],
                        weight_names=config['models']['default_load_weights']['semantic_segmentation'])

###### Select the Dataset ######
dataset_info = None
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

###### Load the Models ######
with col_model:
    st.markdown('**Models**')
    model_info = [{k: v for k, v in model.items() if k != 'model'} for model in st.session_state['seg_models']]
    st.dataframe(pd.DataFrame(model_info))

###### Get the evaluation history ######
eval_history = []

###### Display data in the Dataset ######
if dataset_info is not None:
    df_dataset = pd.DataFrame(dataset_info)
    n_images = len(df_dataset)
    # Merge the evaluation history
    if len(eval_history) > 0:
        # Sort by the metrics
        sort_model = st.selectbox('Sort by', SegmentationDataFormat.__members__.keys())
        df_dataset = df_dataset.sort_values(sort_model, ascending=True)
    # Divide the data based on a radio button
    col_radio, col_radioinfo = st.columns([4, 1])
    with col_radio:
        records_per_table = config['display']['records_per_table']
        n_tables = int(round(n_images / records_per_table, 0))
        table_idx = st.radio('Page', tuple(range(1, n_tables + 1)), horizontal=True)
        i_start = records_per_table * (table_idx - 1)
        i_end = records_per_table * table_idx
        df_dataset_display = df_dataset[i_start:i_end]
    with col_radioinfo:
        st.text('\n\n')
        st.text(f'{i_start + 1}-{i_end}\nin {n_images} images')
    # Display interactive table by st_aggrid
    gd = GridOptionsBuilder.from_dataframe(df_dataset_display)
    gd.configure_columns(df_dataset_display, width=120)
    gd.configure_selection(selection_mode='single', use_checkbox=False)
    gd.configure_grid_options(alwaysShowHorizontalScroll=True, rowHeight=25)  # Scroll bar
    grid_table = AgGrid(df_dataset_display, gridOptions=gd.build(), height=300,
                        update_mode=GridUpdateMode.SELECTION_CHANGED)
    selected_fieldgroups = [row for row in grid_table['selected_rows']]
    ###### Display the selected image ######
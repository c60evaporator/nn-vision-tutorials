import os
from enum import Enum
import yaml
import streamlit as st
from st_aggrid import AgGrid, GridUpdateMode, GridOptionsBuilder
from streamlit_image_comparison import image_comparison
import numpy as np
import pandas as pd
from PIL import Image

from torch_extend.segmentation.display import array1d_to_pil_image

from utils.segmentation.dataset import list_datasets, get_seg_voc_dataset, convert_seg_voc_mask
from utils.segmentation.models import load_seg_default_models, inference_and_score
from utils.segmentation.display import show_seg_legend, create_overlayed_annotation, get_segmentation_palette

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
    # If the format is VOC detection
    if selected_format == SegmentationDataFormat.VOC.name:
        # Select subfolder (Option)
        subfolders = [f for f in os.listdir(selected_dataset_path) if os.path.isdir(os.path.join(selected_dataset_path, f))]
        subfolder = st.selectbox('Select subfolder', [None] + subfolders)
        if subfolder is None:
            dataset_root = selected_dataset_path
        else:
            dataset_root = f'{selected_dataset_path}/{subfolder}'
        # Get the class names
        class_names_yml = f'/{dataset_root}/{config["class_names"]["voc"]}'
        if not os.path.isfile(class_names_yml):
            st.error('Please put the idx_to_class.yml file on the root folder of the dataset')
        else:
            with open(f'/{dataset_root}/{config["class_names"]["voc"]}', 'r') as yml:
                idx_to_class = yaml.safe_load(yml)['names']
            bg_idx = 0
            border_idx=len(idx_to_class)
            # Load the dataset information
            dataset_info = get_seg_voc_dataset(dataset_root)

###### Load the Models ######
with col_model:
    st.markdown('**Models**')
    model_info = [{k: v for k, v in model.items() if k != 'model'} for model in st.session_state['seg_models']]
    st.dataframe(pd.DataFrame(model_info))

###### Get the evaluation history ######
eval_history = []

###### Display image list in the Dataset ######
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
    col_image_list, col_image_list = st.columns([4, 1])
    gd = GridOptionsBuilder.from_dataframe(df_dataset_display)
    gd.configure_columns(df_dataset_display, width=120)
    gd.configure_selection(selection_mode='single', use_checkbox=False)
    gd.configure_grid_options(alwaysShowHorizontalScroll=True, rowHeight=25)  # Scroll bar
    grid_table = AgGrid(df_dataset_display, gridOptions=gd.build(), height=300,
                        update_mode=GridUpdateMode.SELECTION_CHANGED)
    selected_images = [row for row in grid_table['selected_rows']]

    ###### Display the selected image ######
    if len(selected_images) > 0:
        selected_image = selected_images[0]
        # Read the raw image
        raw_image = Image.open(selected_image['image'])
        # Read the annotation mask image
        if selected_format == SegmentationDataFormat.VOC.name:
            ann_image = Image.open(selected_image['annotation'])
        # Convert the mask to class labels
        mask = convert_seg_voc_mask(ann_image, idx_to_class)
        # Selectbox for comparison or overlay
        col_process_type, col_display_type = st.columns(2)
        with col_process_type:
            img_process_type = st.radio('What to see?', ['Annotation', 'Model comparison'], horizontal=True)

        ###### Display the annotation mask ######
        if img_process_type == 'Annotation':
            with col_display_type:
                img_display_type = st.radio('Image display type', ['comparison', 'overlay'], horizontal=True)
            # Write the legend
            palette = get_segmentation_palette(selected_format, bg_idx, border_idx)
            mask_image = array1d_to_pil_image(mask[0], palette)
            show_seg_legend(mask, idx_to_class, palette, border_idx, dark_indices=[border_idx])
            # Show the images with image_comparison
            if img_display_type == 'comparison':
                image_comparison = image_comparison(
                    img1=raw_image, label1='raw',
                    img2=mask_image, label2='annotation'
                )
            # Show the overlayed image
            elif img_display_type == 'overlay':
                segmentation_overlay_alpha = st.slider('Alpha', 0.0, 1.0, value=0.5, step=0.1)
                overlayed_image = create_overlayed_annotation(raw_image, ann_image, segmentation_overlay_alpha)
                st.image(overlayed_image)

        ###### Display the model comparison ######
        elif img_process_type == 'Model comparison':
            with col_display_type:
                img_display_type = st.radio('Image display type', ['comparison', 'double'], horizontal=True)
            # Select the models
            col_left_model, col_right_model = st.columns(2)
            selected_model_names = [None, None]
            with col_left_model:
                left_model_names = ['Annotation'] + [model['model_name'] for model in st.session_state['seg_models']]
                selected_model_names[0] = st.selectbox('Model in the left', left_model_names)
            with col_right_model:
                right_model_names = [model_name for model_name in left_model_names if model_name != selected_model_names[0]]
                selected_model_names[1] = st.selectbox('Model in the right', right_model_names)
            # Load the selected models
            selected_models = []
            for model_name in selected_model_names:
                selected_models.append(None if model_name == 'Annotation' else
                                   [model for model in st.session_state['seg_models'] 
                                   if model['model_name'] == model_name][0])
            # Select Alpha of the overlayed images
            col_alpha_legend = st.columns(3)
            with col_alpha_legend[1]:
                segmentation_overlay_alpha = st.slider('Alpha', 0.0, 1.0, value=0.5, step=0.1)
            # Generate a palette for the mask images
            palette = get_segmentation_palette(selected_format, bg_idx, border_idx, bg_bright=True, border_dark=True)
            # Conduct inference
            displayed_images = []
            displayed_text = []
            for i, model in enumerate(selected_models):
                # Annotation
                if model is None:
                    shown_mask, iou_dict = mask[0], None
                # Inference result of the model
                else:
                    predicted_labels, iou_dict = inference_and_score(raw_image, model, idx_to_class, mask, border_idx)
                    shown_mask = predicted_labels[0]
                mask_image = array1d_to_pil_image(shown_mask, palette)
                displayed_image = create_overlayed_annotation(raw_image, mask_image, segmentation_overlay_alpha)
                displayed_images.append(displayed_image)
                with col_alpha_legend[0 if i == 0 else 2]:
                    # Show the legend with the score
                    show_seg_legend(shown_mask, idx_to_class, palette, border_idx, dark_indices=[bg_idx], iou_dict=iou_dict)
            # Display the prediction result with image_comparison
            if img_display_type == 'comparison':
                image_comparison = image_comparison(
                    img1=displayed_images[0], label1=selected_model_names[0],
                    img2=displayed_images[1], label2=selected_model_names[1]
                )
            # Display each predicted image
            elif img_display_type == 'double':
                col_left_image, col_right_image = st.columns(2)
                with col_left_image:
                    st.image(displayed_images[0])
                with col_right_image:
                    st.image(displayed_images[1])
import os
from enum import Enum
import yaml
import json
import pandas as pd
import streamlit as st
from st_aggrid import AgGrid, GridUpdateMode, GridOptionsBuilder

import torch_extend.segmentation.metrics as metrics
from utils.segmentation.dataset import list_datasets, get_seg_voc_dataset
from utils.segmentation.models import list_seg_models, list_seg_weights, load_seg_model
from utils.segmentation.evaluation import get_evaluation_dataset, segmentation_eval_torchvison
from sql.database import get_db
import sql.crud as crud
import preprocessing.seg_preprocessing as preproessing

DEVICE = 'cuda'

class SegmentationDataFormat(Enum):
    VOC = 'VOC'
    COCO = 'COCO'

class SegmentationModelFormat(Enum):
    TorchVision = 'TorchVision'
    SMP = 'SMP'

###### Initialization ######
st.title('Segmentation Evaluation')
# Read the config file
with open('./config/config.yml', 'r') as yml:
    config = yaml.safe_load(yml)
# Load the database operator
get_db(config['directories']['database_dir'])

tab_history, tab_new = st.tabs(["History", "New evaluation"])

with tab_history:
    col_dataset, col_hostory = st.columns([1, 3])
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
                dataset_name = selected_dataset
            else:
                dataset_root = f'{selected_dataset_path}/{subfolder}'
                dataset_name = f'{selected_dataset}/{subfolder}'
            dataset_info = get_seg_voc_dataset(dataset_root, config['class_names']['voc'])

    if dataset_info is not None:
        # Get the dataset parameters
        image_set = dataset_info['image_set']
        idx_to_class = dataset_info['idx_to_class']
        # Get the border index
        border_idx = len(idx_to_class) if selected_format == SegmentationDataFormat.VOC.name else None
        ###### Show the evaluation history ######
        with col_hostory:
            st.markdown('**Evaluation history**')
            ###### Display evaluation history ######
            df_evals = crud.get_evaluations(st.session_state['db'], convert_df=True)
            if len(df_evals) > 0:
                # Dataset filter
                df_eval_filtered = df_evals[df_evals['dataset'] == dataset_name]
                # Image set filter
                df_eval_filtered = df_eval_filtered[df_eval_filtered['image_set'] == image_set]
                # Select the rows
                df_eval_display = df_eval_filtered[['evaluation_id','model_name', 'mean_iou']]
                # Display interactive table by st_aggrid
                gd = GridOptionsBuilder.from_dataframe(df_eval_display)
                gd.configure_columns(df_eval_display, width=120)
                gd.configure_selection(selection_mode='single', use_checkbox=False)
                gd.configure_grid_options(alwaysShowHorizontalScroll=True)  # Scroll bar
                grid_table = AgGrid(df_eval_display, gridOptions=gd.build(),
                                    update_mode=GridUpdateMode.SELECTION_CHANGED)
                selected_evals = [row for row in grid_table['selected_rows']]

        ###### Show the label metrics of selected evaluation ######
        if len(df_evals) > 0 and len(selected_evals) > 0:
            st.subheader('Metrics of each label')
            selected_eval_id = selected_evals[0]['evaluation_id']
            selected_eval = df_eval_filtered[df_eval_filtered['evaluation_id'] == selected_eval_id].iloc[0]
            # Read the IoUs
            label_names = json.loads(selected_eval['label_names'])
            ious = json.loads(selected_eval['ious'])
            tps = json.loads(selected_eval['tps'])
            fps = json.loads(selected_eval['fps'])
            fns = json.loads(selected_eval['fns'])
            unions = json.loads(selected_eval['unions'])
            label_metrics = [
                {
                    'label_id': k,
                    'label_name': v,
                    'tp': tps[i],
                    'fp': fps[i],
                    'fn': fns[i],
                    'union': unions[i],
                    'iou': ious[i]
                }
                for i, (union, (k, v)) in enumerate(zip(unions, idx_to_class.items()))
            ]
            df_label_metrics = pd.DataFrame(label_metrics)
            st.dataframe(df_label_metrics)

with tab_new:
    if dataset_info is None:
        st.warning('Please select the dataset')
    else:
        col_model, col_loader = st.columns(2)
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

        if selected_weight_name is not None:
            ###### Set the loader parameters ######
            with col_loader:
                st.markdown('**Loader parameters**')
                # Get the PyTorch dataset
                dataset, transform, target_transform, albumentations_transform = get_evaluation_dataset(
                    selected_format, selected_modelname, idx_to_class, dataset_root, image_set)
                # Evaluation parameters
                col_eval_params = st.columns(3)
                with col_eval_params[0]:
                    batch_size = st.number_input('Batch size', min_value=1, max_value=64, value=1, step=1)
                with col_eval_params[1]:
                    num_workers = st.number_input('Num workers', min_value=1, max_value=8, value=4, step=1)
                # Create dataloader
                dataloader = preproessing.get_dataloader(selected_modelname, dataset, batch_size, num_workers)
            ###### Start the evaluation ######
            if st.button('Start evaluation'):
                # Create the evaluation record
                evaluation = {
                    'dataset': dataset_name,
                    'image_set': image_set,
                    'num_images': dataset.__len__(),
                    'model_name': selected_modelname,
                    'weight': selected_weight_name,
                    'dataloader': dataloader.__class__.__name__,
                    'transform': str(transform),
                    'target_transform': str(target_transform),
                    'albumentations_transform': str(albumentations_transform),
                    'batch_size': batch_size,
                    'mean_iou': None,
                    'label_names': json.dumps(idx_to_class),
                    'tps': None,
                    'fps': None,
                    'fns': None,
                    'ious': None,
                    'elapsed_time': None,
                }
                db_evaluation = crud.create_evaluation(st.session_state['db'], evaluation)
                # Start evaluation
                segmentation_eval_torchvison(st.session_state['db'], db_evaluation.evaluation_id, db_evaluation.created_at,
                                            dataloader, model, DEVICE, idx_to_class, border_idx)
                if st.button('Refresh'):
                    st.rerun()
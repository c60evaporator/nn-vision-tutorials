import streamlit as st
from PIL import Image

VOC_COLORMAP = [
    [0, 0, 0],
    [128, 0, 0],
    [0, 128, 0],
    [128, 128, 0],
    [0, 0, 128],
    [128, 0, 128],
    [0, 128, 128],
    [128, 128, 128],
    [64, 0, 0],
    [192, 0, 0],
    [64, 128, 0],
    [192, 128, 0],
    [64, 0, 128],
    [192, 0, 128],
    [64, 128, 128],
    [192, 128, 128],
    [0, 64, 0],
    [128, 64, 0],
    [0, 192, 0],
    [128, 192, 0],
    [0, 64, 128],
]

def get_segmentation_palette(dataset_format, bg_idx=None, border_idx=None, bg_bright=False, border_dark=False):
    if dataset_format == 'VOC':
        palette = [rgb for rgb in VOC_COLORMAP]
    # imputation based on the bg_idx
    if bg_idx is not None and bg_idx >= len(palette):
        for i in range(len(palette), bg_idx + 1):
            palette.append([i, i, i])
        palette[bg_idx] = [0, 0, 0]
    # imputation based on the border_idx
    if border_idx is not None and border_idx >= len(palette):
        for i in range(len(palette), border_idx + 1):
            palette.append([i, i, i])
        palette[border_idx] = [255, 255, 255]
    # Replace the background
    if bg_bright:
        palette[bg_idx] = [255, 255, 255]
    # Replace the border
    if border_dark:
        palette[border_idx] = [0, 0, 0]
    return palette

def show_seg_legend(mask, idx_to_class, palette, border_idx=None, dark_indices=[], iou_dict=None, score_decimal=2):
    """Show the legend of the segmentation using st.markdown"""
    # Add the border label to idx_to_class
    idx_to_class_bd = {k: v for k, v in idx_to_class.items()}
    if border_idx is not None:
        idx_to_class_bd[border_idx] = 'border'
    # Create the legend HTML text
    idx_unique = mask.ravel().unique().tolist()
    legend_html = ''
    for label_idx in idx_unique:
        # Add IoU
        if iou_dict is not None:
            iou_text = f'(IoU={round(iou_dict[label_idx]["iou"], score_decimal)})' if label_idx in iou_dict else ''
        else:
            iou_text = ''
        # Calculate hex RGB
        r, g, b = palette[label_idx]
        r_hex, g_hex, b_hex = format(r, '02x'), format(g, '02x'), format(b, '02x')
        if label_idx in dark_indices:  # Dark background color
            legend_html += f'<span style="background-color:#bbbbbb;color:#{r_hex}{g_hex}{b_hex};font-size:12px;">■{idx_to_class_bd[label_idx]}{iou_text} </span>'
        else:  # Bright background color
            legend_html += f'<span style="color:#{r_hex}{g_hex}{b_hex};font-size:12px;">■{idx_to_class_bd[label_idx]}{iou_text} </span>'
    # Show the legend
    st.markdown(legend_html, unsafe_allow_html=True)

def create_overlayed_annotation(raw_image, ann_image, alpha):
    overlay_mask = Image.new("L", raw_image.size, int(alpha*255))
    overlayed_image = raw_image.copy()
    overlayed_image.paste(ann_image.convert('RGB'), (0, 0), overlay_mask)
    return overlayed_image

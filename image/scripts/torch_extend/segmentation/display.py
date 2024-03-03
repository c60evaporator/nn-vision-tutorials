import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image

# def _create_segmentation_palette():
#     BLIGHTNESSES = [0, 64, 128, 192]
#     len_blt = len(BLIGHTNESSES)
#     pattern_list = []
#     for i in range(255):
#         r = BLIGHTNESSES[i % len_blt]
#         g = BLIGHTNESSES[(i // len_blt) % len_blt]
#         b = BLIGHTNESSES[(i // (len_blt ** 2)) % len_blt]
#         pattern_list.append([r, g, b])
#     pattern_list.append([255, 255, 255])
#     return np.array(pattern_list, dtype=np.uint8)

def _create_segmentation_palette():
    """
    # Color palette for segmentation masks
    """
    palette = sns.color_palette(n_colors=256).as_hex()
    palette = [list(int(ip[i:i+2],16) for i in (1, 3, 5)) for ip in palette]  # Convert hex to RGB
    return palette

def _array1d_to_pil_image(array: torch.Tensor, palette=None, bg_idx=None, border_idx=None):
    """
    Convert 1D class image to colored PIL image
    """
    # Auto palette generation
    if palette is None:
        palette = _create_segmentation_palette()
    # Replace the background
    if bg_idx is not None:
        palette[bg_idx] = [255, 255, 255]
    # Replace the border
    if border_idx is not None:
        palette[border_idx] = [0, 0, 0]
    # Convert the array from torch.tensor to np.ndarray
    array_numpy = array.detach().to('cpu').numpy().astype(np.uint8)
    # Convert the array
    pil_out = Image.fromarray(array_numpy, mode='P')
    pil_out.putpalette(np.array(palette, dtype=np.uint8))
    return pil_out

def show_segmentations(image, target, 
                       alpha=0.5, palette=None,
                       bg_idx=0, border_idx=None,
                       ax=None):
    """
    Show the image with the segmentation.

    Parameters
    ----------
    image : torch.Tensor (ch, x, y)
        Input image
    target : torch.Tensor (x, y)
        Target segmentation class with Torchvision segmentation format 
    alpha : float
        Transparency of the segmentation 
    palette : List ([[R1, G1, B1], [R2, G2, B2],..])
        Color palette for specifying the classes
    bg_idx : int
        Index of the background class
    ax : matplotlib Axes
        Axes object to draw the plot onto, otherwise uses the current Axes.
    """
    # If ax is None, use matplotlib.pyplot.gca()
    if ax is None:
        ax=plt.gca()
    # Display Image
    ax.imshow(image.permute(1, 2, 0))
    # Display Segmentations
    segmentation_img = _array1d_to_pil_image(target, palette, bg_idx, border_idx)
    ax.imshow(segmentation_img, alpha=alpha)
    # Add Ledgends
    

    #image_with_boxes = image_with_boxes.permute(1, 2, 0)  # Change axis order from (ch, x, y) to (x, y, ch)

import os
import h5py
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image

def read_h5_data(file_path):
    with h5py.File(file_path, 'r') as f:
        x = f['x'][:]
        y = f['y'][:]
        tissue_type = f['tissue_type'][:]
        max_pro = f['max_pro'][:]
        attrs = f.attrs
        slide_size = attrs['slide_size']
        real_patch_size = attrs['coords_patch_size']
        return x, y, tissue_type, max_pro, slide_size, real_patch_size

def create_white_image(size):
    return Image.new('RGB', tuple(size), 'white')

def TN_map_drawer(slide_name, h5_path):
    x, y, tissue_type, max_pro, slide_size, real_patch_size = read_h5_data(h5_path)

    output_folder_tn = '～/eval_result/TN_map/'
    output_folder_map = '～/eval_result/map/'
    os.makedirs(output_folder_tn, exist_ok=True)
    os.makedirs(output_folder_map, exist_ok=True)

    width, height = slide_size
    downsample = 32
    H = width // downsample
    W = height // downsample
    real_patch_size = real_patch_size // downsample
    image = create_white_image((W, H))

    result = np.ones([W, H]) * -1

    for i in range(len(x)):
        if tissue_type[i] == 1:
            result[y[i] // downsample:y[i] // downsample + real_patch_size,
            x[i] // downsample:x[i] // downsample + real_patch_size] = max_pro[i]
        elif tissue_type[i] == 0:
            result[y[i] // downsample:y[i] // downsample + real_patch_size,
            x[i] // downsample:x[i] // downsample + real_patch_size] = 1 - max_pro[i]

    dpi = 100
    plt.figure(figsize=(2 * slide_size[1] // downsample / dpi, 2 * slide_size[0] // downsample / dpi), dpi=dpi)
    s = sns.heatmap(result, cmap="RdYlBu_r", vmax=1.001, vmin=-0.001, mask=(result == -1),
                    xticklabels=False, yticklabels=False, cbar=False)
    plt.savefig(os.path.join(output_folder_map, slide_name + '.png'))
    plt.close()

    result = np.ones([W, H]) * -1

    for i in range(len(x)):
        if tissue_type[i] == 1:
            result[y[i] // downsample:y[i] // downsample + real_patch_size,
            x[i] // downsample:x[i] // downsample + real_patch_size] = tissue_type[i]
        elif tissue_type[i] == 0:
            result[y[i] // downsample:y[i] // downsample + real_patch_size,
            x[i] // downsample:x[i] // downsample + real_patch_size] = tissue_type[i]

    dpi = 100
    plt.figure(figsize=(slide_size[1] // downsample / dpi, slide_size[0] // downsample / dpi), dpi=dpi)
    s = sns.heatmap(result, cmap="RdYlBu_r", vmax=1.001, vmin=-0.001, mask=(result == -1),
                    xticklabels=False, yticklabels=False, cbar=False)
    plt.savefig(os.path.join(output_folder_tn, slide_name + '.png'))
    plt.close()
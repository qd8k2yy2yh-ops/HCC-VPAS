import json
import h5py
import os
import numpy as np
from openslide import OpenSlide, open_slide
import random

def h5_writer(JSON_file):
    save_path_h5 = None
    with open(JSON_file, 'r') as json_file:
        data = json.load(json_file)
    
    # Extract required fields from the loaded JSON data
    labels = data['labels']
    slide_names = data['slide_names']
    x_values = data['x']
    y_values = data['y']
    max_pro = data['max_probabilities']

    unique_slide_names = list(set(slide_names))
    for slide_name in unique_slide_names:
        indices_for_slide = [i for i, name in enumerate(slide_names) if name == slide_name]

        save_path = '～/eval_result/result_h5/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    
        save_path_h5 = os.path.join(save_path, slide_name + '.h5')
    
        slide_path = os.path.join('～/CD34/', slide_name + '.svs')
        slide = open_slide(slide_path)
        slide_size = slide.dimensions
        slide_size = np.array(slide_size).tolist()

        N = 10
        save_patch_size = 1024
        img_path = '～/eval_result/view_img/'
        if not os.path.exists(img_path):
            os.makedirs(img_path)
        tumor_indices = [i for i, label in enumerate(labels) if label == 1]
        if len(tumor_indices) > 10:
            tumor_indices = [i for i, label in enumerate(labels) if label == 1]
            random_tumor_indices = random.sample(tumor_indices, N)
            for i in random_tumor_indices:
                img = slide.read_region((x_values[i], y_values[i]), 0, (save_patch_size, save_patch_size)).convert('RGB')
                img.save(img_path + slide_name + '_10x_1024_' + str(x_values[i]) + '_' + str(y_values[i]) + "_1.png")

        slide.close()
    
        pixel_size = 0.276
        real_patch_size = 1024
        tissue_area = len(indices_for_slide) * real_patch_size * real_patch_size * pixel_size * pixel_size / 100000000
        tumor_area = labels.count(1) * real_patch_size * real_patch_size * pixel_size * pixel_size / 100000000

        asset_dict = {
            "x": [x_values[i] for i in indices_for_slide],
            'y': [y_values[i] for i in indices_for_slide],
            "tissue_type": [labels[i] for i in indices_for_slide],
            'max_pro': [max_pro[i] for i in indices_for_slide]
        }
    
        attrs = {
            'slide_name': slide_name,
            'slide_size': slide_size,
            'max_mag': 20,
            "coords_patch_size": real_patch_size,
            'tissue_area': tissue_area,
            'tumor_area': tumor_area
        }
    
        with h5py.File(save_path_h5, 'w') as f:
            for key, value in asset_dict.items():
                f.create_dataset(key, data=value)
            for key, value in attrs.items():
                f.attrs[key] = value
                
    return save_path_h5

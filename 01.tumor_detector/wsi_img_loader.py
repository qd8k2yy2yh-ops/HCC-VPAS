import os
import openslide
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset

class Get_h5_Dataset(Dataset):
    def __init__(self, slide_dir, patch_size,slide_level, slide_names, coords, labels, indices, transform=None):
        self.slide_dir = slide_dir
        self.patch_size = patch_size
        self.slide_level = slide_level
        self.slide_names = slide_names
        self.coords = coords
        self.labels = labels
        self.indices = indices
        self.transform = transform
    def __getitem__(self, ind):
        slide_name = self.slide_names[ind]
        coord = self.coords[ind]
        label = self.labels[ind]
        idx = self.indices[ind]
        wsi_path = os.path.join(self.slide_dir, slide_name+'.svs')
        wsi = openslide.open_slide(wsi_path)
        x, y = coord
        real_patch_size = (self.slide_level + 1) * self.patch_size
        img = wsi.read_region((x, y), 0, (real_patch_size, real_patch_size)).convert('RGB').resize((self.patch_size, self.patch_size))
        if self.transform:
            img = self.transform(img)
        if label == 55:
            return img, label, slide_name, x, y, idx
        else:
            return img, label, slide_name, x, y, idx
    def __len__(self):
        return len(self.coords)
    def get_label_list(self):
        return self.labels

import os
import h5py
import openslide
import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, slide_dir, coords_dir, patch_size, slide_level, stride, tissue_type=None, transform=None,
                 files=None):
        self.slide_dir = slide_dir
        self.coords_dir = coords_dir
        self.patch_size = patch_size
        self.slide_level = slide_level
        self.stride = stride
        self.tissue_type = tissue_type
        self.transform = transform
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        slide_name, idx = self.files[index]
        slide_file = os.path.join(self.slide_dir, slide_name + '.svs')
        coords_file = os.path.join(self.coords_dir, slide_name + '.h5')
        wsi = openslide.open_slide(slide_file)
        coords, labels = self.load_coords(coords_file)

        coord, label = coords[idx], labels[idx]
        # 当 label == 55 时，patch是无 组织标签 的状态
        if label == 55:
            x, y = coord
            img = wsi.read_region((x, y), self.slide_level, (self.patch_size, self.patch_size)).convert('RGB')
            if self.transform:
                img = self.transform(img)

            return img, slide_name, idx

        x, y = coord
        img = wsi.read_region((x, y), self.slide_level, (self.patch_size, self.patch_size)).convert('RGB')
        if self.transform:
            img = self.transform(img)

        if self.tissue_type is not None and label != self.tissue_type:
            return None

        if self.patch_size != 256 and self.slide_level != 0:
            real_patch_size = (self.slide_level + 1) * patch_size
            patch_num = real_patch_size // self.patch_size
            effective_patch = []
            for i in range(patch_num):
                for j in range(patch_num):
                    patch_coords = (x + i * self.patch_size, y + j * self.patch_size)
                    if self.is_valid_patch(patch_coords, labels, self.patch_size):
                        effective_patch.append(labels[idx])
            effective_num = len(effective_patch)

            if effective_num > (real_patch_size / 256 * real_patch_size / 256) / 2:
                tumor_num = effective_patch.count(1)
                if tumor_num > effective_num / 2:
                    new_label = 1
                else:
                    new_label = 0
                label = new_label

            # Remove redundant coordinates within real_patch_size
            if real_patch_size > self.patch_size:
                valid_coords = []
                for c in coords:
                    c_x, c_y = c
                    if not (x < c_x < x + real_patch_size and y < c_y < y + real_patch_size):
                        valid_coords.append(c)
                coords = valid_coords

        if self.stride < self.patch_size:
            aug_img = wsi.read_region((x + self.stride, y + self.stride), self.slide_level,
                                      (self.patch_size, self.patch_size)).convert('RGB')
            aug_label = label
            if self.is_valid_patch((x + self.stride, y), labels, self.patch_size) and \
                    self.is_valid_patch((x, y + self.stride), labels, self.patch_size) and \
                    self.is_valid_patch((x + self.stride, y + self.stride), labels, self.patch_size):
                aug_label = labels[idx]
            if self.transform:
                aug_img = self.transform(aug_img)
            return (img, aug_img), (label, aug_label)

        if self.stride > self.patch_size:
            if index % (self.stride // self.patch_size) == 0:
                return img, label

        if self.stride == self.patch_size:
            return img, label

    def is_valid_patch(self, patch_coords, labels, patch_size):
        x, y = patch_coords
        patch_label = []
        for i in range(patch_size):
            for j in range(patch_size):
                patch_label.append(labels[(x + i, y + j)])
        return all(x == patch_label[0] for x in patch_label)

    def load_coords(self, coords_file):
        with h5py.File(coords_file, 'r') as f:
            coords = f['coords'][:]
            if 'tissue_type' in f:
                # print(f"tissue_type --------------------------------------: {coords_file}")
                labels = f['tissue_type'][:]
            else:
                labels = [55] * len(coords)  # 如果缺少'tissue_type'，将labels设置为55

            if len(coords) == 0:
                print(f"Coords is empty in file: {coords_file}")

            # print("Coords length:", len(coords))
            # print("Labels length:", len(labels))

        return coords, labels

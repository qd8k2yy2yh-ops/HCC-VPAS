import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

class MyDataSet(Dataset):

    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item]).convert('RGB')
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        label = self.images_class[item]

        width, height = img.size
        ves_tumor_px = width * height
        if width != height:
            padding_left = (height - width) // 2 if height > width else 0
            padding_right = padding_left
            padding_top = (width - height) // 2 if width > height else 0
            padding_bottom = padding_top
            padding = (padding_left, padding_top, padding_right, padding_bottom)
            img = transforms.functional.pad(img, padding)

        if self.transform is not None:
            img = self.transform(img)

        image_path = self.images_path[item]
        return img, label, image_path, ves_tumor_px, width, height

    def get_label_list(self):
        return self.images_class

    @staticmethod
    def collate_fn(batch):
        images, labels, image_path, ves_tumor_px, width, height = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels, image_path, ves_tumor_px, width, height

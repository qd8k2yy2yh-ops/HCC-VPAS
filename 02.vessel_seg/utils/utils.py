import random
import numpy as np
import torch
from PIL import Image
from torch.hub import load_state_dict_from_url


def cvtColor(image):
    """Convert image to RGB if not already in RGB format."""
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    return image.convert('RGB')


def resize_image(image, size):
    """Resize image to the specified size, preserving aspect ratio."""
    iw, ih = image.size
    w, h = size
    scale = min(w / iw, h / ih)
    nw, nh = int(iw * scale), int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))

    return new_image, nw, nh


def get_lr(optimizer):
    """Get the learning rate from the optimizer."""
    for param_group in optimizer.param_groups:
        return param_group['lr']


def seed_everything(seed=11):
    """Set the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id, rank, seed):
    """Initialize the dataloader worker with a unique seed."""
    worker_seed = rank + seed
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def preprocess_input(image):
    """Preprocess input image by normalizing pixel values to [0, 1]."""
    return image / 255.0


def show_config(**kwargs):
    """Display the configuration parameters."""
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    for key, value in kwargs.items():
        print('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)


def download_weights(backbone, model_dir="./model_data"):
    """Download pre-trained model weights for specified backbone."""
    download_urls = {
        'vgg': 'https://download.pytorch.org/models/vgg16-397923af.pth',
        'resnet50': 'https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth'
    }
    url = download_urls.get(backbone)

    if not url:
        raise ValueError(f"Unsupported backbone: {backbone}")

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    load_state_dict_from_url(url, model_dir)

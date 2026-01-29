import os
import torch
import torch.nn as nn
from torchvision import transforms, models
import pickle
import json
from tqdm import tqdm


def run(model, dataloaders, phase, device):
    if phase == 'train':
        model.train()
    elif phase == 'valid':
        model.eval()
    else:
        raise Exception("Error phase")

    results = {'slide_names': [], 'x': [], 'y': [], 'idxes': [], 'labels': [], 'max_probabilities': []}

    for i, (inputs, labels, slide_names, x, y, idxes) in enumerate(dataloaders):
        inputs, labels = inputs.cuda(non_blocking=True), labels.cuda(non_blocking=True)

        with torch.set_grad_enabled(phase == 'train'):
            outputs = model(inputs)
            pre_pro, pre_labels = torch.max(outputs, dim=1)

        predicted_probabilities = torch.softmax(outputs, dim=1)
        max_probabilities, predicted_classes = torch.max(predicted_probabilities, dim=1)

        pre_labels_prob = max_probabilities.tolist()

        results['slide_names'].extend(slide_names)
        results['x'].extend(x.tolist())
        results['y'].extend(y.tolist())
        results['labels'].extend(pre_labels)
        results['max_probabilities'].extend(pre_labels_prob)

    return results


def main(args):
    if not torch.cuda.is_available():
        raise EnvironmentError("not find GPU device for training.")

    init_distributed_mode(args)
    rank = args.rank
    device = torch.device(args.device)
    batch_size = args.batch_size
    args.lr *= args.world_size
    weights_path = args.weights
    checkpoint_path = ""

    if rank == 0:
        print(args)
        print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
        if not os.path.exists("./weights"):
            os.makedirs("./weights")

    classes = ['N', 'T']
    num_classes = len(classes)
    assert args.num_classes == num_classes, f"dataset num_classes: {args.num_classes}, input {num_classes}"

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    model = models.resnet50()
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    model = model.to(device)

    state_dict = torch.load(weights_path, map_location=device)
    for key in state_dict.keys():
        if 'module.' in key:
            model = torch.nn.DataParallel(model)
            break
    model.load_state_dict(state_dict, True)
    model = model.to(device)

    if args.freeze_layers:
        for name, para in model.named_parameters():
            if "fc" not in name:
                para.requires_grad_(False)

    h5_data_path = '/h5_data_unlabeled/'
    slide_names = [f.split('.svs')[0] for f in os.listdir('/data15/junguang_usb/CD34/') if
                   f.endswith('.svs') and not (f.startswith('con') or f.startswith('sora'))]

    for slide_name in tqdm(slide_names):
        pickle_path = os.path.join('./eval_result/result_pickle/', slide_name)
        os.makedirs(pickle_path, exist_ok=True)

        pickle_filename = os.path.join(pickle_path,
                                       'TN_unlabel_CD34_10x_512_all_data_names_coords_labels_indices.pickle')

        if os.path.exists(pickle_filename):
            with open(pickle_filename, 'rb') as file:
                all_data = pickle.load(file)
        else:
            all_data = []
            current_coords_dir = os.path.join(h5_data_path, slide_name + '.h5')
            filtered_slide_names, filtered_coords, filtered_labels, filtered_indices = filter_coords(
                coords_dir=current_coords_dir,
                slide_level=1,
                patch_size=512,
                tissue_type=None,
                random_sample=None,
                stride=None
            )
            all_data.extend(zip(filtered_slide_names, filtered_coords, filtered_labels, filtered_indices))
            with open(pickle_filename, 'wb') as file:
                pickle.dump(all_data, file)

        if rank == 0:
            print(slide_name, '----len(all_data)------', len(all_data))

        val = {'slide_names': [data[0] for data in all_data],
               'coords': [data[1] for data in all_data],
               'labels': [data[2] for data in all_data],
               'indices': [data[3] for data in all_data]}

        slide_dir = '/slide_data_path/'

        try:
            val_dataset = Get_h5_Dataset(
                slide_dir=slide_dir,
                patch_size=512,
                slide_level=1,
                slide_names=val['slide_names'],
                coords=val['coords'],
                labels=val['labels'],
                indices=val['indices'],
                transform=data_transform['val']
            )
        except FileNotFoundError as e:
            print(f"FileNotFoundError: {e}. Skipping {slide_name}.")

        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
        nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])

        if rank == 0:
            print(f'Using {nw} dataloader workers every process')

        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=batch_size,
                                                 sampler=val_sampler,
                                                 pin_memory=True,
                                                 num_workers=nw)

        results = run(model, val_loader, 'valid', device)

        save_dir = os.path.join('./eval_result/result_json/', slide_name)
        os.makedirs(save_dir, exist_ok=True)
        json_path = os.path.join(save_dir, '02_result_slide_names_idxes_labels.json')

        with open(json_path, 'w') as json_file:
            json.dump(results, json_file)

        from PIL import Image
        from eval_result_write_in_h5 import h5_writer
        h5_path = h5_writer(json_path)

        from make_h5_heatmap import TN_map_drawer, create_white_image, read_h5_data
        TN_map_drawer(slide_name, h5_path)

    if rank == 0 and os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    cleanup()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--weights', type=str,
                        default='./bestloss.pt')
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--dist-url', default='env://')
    opt = parser.parse_args()

    main(opt)

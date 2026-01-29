import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, models
from sklearn.model_selection import KFold
import h5py
import pickle
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix
from matplotlib import pyplot as plt
import glob
from CD34_coords_dataloder import MyDataset
from ger_coords_labels_indices import filter_coords
from wsi_img_loader import Get_h5_Dataset

def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    fig = plt.figure(figsize=(12, 6))
    plt.title(title)
    for plot_index in range(2):
        if plot_index == 1:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        plt.subplot(1, 2, plot_index + 1)
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=15)
        plt.yticks(tick_marks, classes)

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j] if plot_index == 0 else "{:.2f}%".format(cm[i, j] * 100),
                     horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
    return fig


def print_conf_matrix(confusion_matrix, classes):
    first_line = [' '] + [c.ljust(20) for c in classes]
    print(''.join(first_line))
    all_acc = 0
    i = 0
    for row, _class in zip(confusion_matrix, classes):
        row_pretty = [_class] + [f'{(num * 100 / sum(row)):.2f}%({num}/{sum(row)})'.ljust(20) for num in row]
        all_acc += (row[i] / sum(row))
        i += 1
        print(''.join(row_pretty))
    return all_acc / len(classes)


def run(model, dataloaders, phase, device, criterion, optimizer, epoch):
    if phase == 'train':
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    running_corrects = 0
    all_labels = np.array([])
    all_predicts = np.array([])
    all_values = np.array([])

    for i, (inputs, labels, slide_names, x, y, idxes) in enumerate(dataloaders):
        if (i != 0) and (i % 200) == 0:
            fpr, tpr, threshold = roc_curve(all_labels, all_values, pos_label=1)
            print(
                f'Process<{phase}> [{i}/{len(dataloaders)}] Current acc: {np.mean(all_labels == all_predicts):.5f} AUC : {auc(fpr, tpr):.5f}')

        inputs, labels = inputs.cuda(non_blocking=True), labels.cuda(non_blocking=True)
        optimizer.zero_grad()

        with torch.set_grad_enabled(phase == 'train'):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            value = outputs[:, 1]
            loss = criterion(outputs, labels)

            if phase == 'train':
                loss.backward()
                optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        all_labels = np.concatenate((all_labels, labels.cpu().numpy()))
        all_predicts = np.concatenate((all_predicts, preds.cpu().numpy()))
        all_values = np.concatenate((all_values, value.detach().cpu().numpy()))

    Loss = running_loss / len(all_labels)
    Acc = (all_labels == all_predicts).sum() / len(all_predicts)
    TP = ((all_predicts == 1) & (all_labels == 1)).sum()
    TN = ((all_predicts == 0) & (all_labels == 0)).sum()
    FN = ((all_predicts == 0) & (all_labels == 1)).sum()
    FP = ((all_predicts == 1) & (all_labels == 0)).sum()

    p = TP / (TP + FP)
    r = TP / (TP + FN)
    F1 = 2 * r * p / (r + p)

    classes = ['N', 'T']
    conf_matrix = confusion_matrix(all_labels, all_predicts, labels=list(range(len(classes))))

    avg_acc = print_conf_matrix(conf_matrix, classes)

    Writer.add_scalar(f'{phase}/Loss', Loss, epoch)
    Writer.add_scalar(f'{phase}/Acc', Acc, epoch)

    fpr, tpr, threshold = roc_curve(all_labels, all_values, pos_label=1)
    print(f'AUC {auc(fpr, tpr)}')

    Writer.add_scalar(f'{phase}/AUC', auc(fpr, tpr), epoch)
    Writer.add_scalar(f'{phase}/Precision', p, epoch)
    Writer.add_scalar(f'{phase}/Recall', r, epoch)
    Writer.add_scalar(f'{phase}/F1', F1, epoch)
    Writer.add_figure(f'{phase}[{epoch}]', figure=plot_confusion_matrix(conf_matrix, classes=classes),
                      global_step=epoch)

    return Loss, p, r, F1, Acc, auc(fpr, tpr), conf_matrix


def make_weights_for_balanced_classes(label_list, nclasses):
    count = [0] * nclasses
    for item in label_list:
        count[item] += 1

    weight_per_class = [float(sum(count)) / float(c) for c in count]
    weight = [weight_per_class[val] for val in label_list]
    return weight, weight_per_class


def main(args):
    if not torch.cuda.is_available():
        raise EnvironmentError("not find GPU device for training.")

    init_distributed_mode(args)
    rank = args.rank
    device = torch.device(args.device)
    batch_size = args.batch_size
    args.lr *= args.world_size

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

    h5_data_path = '../00_slide_CD34_svs/h5_data_for_train/'
    tumor_count, normal_count = 0, 0
    for file in glob.glob(os.path.join(h5_data_path, '*.h5')):
        with h5py.File(file, 'r') as hdf5_file:
            labels = hdf5_file['tissue_type'][:]
            tumor_count += sum(labels == 1)
            normal_count += sum(labels == 0)

    total_count = tumor_count + normal_count
    if rank == 0:
        print(f"Tumor patches: {tumor_count}")
        print(f"Normal patches: {normal_count}")
        print(f"Total patches: {total_count}")

    slide_names = [f.split('.h5')[0] for f in os.listdir(h5_data_path) if f.endswith('.h5')]
    pickle_filename = './CD34_10x_512_all_data_names_coords_labels_indices.pickle'

    if os.path.exists(pickle_filename):
        with open(pickle_filename, 'rb') as file:
            all_data = pickle.load(file)
    else:
        all_data = []
        for slide_name in slide_names:
            current_coords_dir = os.path.join(h5_data_path, slide_name + '.h5')
            filtered_slide_names, filtered_coords, filtered_labels, filtered_indices = filter_coords(
                coords_dir=current_coords_dir,
                slide_level=1,
                patch_size=512,
                tissue_type=None,
                random_sample=2,
                stride=None
            )
            all_data.extend(zip(filtered_slide_names, filtered_coords, filtered_labels, filtered_indices))

        with open(pickle_filename, 'wb') as file:
            pickle.dump(all_data, file)

    if rank == 0:
        print(f'len(all_data)------ {len(all_data)}')

    model = models.resnet50(pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    model = model.to(device)

    if args.freeze_layers:
        for name, para in model.named_parameters():
            if "fc" not in name:
                para.requires_grad_(False)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0.)

    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True)

    for fold_idx, (train_index, val_index) in enumerate(kf.split(slide_names)):
        train_names = [slide_names[i] for i in train_index]
        val_names = [slide_names[i] for i in val_index]
        if rank == 0:
            print(f'\n##### Fold [{fold_idx}/{n_splits}]')
            print(f"Fold {fold_idx + 1}:")
            print(f' num train is {len(train_names)} and num val is {len(val_names)}\n')

        train_data = [data for data in all_data if data[0] in train_names]
        val_data = [data for data in all_data if data[0] in val_names]

        train = {'slide_names': [data[0] for data in train_data],
                 'coords': [data[1] for data in train_data],
                 'labels': [data[2] for data in train_data],
                 'indices': [data[3] for data in train_data]}

        val = {'slide_names': [data[0] for data in val_data],
               'coords': [data[1] for data in val_data],
               'labels': [data[2] for data in val_data],
               'indices': [data[3] for data in val_data]}

        slide_dir = '/slide_data_path/'

        train_dataset = Get_h5_Dataset(
            slide_dir=slide_dir,
            patch_size=512,
            slide_level=1,
            slide_names=train['slide_names'],
            coords=train['coords'],
            labels=train['labels'],
            indices=train['indices'],
            transform=data_transform['train']
        )

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

        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)

        train_batch_sampler = torch.utils.data.BatchSampler(
            train_sampler, batch_size, drop_last=True)
        nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])

        if rank == 0:
            print(f'Using {nw} dataloader workers every process')

        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_sampler=train_batch_sampler,
                                                   shuffle=False,
                                                   pin_memory=True,
                                                   num_workers=nw)
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=batch_size,
                                                 sampler=val_sampler,
                                                 shuffle=False,
                                                 pin_memory=True,
                                                 num_workers=nw)

        checkpoints = os.path.join("BS_" + str(batch_size) + "_fold_" + str(fold_idx))
        print(f'Summary write in {checkpoints}')
        Writer = SummaryWriter(log_dir=checkpoints)

        _, weight_per_class = make_weights_for_balanced_classes(train_dataset.get_label_list(), num_classes)
        weight_per_class = torch.Tensor(weight_per_class).to(device)
        criterion = nn.CrossEntropyLoss(weight=weight_per_class)

        num_epochs = args.epochs
        for epoch in range(1, num_epochs + 1):
            print(f'\n##### Epoch [{epoch}/{num_epochs}]')

            print('\n####### Train #######')
            train_loss, t_p, t_r, t_F1, train_acc, train_auc, train_cm = run(model, train_loader, 'train', device,
                                                                             criterion, optimizer, epoch)

            print('\n####### Valid #######')
            val_loss, v_p, v_r, v_F1, val_acc, val_auc, val_cm = run(model, val_loader, 'valid', device, criterion,
                                                                     optimizer, epoch)

            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]

            Writer.add_scalar('Learning Rate', current_lr, epoch)

            print(f'Epoch {epoch} with lr {current_lr:.15f}: t_loss: {train_loss:.4f} t_acc: {train_acc:.4f} v_loss:{val_loss:.4f} v_acc: {val_acc:.4f}')
            print(f'HCC: t_precision: {t_p:.4f}, t_recall: {t_r:.4f}, t_F1: {t_F1:.4f}')
            print(f'HCC: v_precision: {v_p:.4f}, v_recall: {v_r:.4f}, v_F1: {v_F1:.4f}')

            best_F1 = 0.0
            best_auc = 0.0
            best_loss = 1000000

            if v_F1 > best_F1:
                best_F1 = v_F1
                torch.save(model.state_dict(), os.path.join(checkpoints, 'bestF1.pt'))

            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), os.path.join(checkpoints, 'bestloss.pt'))

            if val_auc > best_auc:
                best_auc = val_auc
                torch.save(model.state_dict(), os.path.join(checkpoints, 'bestauc.pt'))

    if rank == 0 and os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    cleanup()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.1)
    parser.add_argument('--syncBN', type=bool, default=True)
    parser.add_argument('--weights', type=str, default='resNet34.pth')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--world-size', default=4, type=int)
    parser.add_argument('--dist-url', default='env://')
    opt = parser.parse_args()
    main(opt)

import datetime
import os
from functools import partial

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.unet import Unet
from nets.unet_training import get_lr_scheduler, set_optimizer_lr, weights_init
from utils.callbacks import EvalCallback, LossHistory
from utils.dataloader import UnetDataset, unet_dataset_collate
from utils.utils import download_weights, seed_everything, show_config, worker_init_fn
from utils.utils_fit import fit_one_epoch

def setup_distributed(distributed):
    ngpus_per_node = torch.cuda.device_count()
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        device = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        local_rank = 0
        rank = 0
    return device, local_rank, rank, ngpus_per_node

def maybe_download_backbone_weights(pretrained, distributed, backbone, local_rank):
    if not pretrained:
        return
    if distributed:
        if local_rank == 0:
            download_weights(backbone)
        dist.barrier()
    else:
        download_weights(backbone)

def build_model(num_classes, pretrained, backbone, model_path, device, local_rank):
    model = Unet(num_classes=num_classes, pretrained=pretrained, backbone=backbone).train()
    if not pretrained:
        weights_init(model)

    if model_path:
        if local_rank == 0:
            print(f"Load weights {model_path}.")

        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)

        temp_dict = {
            k: v
            for k, v in pretrained_dict.items()
            if k in model_dict and np.shape(model_dict[k]) == np.shape(v)
        }

        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)

    return model

def wrap_model_for_cuda(model, Cuda, distributed, sync_bn, fp16, ngpus_per_node, local_rank):
    scaler = None
    if fp16:
        from torch.cuda.amp import GradScaler
        scaler = GradScaler()

    model_train = model.train()

    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)

    if Cuda:
        if distributed:
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(
                model_train, device_ids=[local_rank], find_unused_parameters=True
            )
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()

    return model_train, scaler


def load_train_val_lines(train_txt_path, val_txt_path):
    with open(train_txt_path, "r") as f:
        train_lines = f.readlines()
    with open(val_txt_path, "r") as f:
        val_lines = f.readlines()
    return train_lines, val_lines


def make_optimizer(optimizer_type, model, init_lr_fit, momentum, weight_decay):
    if optimizer_type == "adam":
        return optim.Adam(model.parameters(), init_lr_fit, betas=(momentum, 0.999), weight_decay=weight_decay)
    if optimizer_type == "sgd":
        return optim.SGD(model.parameters(), init_lr_fit, momentum=momentum, nesterov=True, weight_decay=weight_decay)
    raise ValueError


def compute_lr_fit(batch_size, nbs, Init_lr, Min_lr, optimizer_type):
    lr_limit_max = 1e-4 if optimizer_type == "adam" else 1e-1
    lr_limit_min = 1e-4 if optimizer_type == "adam" else 5e-4
    init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
    min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
    return init_lr_fit, min_lr_fit


def main():
    Cuda = True
    seed = 11
    distributed = True
    sync_bn = True
    fp16 = False

    num_classes = 2
    backbone = "resnet50"
    pretrained = False
    model_path = "～/model_data/unet_resnet.pth"
    input_shape = [1024, 1024]

    Init_Epoch = 0
    Freeze_Epoch = 50
    Freeze_batch_size = 2
    UnFreeze_Epoch = 100
    Unfreeze_batch_size = 2
    Freeze_Train = False

    Init_lr = 1e-2
    Min_lr = Init_lr * 0.01
    optimizer_type = "sgd"
    momentum = 0.9
    weight_decay = 0.0005
    lr_decay_type = "cos"

    save_period = 1
    save_dir = "logs"
    eval_flag = True
    eval_period = 1

    dataset_root = "～/"
    dice_loss = True
    focal_loss = True
    cls_weights = np.ones([num_classes], np.float32)
    num_workers = 8

    train_txt_path = "～/train.txt"
    val_txt_path = "～/val.txt"

    seed_everything(seed)
    device, local_rank, rank, ngpus_per_node = setup_distributed(distributed)
    maybe_download_backbone_weights(pretrained, distributed, backbone, local_rank)

    model = build_model(num_classes, pretrained, backbone, model_path, device, local_rank)
    model_train, scaler = wrap_model_for_cuda(model, Cuda, distributed, sync_bn, fp16, ngpus_per_node, local_rank)

    if local_rank == 0:
        time_str = datetime.datetime.strftime(datetime.datetime.now(), "%Y_%m_%d_%H_%M_%S")
        log_dir = os.path.join(save_dir, "loss_" + time_str)
        loss_history = LossHistory(log_dir, model, input_shape=input_shape)
    else:
        log_dir = None
        loss_history = None

    train_lines, val_lines = load_train_val_lines(train_txt_path, val_txt_path)
    num_train = len(train_lines)
    num_val = len(val_lines)

    if local_rank == 0:
        show_config(
            num_classes=num_classes,
            backbone=backbone,
            model_path=model_path,
            input_shape=input_shape,
            Init_Epoch=Init_Epoch,
            Freeze_Epoch=Freeze_Epoch,
            UnFreeze_Epoch=UnFreeze_Epoch,
            Freeze_batch_size=Freeze_batch_size,
            Unfreeze_batch_size=Unfreeze_batch_size,
            Freeze_Train=Freeze_Train,
            Init_lr=Init_lr,
            Min_lr=Min_lr,
            optimizer_type=optimizer_type,
            momentum=momentum,
            lr_decay_type=lr_decay_type,
            save_period=save_period,
            save_dir=save_dir,
            num_workers=num_workers,
            num_train=num_train,
            num_val=num_val,
        )

    UnFreeze_flag = False
    if Freeze_Train:
        model.freeze_backbone()

    batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size
    nbs = 16
    Init_lr_fit, Min_lr_fit = compute_lr_fit(batch_size, nbs, Init_lr, Min_lr, optimizer_type)
    optimizer = make_optimizer(optimizer_type, model, Init_lr_fit, momentum, weight_decay)
    lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

    epoch_step = num_train // batch_size
    epoch_step_val = num_val // batch_size
    if epoch_step == 0 or epoch_step_val == 0:
        raise ValueError

    train_dataset = UnetDataset(train_lines, input_shape, num_classes, True, dataset_root)
    val_dataset = UnetDataset(val_lines, input_shape, num_classes, False, dataset_root)

    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
        batch_size //= ngpus_per_node
        shuffle = False
    else:
        train_sampler = None
        val_sampler = None
        shuffle = True

    gen = DataLoader(
        train_dataset,
        shuffle=shuffle,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=unet_dataset_collate,
        sampler=train_sampler,
        worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed),
    )
    gen_val = DataLoader(
        val_dataset,
        shuffle=shuffle,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=unet_dataset_collate,
        sampler=val_sampler,
        worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed),
    )

    if local_rank == 0:
        eval_callback = EvalCallback(
            model,
            input_shape,
            num_classes,
            val_lines,
            dataset_root,
            log_dir,
            Cuda,
            eval_flag=eval_flag,
            period=eval_period,
        )
    else:
        eval_callback = None

    for epoch in range(Init_Epoch, UnFreeze_Epoch):
        if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
            batch_size = Unfreeze_batch_size
            Init_lr_fit, Min_lr_fit = compute_lr_fit(batch_size, nbs, Init_lr, Min_lr, optimizer_type)
            lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
            model.unfreeze_backbone()

            epoch_step = num_train // batch_size
            epoch_step_val = num_val // batch_size
            if epoch_step == 0 or epoch_step_val == 0:
                raise ValueError

            if distributed:
                batch_size //= ngpus_per_node

            gen = DataLoader(
                train_dataset,
                shuffle=shuffle,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=True,
                collate_fn=unet_dataset_collate,
                sampler=train_sampler,
                worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed),
            )
            gen_val = DataLoader(
                val_dataset,
                shuffle=shuffle,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=True,
                collate_fn=unet_dataset_collate,
                sampler=val_sampler,
                worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed),
            )
            UnFreeze_flag = True

        if distributed:
            train_sampler.set_epoch(epoch)

        set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

        fit_one_epoch(
            model_train,
            model,
            loss_history,
            eval_callback,
            optimizer,
            epoch,
            epoch_step,
            epoch_step_val,
            gen,
            gen_val,
            UnFreeze_Epoch,
            Cuda,
            dice_loss,
            focal_loss,
            cls_weights,
            num_classes,
            fp16,
            scaler,
            save_period,
            save_dir,
            local_rank,
        )

        if distributed:
            dist.barrier()

    if local_rank == 0 and loss_history is not None:
        loss_history.writer.close()


if __name__ == "__main__":
    main()

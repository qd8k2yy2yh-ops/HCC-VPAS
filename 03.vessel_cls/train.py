import os
import math
import argparse
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torchvision
import torch.nn as nn
from cjg_utils import ConfusionMatrix
from my_dataset import MyDataSet
from utils import read_split_data
from multi_train_utils.distributed_utils import init_distributed_mode, dist, cleanup
from multi_train_utils.train_eval_utils import train_one_epoch
import sys
from multi_train_utils.distributed_utils import reduce_value


def main(args):
    if not torch.cuda.is_available():
        raise EnvironmentError("No GPU device found for training.")

    init_distributed_mode(args)

    rank = args.rank
    device = torch.device(args.device)
    batch_size = args.batch_size
    args.lr *= args.world_size
    checkpoint_path = ""

    if rank == 0:
        print(args)
        print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
        tb_writer = SummaryWriter()
        os.makedirs("./weights", exist_ok=True)

    train_info, val_info, num_classes = read_split_data(args.data_path)
    train_images_path, train_images_label = train_info
    val_images_path, val_images_label = val_info

    assert args.num_classes == num_classes, f"Dataset num_classes: {num_classes}, input {args.num_classes}"

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    train_data_set = MyDataSet(images_path=train_images_path, images_class=train_images_label, transform=data_transform["train"])
    val_data_set = MyDataSet(images_path=val_images_path, images_class=val_images_label, transform=data_transform["val"])

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data_set)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_data_set)

    train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, batch_size, drop_last=True)
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    if rank == 0:
        print(f'Using {nw} dataloader workers every process')

    train_loader = torch.utils.data.DataLoader(train_data_set, batch_sampler=train_batch_sampler, pin_memory=True, num_workers=nw, collate_fn=train_data_set.collate_fn)
    val_loader = torch.utils.data.DataLoader(val_data_set, batch_size=batch_size, sampler=val_sampler, pin_memory=True, num_workers=nw, collate_fn=val_data_set.collate_fn)

    model = torchvision.models.resnet50(pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    model = model.to(device)

    model_name = 'resnet50'
    os.makedirs(f"./weights_{model_name}", exist_ok=True)

    if args.freeze_layers:
        for name, para in model.named_parameters():
            if "fc" not in name:
                para.requires_grad_(False)
    else:
        if args.syncBN:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=0.005)

    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    best_acc = 0.0

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)

        mean_loss = train_one_epoch(model=model,
                                    optimizer=optimizer,
                                    data_loader=train_loader,
                                    data_set=train_data_set,
                                    num_classes=num_classes,
                                    device=device,
                                    epoch=epoch)

        scheduler.step()

        cm = ConfusionMatrix(num_classes=num_classes, labels=["class1", "class2"])
        model.eval()
        sum_num = torch.zeros(1).to(device)
        with torch.no_grad():
            for step, data in enumerate(tqdm(val_loader, file=sys.stdout)):
                images, labels, _ = data
                images = images.to(device)
                preds = model(images)
                preds = torch.max(preds, dim=1)[1]
                sum_num += torch.eq(preds, labels.to(device)).sum()
                cm.update(preds.cpu().numpy(), labels.cpu().numpy())

        if device != torch.device("cpu"):
            torch.cuda.synchronize(device)

        sum_num = reduce_value(sum_num, average=False)
        acc = sum_num.item() / val_sampler.total_size

        if rank == 0:
            print(f"[epoch {epoch}] accuracy: {acc:.3f}")
            tb_writer.add_scalar("loss", mean_loss, epoch)
            tb_writer.add_scalar("accuracy", acc, epoch)
            tb_writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], epoch)

            torch.save(model.module.state_dict(), f"./weights_{model_name}/model-{epoch}.pth")

            if acc > best_acc:
                best_acc = acc
                best_epoch = epoch
                print(f'Best epoch = {best_epoch}')
                cm.save_cm(save_csv_filename=f'./results_{model_name}/confusion_matrix_acc={best_acc}_best_epoch={best_epoch}.csv',
                           save_percentage_filename=f'./results_{model_name}/percent_confusion_matrix_acc={best_acc}_best_epoch={best_epoch}.csv')
                cm.summary()

    if rank == 0:
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)

    cleanup()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=48)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.1)
    parser.add_argument('--syncBN', type=bool, default=True)
    parser.add_argument('--data-path', type=str, default="～/dataset/")
    parser.add_argument('--weights', type=str, default='./resnet34.pth', help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--world-size', default=4, type=int, help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    opt = parser.parse_args()

    main(opt)

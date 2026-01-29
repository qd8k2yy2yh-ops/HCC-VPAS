import os
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
import scipy.signal
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from .utils import cvtColor, preprocess_input, resize_image
from .utils_metrics import compute_mIoU, show_results


class LossHistory():
    def __init__(self, log_dir, model, input_shape, val_loss_flag=True):
        self.log_dir = log_dir
        self.val_loss_flag = val_loss_flag
        self.losses = []
        if self.val_loss_flag:
            self.val_loss = []
        os.makedirs(self.log_dir)
        self.writer = SummaryWriter(self.log_dir)
        try:
            dummy_input = torch.randn(2, 3, input_shape[0], input_shape[1])
            self.writer.add_graph(model, dummy_input)
        except:
            pass

    def append_loss(self, epoch, loss, val_loss=None):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.losses.append(loss)
        if self.val_loss_flag:
            self.val_loss.append(val_loss)

        with open(os.path.join(self.log_dir, "epoch_loss.txt"), 'a') as f:
            f.write(f"{loss}\n")
        if self.val_loss_flag:
            with open(os.path.join(self.log_dir, "epoch_val_loss.txt"), 'a') as f:
                f.write(f"{val_loss}\n")

        self.writer.add_scalar('loss', loss, epoch)
        if self.val_loss_flag:
            self.writer.add_scalar('val_loss', val_loss, epoch)
        self.loss_plot()

    def loss_plot(self):
        iters = range(len(self.losses))
        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth=2, label='train loss')
        if self.val_loss_flag:
            plt.plot(iters, self.val_loss, 'coral', linewidth=2, label='val loss')
        try:
            num = 5 if len(self.losses) < 25 else 15
            plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle='--', linewidth=2,
                     label='smooth train loss')
            if self.val_loss_flag:
                plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), '#8B4513', linestyle='--',
                         linewidth=2, label='smooth val loss')
        except:
            pass
        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")
        plt.savefig(os.path.join(self.log_dir, "epoch_loss.png"))
        plt.cla()
        plt.close("all")


class EvalCallback():
    def __init__(self, net, input_shape, num_classes, image_ids, dataset_path, log_dir, cuda,
                 miou_out_path=".temp_miou_out", eval_flag=True, period=1):
        self.net = net
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.image_ids = [image_id.split()[0] for image_id in image_ids]
        self.dataset_path = dataset_path
        self.log_dir = log_dir
        self.cuda = cuda
        self.miou_out_path = miou_out_path
        self.eval_flag = eval_flag
        self.period = period
        self.mious = [0]
        self.epoches = [0]
        if self.eval_flag:
            with open(os.path.join(self.log_dir, "epoch_miou.txt"), 'a') as f:
                f.write("0\n")

    def get_miou_png(self, image):
        image = cvtColor(image)
        orininal_h, orininal_w = np.array(image).shape[:2]
        image_data, nw, nh = resize_image(image, (self.input_shape[1], self.input_shape[0]))
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            pr = self.net(images)[0]
            pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()
            pr = pr[int((self.input_shape[0] - nh) // 2):int((self.input_shape[0] - nh) // 2 + nh),
                 int((self.input_shape[1] - nw) // 2):int((self.input_shape[1] - nw) // 2 + nw)]
            pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation=cv2.INTER_LINEAR)
            pr = pr.argmax(axis=-1)

        return Image.fromarray(np.uint8(pr))

    def on_epoch_end(self, epoch, model_eval):
        if epoch % self.period == 0 and self.eval_flag:
            self.net = model_eval
            gt_dir = os.path.join(self.dataset_path, "00_seg_label")
            pred_dir = os.path.join(self.miou_out_path, 'detection-results')
            os.makedirs(self.miou_out_path, exist_ok=True)
            os.makedirs(pred_dir, exist_ok=True)
            print("Get miou.")
            for image_id in tqdm(self.image_ids):
                image_path = os.path.join(self.dataset_path, "00_original_images", f"{image_id}.jpg")
                image = Image.open(image_path)
                image = self.get_miou_png(image)
                image.save(os.path.join(pred_dir, f"{image_id}.png"))

            print("Calculate miou.")
            hist, IoUs, PA_Recall, Precision = compute_mIoU(gt_dir, pred_dir, self.image_ids, self.num_classes, None)
            mean_IoUs = np.nanmean(IoUs)
            mean_PA_Recall = np.nanmean(PA_Recall)
            mean_Precision = np.nanmean(Precision)

            self.epoches.append(epoch)

            miou_out_path = self.log_dir
            name_classes = ["background", "microvasculature"]

            with open(os.path.join(self.log_dir, "epoch_metrics.txt"), 'a') as f:
                f.write(f"{epoch}\t{mean_IoUs}\t{mean_PA_Recall}\t{mean_Precision}\n")
                for class_idx in range(hist.shape[0]):
                    f.write(f"{name_classes[class_idx] if name_classes else class_idx}\t")
                    f.write("\t".join([str(round(hist[class_idx, i], 2)) for i in range(hist.shape[1])]))
                    f.write("\n")

            show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes)

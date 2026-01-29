import csv
import os
from os.path import join
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

def compute_segmentation_metrics(inputs, target, beta=1, smooth=1e-5, threshold=0.5):
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()

    if h != ht or w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    pred_probs = torch.softmax(inputs.permute(0, 2, 3, 1).contiguous().view(n, -1, c), dim=-1)
    target_onehot = target.view(n, -1, ct)

    pred_bin = (pred_probs > threshold).float()
    pred_bin = pred_bin[..., :-1]
    target_onehot = target_onehot[..., :-1]

    tp = (pred_bin * target_onehot).sum(dim=(0, 1))
    fp = (pred_bin * (1 - target_onehot)).sum(dim=(0, 1))
    fn = ((1 - pred_bin) * target_onehot).sum(dim=(0, 1))
    tn = ((1 - pred_bin) * (1 - target_onehot)).sum(dim=(0, 1))

    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    f1 = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    iou = (tp + smooth) / (tp + fp + fn + smooth)
    accuracy = (tp + tn + smooth) / (tp + tn + fp + fn + smooth)

    metrics = {
        "f_score": f1.mean().item(),
        "precision": precision.mean().item(),
        "recall": recall.mean().item(),
        "iou": iou.mean().item(),
        "accuracy": accuracy.mean().item()
    }
    return metrics


def f_score(inputs, target, beta=1, smooth=1e-5, threshold=0.5):
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()

    if h != ht or w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c), -1)
    temp_target = target.view(n, -1, ct)

    temp_inputs = torch.gt(temp_inputs, threshold).float()
    tp = torch.sum(temp_target[..., :-1] * temp_inputs, axis=[0, 1])
    fp = torch.sum(temp_inputs, axis=[0, 1]) - tp
    fn = torch.sum(temp_target[..., :-1], axis=[0, 1]) - tp

    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    score = torch.mean(score)
    return score


def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    return np.diag(hist) / np.maximum((hist.sum(1) + hist.sum(0) - np.diag(hist)), 1)


def per_class_PA_Recall(hist):
    return np.diag(hist) / np.maximum(hist.sum(1), 1)


def per_class_Precision(hist):
    return np.diag(hist) / np.maximum(hist.sum(0), 1)


def per_Accuracy(hist):
    return np.sum(np.diag(hist)) / np.maximum(np.sum(hist), 1)


def compute_mIoU(gt_dir, pred_dir, png_name_list, num_classes, name_classes=None):
    print('Num classes', num_classes)
    hist = np.zeros((num_classes, num_classes))

    gt_imgs = [join(gt_dir, x + ".png") for x in png_name_list]
    pred_imgs = [join(pred_dir, x + ".png") for x in png_name_list]

    for ind in range(len(gt_imgs)):
        pred = np.array(Image.open(pred_imgs[ind]))
        label = np.array(Image.open(gt_imgs[ind]))

        if len(label.flatten()) != len(pred.flatten()):
            print(f'Skipping: len(gt) = {len(label.flatten())}, len(pred) = {len(pred.flatten())}, {gt_imgs[ind]}, {pred_imgs[ind]}')
            continue

        hist += fast_hist(label.flatten(), pred.flatten(), num_classes)

        if name_classes is not None and ind > 0 and ind % 10 == 0:
            print(f'{ind} / {len(gt_imgs)}: mIou-{100 * np.nanmean(per_class_iu(hist)):.2f}%; mPA-{100 * np.nanmean(per_class_PA_Recall(hist)):.2f}%; Accuracy-{100 * per_Accuracy(hist):.2f}%')

    IoUs = per_class_iu(hist)
    PA_Recall = per_class_PA_Recall(hist)
    Precision = per_class_Precision(hist)

    total_pixels = np.sum(hist)
    hist_percent = (hist / total_pixels) * 100

    if name_classes is not None:
        for ind_class in range(num_classes):
            print(f'===>{name_classes[ind_class]}: IoU-{IoUs[ind_class] * 100:.2f}%; Recall-{PA_Recall[ind_class] * 100:.2f}%; Precision-{Precision[ind_class] * 100:.2f}%')

    print(f'===> mIoU: {np.nanmean(IoUs) * 100:.2f}%; mPA: {np.nanmean(PA_Recall) * 100:.2f}%; Accuracy: {per_Accuracy(hist) * 100:.2f}%')

    print("\nConfusion Matrix (%):")
    class_names = name_classes if name_classes is not None else [f"Class {i}" for i in range(num_classes)]
    header = ["{:>12}".format(name) for name in class_names]
    print("{:>12} | {}".format("GT\\Pred", " ".join(header)))
    print("-" * (14 + len(header) * 13))

    for i, row in enumerate(hist_percent):
        row_str = " ".join([f"{val:>12.2f}" for val in row])
        print(f"{class_names[i]:>12} | {row_str}")

    return np.array(hist_percent, np.float32), IoUs, PA_Recall, Precision


def adjust_axes(r, t, fig, axes):
    bb = t.get_window_extent(renderer=r)
    text_width_inches = bb.width / fig.dpi
    current_fig_width = fig.get_figwidth()
    new_fig_width = current_fig_width + text_width_inches
    propotion = new_fig_width / current_fig_width
    x_lim = axes.get_xlim()
    axes.set_xlim([x_lim[0], x_lim[1] * propotion])


def draw_plot_func(values, name_classes, plot_title, x_label, output_path, tick_font_size=12, plt_show=True):
    fig = plt.gcf()
    axes = plt.gca()
    plt.barh(range(len(values)), values, color='royalblue')
    plt.title(plot_title, fontsize=tick_font_size + 2)
    plt.xlabel(x_label, fontsize=tick_font_size)
    plt.yticks(range(len(values)), name_classes, fontsize=tick_font_size)

    r = fig.canvas.get_renderer()
    for i, val in enumerate(values):
        str_val = f" {val:.2f}" if val < 1.0 else f" {val}"
        t = plt.text(val, i, str_val, color='royalblue', va='center', fontweight='bold')
        if i == len(values) - 1:
            adjust_axes(r, t, fig, axes)

    fig.tight_layout()
    fig.savefig(output_path)
    if plt_show:
        plt.show()
    plt.close()


def show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes, tick_font_size=12, epoch=None):
    draw_plot_func(IoUs, name_classes, f"mIoU = {np.nanmean(IoUs) * 100:.4f}%", "Intersection over Union", os.path.join(miou_out_path, "mIoU.png"), tick_font_size=tick_font_size)
    draw_plot_func(PA_Recall, name_classes, f"mPA = {np.nanmean(PA_Recall) * 100:.4f}%", "Pixel Accuracy", os.path.join(miou_out_path, "mPA.png"), tick_font_size=tick_font_size, plt_show=False)
    draw_plot_func(PA_Recall, name_classes, f"mRecall = {np.nanmean(PA_Recall) * 100:.4f}%", "Recall", os.path.join(miou_out_path, "Recall.png"), tick_font_size=tick_font_size, plt_show=False)
    draw_plot_func(Precision, name_classes, f"mPrecision = {np.nanmean(Precision) * 100:.4f}%", "Precision", os.path.join(miou_out_path, "Precision.png"), tick_font_size=tick_font_size, plt_show=False)

    log_dir = "～/logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "metrics_log.csv")

    headers = ["Epoch", "mIoU", "mPA", "mRecall", "mPrecision"]
    values = [epoch, round(np.nanmean(IoUs), 4), round(np.nanmean(PA_Recall), 4), round(np.nanmean(PA_Recall), 4), round(np.nanmean(Precision), 4)]

    file_exists = os.path.isfile(log_path)
    with open(log_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(headers)
        writer.writerow(values)

    print(f"Saved metrics for epoch {epoch} to {log_path}")

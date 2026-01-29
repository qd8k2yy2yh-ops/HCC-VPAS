import pickle
import os
import json
import random
import matplotlib.pyplot as plt


def read_split_data(root: str, val_rate: float = 0.3):
    random.seed(0)
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    class_names = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    class_names.sort()
    class_indices = dict((k, v) for v, k in enumerate(class_names))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path, train_images_label = [], []
    val_images_path, val_images_label = [], []
    every_class_num = []
    supported = [".jpg", ".JPG", ".png", ".PNG"]

    patient_images = {}
    for cla in class_names:
        cla_path = os.path.join(root, cla)
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path) if os.path.splitext(i)[-1] in supported]
        for img in images:
            patient_id = os.path.basename(img).split('_')[0]
            if patient_id not in patient_images:
                patient_images[patient_id] = []
            patient_images[patient_id].append((img, class_indices[cla]))

    patient_ids = list(patient_images.keys())
    random.shuffle(patient_ids)
    split = int(len(patient_ids) * val_rate)
    val_patient_ids = set(patient_ids[:split])

    for patient_id, images in patient_images.items():
        if patient_id in val_patient_ids:
            for img, label in images:
                val_images_path.append(img)
                val_images_label.append(label)
        else:
            for img, label in images:
                train_images_path.append(img)
                train_images_label.append(label)

        every_class_num.append(len(images))

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))
    assert len(train_images_path) > 0, "number of training images must be greater than 0."
    assert len(val_images_path) > 0, "number of validation images must be greater than 0."

    plot_image_distribution = False
    if plot_image_distribution:
        plt.bar(range(len(class_names)), every_class_num, align='center')
        plt.xticks(range(len(class_names)), class_names)
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        plt.xlabel('image class')
        plt.ylabel('number of images')
        plt.title('Class Distribution in Dataset')
        plt.show()

    return [train_images_path, train_images_label], [val_images_path, val_images_label], len(class_names)


def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            img = images[i].numpy().transpose(1, 2, 0)
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i+1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])
            plt.yticks([])
            plt.imshow(img.astype('uint8'))
        plt.show()


def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list

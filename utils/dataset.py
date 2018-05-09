"""
-- User: Ashok Kumar Pant (asokpant@gmail.com)
-- Date: 5/7/18
-- Time: 11:18 AM
"""
import os
import traceback

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import sys


def pil2array(image):
    return np.asarray(image)


def array2pil(image):
    return Image.fromarray(np.uint8(image)).convert('RGB')


def imread(filename):
    image = Image.open(filename)
    return image


def im2bw(image):
    return image.convert('1')


def rgb2gray(image):
    return image.convert('L')


def imresize(image, size):
    return image.resize(size, Image.ANTIALIAS)


def normalize_array(image):
    return image / 255.0


def imshow_array(image):
    plt.imshow(image, cmap='gray')
    plt.show()


def imshow(image):
    plt.imshow(np.asarray(image), cmap='gray')
    plt.show()


def show_samples(data_path, row=3, col=10):
    files, id2label, label2id = get_filename_and_class(data_path=data_path)
    np.random.shuffle(files)

    v = None
    i = 0
    for r in range(row):
        h = None
        for c in range(col):
            image = imresize(image=imread(files[i][0]), size=(28, 28))
            image = normalize_array(pil2array(image))
            i += 1
            if h is None:
                h = image.copy()
                h = np.hstack((h, np.zeros((28, 1))))
            else:
                h = np.hstack((h, image, np.zeros((28, 1))))

        if v is None:
            v = h.copy()
            v = np.vstack((v, np.zeros((1, h.shape[1]))))
        else:
            v = np.vstack((v, h, np.zeros((1, h.shape[1]))))
    imshow_array(v)
    return v


def create_handwritten_dataset(data_path, test_ratio=0.2, hot_labels=True):
    files, id2label, label2id = get_filename_and_class(data_path=data_path)
    np.random.shuffle(files)
    num_test = int(test_ratio * len(files))

    np.random.shuffle(files)
    train_files = files[num_test:]
    test_files = files[:num_test]

    train_data = []
    train_labels = []

    test_data = []
    test_labels = []

    num_classes = len(id2label)

    for f in train_files:
        try:
            image = normalize_array(pil2array(image=imread(f[0])))
            image = np.reshape(image, (image.shape[0] * image.shape[1]))
            train_data.append(image)
            if hot_labels:
                y = np.zeros(shape=num_classes, dtype=np.float32)
                y[int(f[1])] = 1.0
                train_labels.append(y)
            else:
                train_labels.append(int(f[1]))
        except Exception as _:
            traceback.print_exc(file=sys.stdout)
            continue

    for f in test_files:
        try:
            image = normalize_array(pil2array(image=imread(f[0])))
            image = np.reshape(image, (image.shape[0] * image.shape[1]))
            test_data.append(image)
            if hot_labels:
                y = np.zeros(shape=num_classes, dtype=np.float32)
                y[int(f[1])] = 1.0
                test_labels.append(y)
            else:
                test_labels.append(int(f[1]))
        except Exception as _:
            traceback.print_exc(file=sys.stdout)
            continue

    train_data = np.array(train_data)
    train_labels = np.array(train_labels)
    test_data = np.array(test_data)
    test_labels = np.array(test_labels)

    if len(train_data) == 0:
        train_data = None
        train_labels = None
    if len(test_data) == 0:
        test_data = None
        test_labels = None
    return train_data, train_labels, test_data, test_labels, id2label


def get_filename_and_class(data_path, max_classes=0, min_samples_per_class=0):
    """Returns a list of filename and inferred class names.
  Args:
      :param data_path: A directory containing a set of subdirectories representing class names. Each subdirectory should contain PNG or JPG encoded images.
      :param min_samples_per_class:
    :param max_classes:
    data_path:
  Returns:
    A list of image file paths, relative to `data_path` and the list of
    subdirectories, representing class names.

  """
    folders = [name for name in os.listdir(data_path) if
               os.path.isdir(os.path.join(data_path, name))]

    if len(folders) == 0:
        raise ValueError(data_path + " does not contain valid sub directories.")
    directories = []
    for folder in folders:
        directories.append(os.path.join(data_path, folder))

    folders = sorted(folders)
    label2id = {}

    i = 0
    c = 0
    total_files = []
    for folder in folders:
        dir = os.path.join(data_path, folder)
        files = os.listdir(dir)
        if min_samples_per_class > 0 and len(files) < min_samples_per_class:
            continue

        for file in files:
            path = os.path.join(dir, file)
            total_files.append([path, i])
        label2id[folder] = i
        i += 1

        if 0 < max_classes <= c:
            break
        c += 1

    id2label = {v: k for k, v in label2id.items()}
    return np.array(total_files), id2label, label2id


if __name__ == '__main__':
    train_data, train_labels, test_data, test_labels, label_map = create_handwritten_dataset(
        "data/nhcd/numerals")
    print("Classes: {}={}".format(len(label_map), label_map))
    print("Train samples: {}, Test samples: {}".format(len(train_labels), len(test_labels)))
    print(train_data[0])

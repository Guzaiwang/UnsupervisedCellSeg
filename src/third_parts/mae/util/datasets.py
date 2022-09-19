# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------


import os
import PIL
import cv2
from torchvision import datasets, transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

import torch.utils.data as data
import numpy as np
import torch

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    root = os.path.join(args.data_path, 'train' if is_train else 'val')
    dataset = datasets.ImageFolder(root, transform=transform)

    print(dataset)

    return dataset


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


class CellDataset(data.Dataset):
    num_classes = 1
    default_resolution = [448, 448]
    mean = np.array([0.40789654, 0.44719302, 0.47026115],
                    dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.28863828, 0.27408164, 0.27809835],
                   dtype=np.float32).reshape(1, 1, 3)

    def __init__(self, root_folder, transform):
        # img_folder = '/home/zaiwang/Data/Cell_split/Lung5_Rep1/nuclear'
        self.img_name = []
        self.transformer = transform
        for sub_folder in os.listdir(root_folder):
            img_folder = os.path.join(root_folder, sub_folder, 'nuclear')
            for sub_folder in os.listdir(img_folder):
                sub_folder_path = os.path.join(img_folder, sub_folder)
                for img_name in os.listdir(sub_folder_path):
                    img_path = os.path.join(sub_folder_path, img_name)
                    if os.path.exists(img_path):
                        self.img_name.append(img_path)
        print("this dataset has {} images".format(len(self.img_name)))

    def __getitem__(self, index):
        img_path = self.img_name[index]

        img = PIL.Image.open(img_path)
        img = np.array(img) / 100.
        # img = cv2.imread(str(img_path), flags=cv2.IMREAD_ANYDEPTH).astype(np.float32)/100.
        img = cv2.resize(img, (224, 224))
        img = np.expand_dims(img, axis=2)
        img = np.tile(img, (1,1,3))
        img = np.transpose(img, (2, 0, 1))/655.36
        # with open(img_path, 'rb') as f:
        #     img = PIL.Image.open(f)
        #     img = img.convert('RGB')
        #
        # if self.transformer is not None:
        #     img = self.transformer(img)

        return torch.from_numpy(img).float(), torch.from_numpy(img).float()

    def __len__(self):
        return len(self.img_name)

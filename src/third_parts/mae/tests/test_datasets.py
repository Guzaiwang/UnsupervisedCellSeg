
import os
import PIL
import cv2
from torchvision import datasets, transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

import torch.utils.data as data
import numpy as np
import torch

from PIL import Image

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
        with open(img_path, 'rb') as f:
            img = PIL.Image.open(f)
            img = img.convert('RGB')

        if self.transformer is not None:
            img = self.transformer(img)

        return img, img

    def __len__(self):
        return len(self.img_name)


transform_train = transforms.Compose([
        transforms.RandomResizedCrop([224, 224], scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

#
# cell_path = '/home/zaiwang/Data/Cell_split'
# dataset_train = CellDataset(cell_path, transform_train)
#
# sampler_train = torch.utils.data.RandomSampler(dataset_train)
#
# data_loader_train = torch.utils.data.DataLoader(
#     dataset_train, sampler=sampler_train,
#     batch_size=1,
#     num_workers=1,
#     pin_memory=True,
#     drop_last=True,
# )
#
# for (sample, _) in enumerate(data_loader_train):
#     # print()
#     img = _[0].squeeze(0).numpy()
#     img = np.transpose(img, (1,2,0))
#     print(np.shape(img))
#     break
#
#


def test_torch_to_tensor_cv2():
    img_path = './demo/0_0.png'
    img = PIL.Image.open(img_path)
    img = np.array(img).astype(np.float32) / 100.
    # img = cv2.imread(str(img_path), flags=cv2.IMREAD_ANYDEPTH).astype(np.float32)/100.
    print(np.max(img))
    print(np.max(img))

    img = np.expand_dims(img, axis=2)
    print(np.shape(img))
    img = np.tile(img, (1,1,3))
    print(img)
    print(np.shape(img))
    img = np.transpose(img, (2, 0, 1))
    print(np.shape(img))




def _test_torch_to_tensor():
    img_path = './demo/0_0.png'
    pic = Image.open(img_path)
    print(pic.mode, 'pic.mode')
    print(pic.getbands(), 'pic.getbands()')
    mode_to_nptype = {"I": np.int32, "I;16": np.int16, "F": np.float32}
    mode_to_nptype.get(pic.mode, np.uint8)
    print(mode_to_nptype)

    img = torch.from_numpy(np.array(pic, mode_to_nptype.get(pic.mode, np.uint8), copy=True))
    print(torch.max(img), 'torch max')
    print(torch.min(img), 'torch min')

    img = img.view(pic.size[1], pic.size[0], len(pic.getbands()))
    print(img.shape)



    pic = pic.convert('RGB')
    img_rgb = torch.from_numpy(np.array(pic, mode_to_nptype.get(pic.mode, np.uint8), copy=True))
    print(torch.max(img_rgb), 'torch max')
    print(torch.min(img_rgb), 'torch min')
    # print("img_channels", img.channels)
    # print("img_height", img.height)
    # print("img_width", img.width)

if __name__ == '__main__':
    test_torch_to_tensor_cv2()
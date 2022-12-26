import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

default_trans = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


class GoodBadImgDataset(Dataset):
    def __init__(
            self,
            img_path_list: np.ndarray,
            label_list: np.ndarray,
            transform=default_trans,
            to_onehot=False,
            n_labels=2) -> None:
        self.img_path_list = img_path_list
        self.label_list = label_list
        self.transform = transform
        self.to_onehot = to_onehot
        self.n_labels = n_labels

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        label = self.label_list[index]
        if self.to_onehot:
            label = np.eye(self.n_labels)[label]
        img = Image.open(img_path)
        img = self.transform(img)
        return img, label


class BasicTrainImgTransform:
    def __init__(self, resize=(256, 256), degrees=180):
        self.transform = transforms.Compose([
            transforms.Resize(resize),
            # 左右反転
            transforms.RandomHorizontalFlip(),
            # 上下反転
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=degrees),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=MEAN,
                std=STD
            )
        ])

    def __call__(self, img) -> torch.Tensor:
        return self.transform(img)


class BasicValidTestImgTransform:
    def __init__(self, resize=(256, 256)):
        self.transform = transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=MEAN,
                std=STD
            )
        ])

    def __call__(self, img) -> torch.Tensor:
        return self.transform(img)

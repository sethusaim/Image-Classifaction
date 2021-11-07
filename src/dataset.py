import numpy as np
import torch
from PIL import Image
from torch.utils import data
from torchvision.datasets import ImageFolder


def get_imagefolder_dataset(dir, transform=None):

    try:
        img_folder_dataset = ImageFolder(root=dir, transform=transform)

        return img_folder_dataset

    except Exception as e:
        print(str(e))

        raise e


class ClassificationDataset(data.Dataset):
    def __init__(self, image_paths, targets, resize=None, augmentations=None):
        self.image_paths = image_paths

        self.targets = targets

        self.resize = resize

        self.augmentations = augmentations

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        image = Image.open(self.image_paths[item])

        image = image.convert("RGB")

        targets = self.targets[item]

        if self.resize is not None:
            image = image.resize(
                (self.resize[1], self.resize[0]), resample=Image.BILINEAR
            )

        image = np.array(image)

        if self.augmentations is not None:
            augmented = self.augmentations(image=image)

            image = augmented["image"]

        image = np.transpose(image, (2, 0, 1)).astype(np.float32)

        return {
            "images": torch.tensor(image, dtype=torch.float),
            "targets": torch.tensor(targets, dtype=torch.long),
        }

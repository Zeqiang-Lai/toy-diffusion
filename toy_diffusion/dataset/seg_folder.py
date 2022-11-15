from functools import partial
from pathlib import Path

import numpy as np
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms as T

from toy_diffusion.utils import exists


class SegFolderDataset(Dataset):
    def __init__(
        self,
        folder,
        image_size,
        exts=['png'],
        augment_horizontal_flip=False,
        convert_image_to=None
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        maybe_convert_fn = partial(convert_image_to, convert_image_to) if exists(convert_image_to) else nn.Identity()

        self.transform = T.Compose([
            T.Lambda(maybe_convert_fn),
            T.Resize(image_size),
            T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
            T.CenterCrop(image_size),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        img = self.transform(img)
        img = np.array(img).astype(np.uint8)
        return img


if __name__ == '__main__':
    dataset = SegFolderDataset('data/ade20k/annotations/training', 128)
    img = dataset.__getitem__(0)
    print(img)
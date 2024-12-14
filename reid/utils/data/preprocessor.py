from __future__ import absolute_import
import os
import os.path as osp
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import math
from PIL import Image

class Preprocessor(Dataset):
    def __init__(self, dataset, root=None, transform=None, transform_basic=None):
        super(Preprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform
        self.transform_basic=transform_basic
        

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
            return self._get_single_item(indices)

    def _get_single_item(self, index):
        # print(self.dataset)
        try:
            fname, pid, camid, domain = self.dataset[index]
        except:
            fname, pid, camid, domain, _ = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)

        img = Image.open(fpath).convert('RGB')

        if self.transform is not None:
            img_aug = self.transform(img)
        if self.transform_basic is not None:
            img_origin = self.transform_basic(img)
        else:
            img_origin=img


        return img_aug,img_origin, fname, pid, camid, domain

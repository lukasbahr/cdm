#!/usr/bin/env python3

import os

import torch
from torch.utils.data import Dataset

import numpy as np
import pandas as pd
from skimage import io, transform
import h5py


class H5Dataset(Dataset):
    """PIV HDF5 dataset."""

    def __init__(self, hdf5file, imgs_key='ComImages',
            labels_key='AllGenDetails'):
        """
        Args:
            hdf5file: The hdf5 file including the images and the label.
            out     : Dict with PIV Images and Flow Vectors.
        """
        self.db = h5py.File(hdf5file, 'r')
        keys = list(self.db.keys())

        if imgs_key not in keys:
            raise (' the ims_key should not be {}, should be one of {}'
                   .format(imgs_key, keys))
        if labels_key not in keys:
            raise (' the labels_key should not be {}, should be one of {}'
                   .format(labels_key, keys))

        self.imgs_key = imgs_key
        self.labels_key = labels_key

    def __len__(self):
        return len(self.db[self.labels_key])

    def __getitem__(self, idx):
        # NHWC to NCHW
        image = np.transpose(self.db[self.imgs_key][idx], (2 , 1, 0)) #NHWC -> NCHW
        image = torch.from_numpy(image)

        # Normalize image
        image = image.float().div(255)

        label = self.db[self.labels_key][idx]
        label = torch.from_numpy(label)
        sample = {self.imgs_key: image, self.labels_key: label}

        return sample


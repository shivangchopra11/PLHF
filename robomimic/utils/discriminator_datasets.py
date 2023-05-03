"""
Take content from Behavior Retrieval Repo
"""
import os
import h5py
import numpy as np
from copy import deepcopy
from contextlib import contextmanager

import torch.utils.data

import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.log_utils as LogUtils
from robomimic.utils.dataset import SequenceDataset


class ContrativeDataset(SequenceDataset):
    def __init__(self):
        pass

    def __getitem__(self, index):
        pass


class RNNDataset(SequenceDataset):
    def __init__(self):
        pass

    def __getitem__(self, index):
        pass


class PUDataset(SequenceDataset):
    def __init__(self):
        pass

    def __getitem__(self, index):
        pass
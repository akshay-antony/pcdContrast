import numpy as np
import open3d as o3d
from torch.utils.data import Dataset
import os
import yaml


class ShapeNetDataset(Dataset):
    def __init__(self, config, is_train=True):
        self.config = config
        self.is_train = is_train
        self.root = config["data"]["root"]
        self.category = config["data"]["category"]
        self.num_points = config["data"]["num_points"]
        self.num_classes = config["data"]["num_classes"]
        self.data = []
        self.label = []
        self.load_data()

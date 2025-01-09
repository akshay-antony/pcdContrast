import numpy as np
import open3d as o3d
from torch.utils.data import Dataset
import os
import yaml
import torch
import trimesh


class ShapeNetDataset(Dataset):
    def __init__(self, 
                 config,
                 filenames,
                 is_train=True):
        self.config = config
        self.pointcloud_folder = config['pointcloud_folder']
        self.image_folder = config['image_folder']
        self.filenames = filenames
        self.is_train = is_train

    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        filename = self.filenames[idx]
        pointcloud_path = os.path.join(self.pointcloud_folder, filename)
        embeddings_path = os.path.join(self.image_folder, filename)
        # insert "models" before the .obj 
        # replace obj with npy
        pcd_filename = pointcloud_path.replace("model_normalized.obj", "models/model_normalized.npy")
        embeddings_filename = "/".join(embeddings_path.split("/")[:-1]) + "/mean_embeddings.npy"

        pcd_np = np.load(pcd_filename)
        if self.is_train:
            pcd_np = self.random_sample(pcd_np)

        if self.is_train and self.config['use_rotate_augmentation']:
            pcd_np = self.random_rotation(pcd_np)

        pcd_np = self.pc_norm(pcd_np)
        pointcloud = torch.from_numpy(pcd_np).float()
        embeddings = np.load(embeddings_filename)
        embeddings = torch.from_numpy(embeddings).float()
        return {'pointcloud': pointcloud, 'embeddings': embeddings}
        
    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc
    
    def random_sample(self, pcd):
        """ pcd: NxC, return num_points x C """
        if len(pcd) > self.config['num_points']:
            idx = np.random.choice(len(pcd), self.config['num_points'], replace=False)
            pcd = pcd[idx]
        else:
            idx = np.random.choice(len(pcd), self.config['num_points'] - len(pcd), replace=True)
            pcd = np.concatenate((pcd, pcd[idx]), axis=0)
        return pcd
        
    def random_rotation(self, pcd):
        # uniformly sample roll, pitch, yaw
        roll = np.random.uniform(0, 2*np.pi)
        pitch = np.random.uniform(0, 2*np.pi)
        yaw = np.random.uniform(0, 2*np.pi)
        R = self.euler2mat(roll, pitch, yaw)
        pcd = np.dot(pcd, R)
        return pcd
    
    def euler2mat(self, roll, pitch, yaw):
        """ roll, pitch, yaw: scalar """
        # ZYX order
        R_roll = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
        R_pitch = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
        R_yaw = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        R = np.dot(R_yaw, np.dot(R_pitch, R_roll))
        return R

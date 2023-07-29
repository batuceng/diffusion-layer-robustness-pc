#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: data.py
@Time: 2018/10/13 6:21 PM

Modified by 
@Author: An Tao, Pengliang Ji, Ziyi Wu
@Contact: ta19@mails.tsinghua.edu.cn, jpl1723@buaa.edu.cn, dazitu616@gmail.com
@Time: 2022/7/30 7:49 PM

MODIFIED TO CONTAIN ONLY MODELNET40 CLASSIFICATION DATASET
"""

import torch
import os
import glob
import h5py
import numpy as np
from copy import copy
from torch.utils.data import Dataset
import random


def download_modelnet40():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget --no-check-certificate %s; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % ('modelnet40_ply_hdf5_2048', DATA_DIR))
        os.system('rm %s' % (zipfile))

def load_data_cls(partition):
    download_modelnet40()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', '*%s*.h5'%partition)):
        f = h5py.File(h5_name, 'r+')
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return torch.from_numpy(all_data), torch.from_numpy(all_label)

def translate_pointcloud(pointcloud):
    xyz1 = (3./2. - 2./3) * torch.rand(3, dtype=pointcloud.dtype) + 2./3
    xyz2 = (0.2 - (-0.2)) * torch.rand(3, dtype=pointcloud.dtype) + (-0.2)
    
    translated_pointcloud = torch.add(torch.multiply(pointcloud, xyz1), xyz2)
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud


def rotate_pointcloud(pointcloud):
    theta = np.pi*2 * np.random.uniform()
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    pointcloud[:,[0,2]] = pointcloud[:,[0,2]].dot(rotation_matrix) # random rotation (x,z)
    return pointcloud


# Rewritten ModelNet40 dataloader based on ShapeNetCore class

class ModelNet40(Dataset):
    def __init__(self, num_points, scale_mode, partition='train'):
        self.data, self.label = load_data_cls(partition)
        self.num_points = num_points
        self.partition = partition
        self.stats = self.get_statistics()
        assert scale_mode is None or scale_mode in ("none", 'global_unit', 'shape_unit', 'shape_bbox', 'shape_half', 'shape_34')
        self.scale_mode = scale_mode
        self.pointclouds = []
        self.normalize()
        
        
    def get_statistics(self):
        # Use train split to get statistics
        data, _ = load_data_cls("train")
        B, N, _ = data.size()
        mean = data.view(B*N, -1).mean(dim=0) # (1, 3)
        std = data.view(-1).std(dim=0)        # (1, )

        self.stats = {'mean': mean, 'std': std}
        
        return self.stats
    
    def normalize(self):
        for idx, (pc, label) in enumerate(zip(self.data, self.label)):
            if self.scale_mode == 'global_unit':
                shift = pc.mean(dim=0).reshape(1, 3)
                scale = self.stats['std'].reshape(1, 1)
            elif self.scale_mode == 'shape_unit':
                shift = pc.mean(dim=0).reshape(1, 3)
                scale = pc.flatten().std().reshape(1, 1)
            elif self.scale_mode == 'shape_half':
                shift = pc.mean(dim=0).reshape(1, 3)
                scale = pc.flatten().std().reshape(1, 1) / (0.5)
            elif self.scale_mode == 'shape_34':
                shift = pc.mean(dim=0).reshape(1, 3)
                scale = pc.flatten().std().reshape(1, 1) / (0.75)
            elif self.scale_mode == 'shape_bbox':
                pc_max, _ = pc.max(dim=0, keepdim=True) # (1, 3)
                pc_min, _ = pc.min(dim=0, keepdim=True) # (1, 3)
                shift = ((pc_min + pc_max) / 2).view(1, 3)
                scale = (pc_max - pc_min).max().reshape(1, 1) / 2
            else:
                shift = torch.zeros([1, 3])
                scale = torch.ones([1, 1])

            pc = (pc - shift) / scale

            self.pointclouds.append({
                    'pointcloud': pc,
                    'cate': label,
                    'id': idx,
                    'shift': shift,
                    'scale': scale
                })
            
        # Deterministically shuffle the dataset
        self.pointclouds.sort(key=lambda data: data['id'], reverse=False)
        random.Random(2020).shuffle(self.pointclouds)
    
    def __getitem__(self, item):
        data = {k:v.clone() if isinstance(v, torch.Tensor) else copy(v) for k, v in self.pointclouds[item].items()}
        data["pointcloud"] = data["pointcloud"][:self.num_points]  # select first 1024 points
        
        if self.partition == 'train':
            pointcloud = data["pointcloud"]
            pointcloud = translate_pointcloud(pointcloud)
            pointcloud = pointcloud[torch.randperm(pointcloud.size()[0])]
            data["pointcloud"] = pointcloud
        
        return data

    def __len__(self):
        return len(self.pointclouds)
    

class ModelNet40Attack(Dataset):
    def __init__(self, num_points):
        self.path = "test_attack_outputs"
        self.transform = False
        self.data, self.label, self.shift, self.scale = self.load_data(self.path)
        self.num_points = num_points
        self.pointclouds = []
        self.normalize()
    
    def load_data(self, path):
        data = np.load(os.path.join(path, "attack.npy"))
        label = np.loadtxt(os.path.join(path, "true_label.txt"))
        shift = np.load(os.path.join(path, "shift.npy"))
        scale = np.load(os.path.join(path, "scale.npy"))
        return data, label, shift, scale
    
    def normalize(self):
        for idx, (pc, label, shift, scale) in enumerate(zip(self.data, self.label, self.shift, self.scale)):
            
            pc = (pc - shift) / scale

            self.pointclouds.append({
                    'pointcloud': pc,
                    'cate': label,
                    'id': idx,
                    'shift': shift,
                    'scale': scale
                })
        
        # Deterministically shuffle the dataset
        self.pointclouds.sort(key=lambda data: data['id'], reverse=False)
        random.Random(2020).shuffle(self.pointclouds)
    
    def __getitem__(self, item):
        data = {k:v.clone() if isinstance(v, torch.Tensor) else copy(v) for k, v in self.pointclouds[item].items()}
        # data["pointcloud"] = data["pointcloud"][:self.num_points]  # select first 1024 points
        
        pointcloud = data["pointcloud"]
        if self.transform:
            pointcloud = translate_pointcloud(pointcloud)
            pointcloud = pointcloud[torch.randperm(pointcloud.size()[0])]
        data["pointcloud"] = pointcloud
        
        return data

    def __len__(self):
        return len(self.pointclouds)
    

LABELS = ["airplane",
"bathtub",
"bed",
"bench",
"bookshelf",
"bottle",
"bowl",
"car",
"chair",
"cone",
"cup",
"curtain",
"desk",
"door",
"dresser",
"flower_pot",
"glass_box",
"guitar",
"keyboard",
"lamp",
"laptop",
"mantel",
"monitor",
"night_stand",
"person",
"piano",
"plant",
"radio",
"range_hood",
"sink",
"sofa",
"stairs",
"stool",
"table",
"tent",
"toilet",
"tv_stand",
"vase",
"wardrobe",
"xbox"
]
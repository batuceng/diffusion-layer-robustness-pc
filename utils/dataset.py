import os
import random
from copy import copy
import torch
from torch.utils.data import Dataset
import numpy as np
import h5py
from tqdm.auto import tqdm


synsetid_to_cate = {
    '02691156': 'airplane', '02773838': 'bag', '02801938': 'basket',
    '02808440': 'bathtub', '02818832': 'bed', '02828884': 'bench',
    '02876657': 'bottle', '02880940': 'bowl', '02924116': 'bus',
    '02933112': 'cabinet', '02747177': 'can', '02942699': 'camera',
    '02954340': 'cap', '02958343': 'car', '03001627': 'chair',
    '03046257': 'clock', '03207941': 'dishwasher', '03211117': 'monitor',
    '04379243': 'table', '04401088': 'telephone', '02946921': 'tin_can',
    '04460130': 'tower', '04468005': 'train', '03085013': 'keyboard',
    '03261776': 'earphone', '03325088': 'faucet', '03337140': 'file',
    '03467517': 'guitar', '03513137': 'helmet', '03593526': 'jar',
    '03624134': 'knife', '03636649': 'lamp', '03642806': 'laptop',
    '03691459': 'speaker', '03710193': 'mailbox', '03759954': 'microphone',
    '03761084': 'microwave', '03790512': 'motorcycle', '03797390': 'mug',
    '03928116': 'piano', '03938244': 'pillow', '03948459': 'pistol',
    '03991062': 'pot', '04004475': 'printer', '04074963': 'remote_control',
    '04090263': 'rifle', '04099429': 'rocket', '04225987': 'skateboard',
    '04256520': 'sofa', '04330267': 'stove', '04530566': 'vessel',
    '04554684': 'washer', '02992529': 'cellphone',
    '02843684': 'birdhouse', '02871439': 'bookshelf',
    # '02858304': 'boat', no boat in our dataset, merged into vessels
    # '02834778': 'bicycle', not in our taxonomy
}
cate_to_synsetid = {v: k for k, v in synsetid_to_cate.items()}


class ShapeNetCore(Dataset):

    GRAVITATIONAL_AXIS = 1
    
    def __init__(self, path, cates, split, scale_mode, transform=None):
        super().__init__()
        assert isinstance(cates, list), '`cates` must be a list of cate names.'
        assert split in ('train', 'val', 'test')
        assert scale_mode is None or scale_mode in ('global_unit', 'shape_unit', 'shape_bbox', 'shape_half', 'shape_34')
        self.path = path
        if 'all' in cates:
            cates = cate_to_synsetid.keys()
        self.cate_synsetids = [cate_to_synsetid[s] for s in cates]
        self.cate_synsetids.sort()
        self.split = split
        self.scale_mode = scale_mode
        self.transform = transform

        self.pointclouds = []
        self.stats = None

        self.get_statistics()
        self.load()

    def get_statistics(self):

        basename = os.path.basename(self.path)
        dsetname = basename[:basename.rfind('.')]
        stats_dir = os.path.join(os.path.dirname(self.path), dsetname + '_stats')
        os.makedirs(stats_dir, exist_ok=True)

        if len(self.cate_synsetids) == len(cate_to_synsetid):
            stats_save_path = os.path.join(stats_dir, 'stats_all.pt')
        else:
            stats_save_path = os.path.join(stats_dir, 'stats_' + '_'.join(self.cate_synsetids) + '.pt')
        if os.path.exists(stats_save_path):
            self.stats = torch.load(stats_save_path)
            return self.stats

        with h5py.File(self.path, 'r') as f:
            pointclouds = []
            for synsetid in self.cate_synsetids:
                for split in ('train', 'val', 'test'):
                    pointclouds.append(torch.from_numpy(f[synsetid][split][...]))

        all_points = torch.cat(pointclouds, dim=0) # (B, N, 3)
        B, N, _ = all_points.size()
        mean = all_points.view(B*N, -1).mean(dim=0) # (1, 3)
        std = all_points.view(-1).std(dim=0)        # (1, )

        self.stats = {'mean': mean, 'std': std}
        torch.save(self.stats, stats_save_path)
        return self.stats

    def load(self):

        def _enumerate_pointclouds(f):
            for synsetid in self.cate_synsetids:
                cate_name = synsetid_to_cate[synsetid]
                for j, pc in enumerate(f[synsetid][self.split]):
                    yield torch.from_numpy(pc), j, cate_name
        
        with h5py.File(self.path, mode='r') as f:
            for pc, pc_id, cate_name in _enumerate_pointclouds(f):
                
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
                    'cate': cate_name,
                    'id': pc_id,
                    'shift': shift,
                    'scale': scale
                })

        # Deterministically shuffle the dataset
        self.pointclouds.sort(key=lambda data: data['id'], reverse=False)
        random.Random(2020).shuffle(self.pointclouds)

    def __len__(self):
        return len(self.pointclouds)

    def __getitem__(self, idx):
        data = {k:v.clone() if isinstance(v, torch.Tensor) else copy(v) for k, v in self.pointclouds[idx].items()}
        if self.transform is not None:
            data = self.transform(data)
        return data

class Layer1(Dataset):
    def __init__(self, layer_name="x1", transform=None, split="train", standardize=False):
        
        self.standardize = standardize
        self.transform = transform
        self.zero_pad = None # Set to 256 for enabling zero padding
        
        self.split = split
        self.layer_name = layer_name
        
        assert self.split in ["train", "val", "test"]
        assert self.layer_name in ["original","x1","x2","x3","x4"]
        
        
        if split == "train" or split == "val":
            self.labels = np.load("/home/robust/Desktop/diffusion-point-cloud-main/layer_data/train/input_label.npy")
            
            if layer_name == "x1":
                self.layer_data = np.load("/home/robust/Desktop/diffusion-point-cloud-main/layer_data/train/x1.npy")
                self.layer_data = np.transpose(self.layer_data, (0, 2, 1))  # B x N x F  (batch, point, channel)
            elif layer_name == "x2":
                self.layer_data = np.load("/home/robust/Desktop/diffusion-point-cloud-main/layer_data/train/x2.npy")
                self.layer_data = np.transpose(self.layer_data, (0, 2, 1))
            elif layer_name == "x3":
                self.layer_data = np.load("/home/robust/Desktop/diffusion-point-cloud-main/layer_data/train/x3.npy")
                self.layer_data = np.transpose(self.layer_data, (0, 2, 1))
            elif layer_name == "x4":
                self.layer_data = np.load("/home/robust/Desktop/diffusion-point-cloud-main/layer_data/train/x4.npy")
                self.layer_data = np.transpose(self.layer_data, (0, 2, 1))
            else:
                self.layer_data = np.load("/home/robust/Desktop/diffusion-point-cloud-main/layer_data/train/input_data.npy")
                self.layer_data = np.transpose(self.layer_data, (0, 2, 1))
            
            # if standardize:
            #     self.mean = np.mean(self.layer_data, axis=1)
            #     self.std = np.std(self.layer_data.reshape(self.labels.shape[0], -1), axis=1)
        
        
            mask = np.random.permutation(len(self.labels))
            train = mask[0:len(mask)*4//5]
            val = mask[len(mask)*4//5:len(mask)]
                            
            if split == "val":
                self.layer_data = self.layer_data[val]            
                self.labels = self.labels[val]
            else:
                self.layer_data = self.layer_data[train]
                self.labels = self.labels[train]

        elif self.split == "test": # test split
            self.labels = np.load("/home/robust/Desktop/diffusion-point-cloud-main/layer_data/test/input_label_test.npy")
            if self.layer_name == "original":
                self.layer_data = np.load(f"/home/robust/Desktop/diffusion-point-cloud-main/layer_data/test/input_data_test.npy")
            else:
                self.layer_data = np.load(f"/home/robust/Desktop/diffusion-point-cloud-main/layer_data/test/{self.layer_name}_test.npy")

        self.mean = np.zeros((self.layer_data.shape[0], self.layer_data.shape[2]))
        self.std = np.zeros(self.layer_data.shape[0])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data = self.layer_data[idx]
        label = self.labels[idx]
        
        if self.standardize:
            mean = np.mean(data, axis=0)
            data -= mean
            std = np.std(data)
            data /= std
            
            self.mean[idx] = mean
            self.std[idx] = std
        
        if self.zero_pad == 256:
            dim_gap = 256 - data.shape[-1]
            data = np.concatenate((data, np.zeros((data.shape[0], dim_gap), dtype=np.float32)), axis=1)
        else: data = data
        
        if self.transform:
            data = self.transform(data)
        
        return data, label
    
    
    
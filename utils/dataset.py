import os
import random
from copy import copy
import torch
from torch.utils.data import Dataset
import numpy as np
import h5py
from tqdm.auto import tqdm
import glob

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
    
class ModelNet40(Dataset):
    def __init__(self, num_points, partition='train'):
        if partition in ['train','val']:
            self.data, self.label = self.__load_data("train")
        else:
            self.data, self.label = self.__load_data(partition)
            
        mask = np.random.permutation(len(self.label))        
        if partition=="val": 
            mask = mask[len(mask)*9//10:len(mask)]
            self.data, self.label = self.data[mask], self.label[mask]
        elif partition=="train": 
            # mask = mask[0:len(mask)*4//5]
            mask = mask[:]
            self.data, self.label = self.data[mask], self.label[mask]
            
        self.num_points = num_points
        self.partition = partition        

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        if self.partition in ['train', 'val']:
            pointcloud = self.__translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]
    
    def __load_data(self, partition):
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        DATA_DIR = os.path.join(BASE_DIR, '../data')
        all_data = []
        all_label = []
        for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5'%partition)):
            f = h5py.File(h5_name)
            data = f['data'][:].astype('float32')
            label = f['label'][:].astype('int64')
            f.close()
            all_data.append(data)
            all_label.append(label)
        all_data = np.concatenate(all_data, axis=0)
        all_label = np.concatenate(all_label, axis=0)
        return all_data, all_label
        
    def __translate_pointcloud(self, pointcloud):
        xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
        xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
        
        translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
        return translated_pointcloud


    def __jitter_pointcloud(self, pointcloud, sigma=0.01, clip=0.02):
        N, C = pointcloud.shape
        pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
        return pointcloud
    
    
class ModelNet40C(Dataset):
    def __init__(self, split="test", test_data_path="../data/modelnet40_c",corruption=None,severity=None):
        assert split == 'test'
        self.split = split
        self.data_path = {
            "test":  test_data_path
        }[self.split]
        assert corruption in ["background", "cutout", "density", "density_inc", "distortion", "distortion_rbf", "distortion_rbf_inv",
                              "gaussian", "impulse", "lidar", "occlusion", "rotation", "shear", "uniform", "upsampling"]
        assert severity in [1,2,3,4,5]
        self.corruption = corruption
        self.severity = severity

        self.data, self.label = self.__load_data(self.data_path, self.corruption, self.severity)
        # self.num_points = num_points
        self.partition =  'test'

    def __getitem__(self, item):
        pointcloud = self.data[item]#[:self.num_points]
        label = self.label[item]
        # return {'pc': pointcloud, 'label': label.item()}
        return pointcloud, label.item()

    def __len__(self):
        return self.data.shape[0]
    
    def __load_data(self, data_path,corruption,severity):

        DATA_DIR = os.path.join(data_path, 'data_' + corruption + '_' +str(severity) + '.npy')
        # if corruption in ['occlusion']:
        #     LABEL_DIR = os.path.join(data_path, 'label_occlusion.npy')
        LABEL_DIR = os.path.join(data_path, 'label.npy')
        all_data = np.load(DATA_DIR)
        all_label = np.load(LABEL_DIR)
        return all_data, all_label


def normalize_points_np(points):
    """points: [K, 3]"""
    points = points - np.mean(points, axis=0)[None, :]  # center
    dist = np.max(np.sqrt(np.sum(points ** 2, axis=1)), 0)
    points = points / dist  # scale
    assert np.sum(np.isnan(points)) == 0
    return points


class ModelNet40Attack(Dataset):
    """Modelnet40 dataset for target attack evaluation.
    We return an additional target label for an example.
    """

    def __init__(self, data_root, num_points, normalize=True):
        self.data, self.label, self.target = self.load_data(data_root)
        self.num_points = num_points
        self.normalize = normalize
    
    def __getitem__(self, item):
        """Returns: point cloud as [N, 3], its label as a scalar
            and its target label for attack as a scalar.
        """
        pc = self.data[item][:self.num_points, :3]
        label = self.label[item]
        target = self.target[item]

        if self.normalize:
            pc = normalize_points_np(pc)
        return pc, label, target

    def load_data(self, data_path):
        DATA_DIR = os.path.join(data_path, "attack_data.npz")
        npz = np.load(DATA_DIR, allow_pickle=True)
        return npz['test_pc'], npz['test_label'], npz['target_label']

    def __len__(self):
        return self.data.shape[0]
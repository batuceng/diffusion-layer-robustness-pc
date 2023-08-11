import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sklearn.metrics as metrics
import numpy as np
from time import time

from dataset.modelnet40 import ModelNet40
from models import PCT_cls, DGCNN_cls, CurveNet_cls, PointMLP_cls, PointNet_cls, PointNet2_cls


def test(model, test_loader):

    model = model.eval()
    test_acc = 0.0
    count = 0.0
    test_true = []
    test_pred = []
    for batch in test_loader:
        data, label = batch['pointcloud'], batch['cate']
        data, label = data.cuda(), label.cuda().squeeze()
        data = data.permute(0, 2, 1)
        batch_size = data.size()[0]
        
        denoiser = Identity()
        logits = model.forward_denoised(data, denoiser=denoiser)
        # logits = model.module(data)
        # logits = model(data)


        if len(logits) == 2:
            logits = logits[0]

        preds = logits.max(dim=1)[1]
        test_true.append(label.cpu().numpy())
        test_pred.append(preds.detach().cpu().numpy())
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    test_acc = metrics.accuracy_score(test_true, test_pred)
    outstr = 'Test :: test acc: %.6f'%(test_acc)
    print(outstr)


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()
        pass
    
    def forward(self, data, *args, **kwargs):
        return data


dataset = ModelNet40(num_points=1024,partition='test', scale_mode='none')

test_loader = DataLoader(dataset, batch_size=16, shuffle=False, drop_last=False)

# modelName = 'PointNet'

model_dict = {
    'CurveNet': CurveNet_cls, #0.938412
    "PCT":      PCT_cls, #0.931524
    "PointMLP": PointMLP_cls, #0.940032
    "DGCNN":    DGCNN_cls, #0.928687
    "PointNet2":PointNet2_cls, #0.911264
    "PointNet": PointNet_cls, #0.896677
}

for modelName in model_dict.keys():
    print(modelName)
    model = model_dict[modelName]()

    model.cuda()
    model.eval()

    test(model, test_loader)



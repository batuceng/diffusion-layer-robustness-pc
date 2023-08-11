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

modelName = 'PointNet2'

if modelName == 'CurveNet':
    model = CurveNet_cls() #0.938412

elif modelName == "PCT":
    model = PCT_cls() #0.9333144

elif modelName == "PointMLP":
    model = PointMLP_cls() #0.940843

elif modelName == "DGCNN": #0.928687
    model = DGCNN_cls()

elif modelName == "PointNet2":#0.915316
    model = PointNet2_cls()

elif modelName == "PointNet":#0.896677
    model = PointNet_cls()





model.cuda()
model.eval()

test(model, test_loader)



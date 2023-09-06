import torch
import torch.nn as nn
from .attack import Attack
import numpy as np

class PointAdd(Attack):
    
    def __init__(self, model, device=None, steps=7, eps=0.05, lr=0.01, p=np.inf, num_points=200, seed=3):
        super().__init__('PointAdd', model, device, seed)
        # Attack Vals
        self.num_points = num_points
        self.steps = steps
        self.eps = eps
        self.lr = lr
        self.p = np.inf if p == "inf" else 2 if p == "2" else 1
        self.supported_mode = ['default']
        # self.supported_mode = ['default', 'targeted']
        self.device = device
        self.targeted = False
    
    def attack(self, data, labels):
        # data: (B, N, d); labels: (B, 1)
        data = data.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        
        indices = torch.Tensor(np.random.choice(data.shape[1], self.num_points, replace=False)).long()
        adv_data_og = data.clone()[:,indices,:]
        adv_data = adv_data_og+(torch.rand_like(adv_data_og)*self.eps*2-self.eps)
        adv_data=adv_data+(torch.rand_like(adv_data)*0.05*2-0.05)
        
        loss = nn.CrossEntropyLoss()
        batchsize = len(data)
        
        for _ in range(self.steps):
            adv_data.requires_grad = True
            
            adv_data_batch = torch.cat([data, adv_data], dim=-2)
            
            self.model.zero_grad()
            outputs = self.get_logits(adv_data_batch)
            # print(adv_data.requires_grad, outputs.requires_grad)
            
            # Calculate loss
            if self.targeted:
                # cost = -loss(outputs, target_labels)
                raise NotImplementedError
            else:
                cost = loss(outputs, labels.squeeze())
            
            # Calculate Gradients
            grad = torch.autograd.grad(cost, adv_data,
                                       retain_graph=False, create_graph=False)[0]

            with torch.no_grad():
                if self.p==np.inf:
                    adv_data = adv_data + self.lr * grad.sign()
                    delta = adv_data-adv_data_og
                    delta = torch.clamp(delta, -self.eps,self.eps)
                else:
                    adv_data = adv_data + self.lr * grad
                    delta = adv_data-adv_data_og
                    normVal = torch.norm(delta.view(batchsize, -1), self.p, 1).view(batchsize, 1, 1)
                    mask = normVal<=self.eps
                    scaling = self.eps / normVal
                    scaling[mask] = 1
                    delta = delta*scaling
                # print(delta)
                adv_data = (adv_data_og+delta).detach_()
        # print('finishing one batch...')
        return adv_data_batch
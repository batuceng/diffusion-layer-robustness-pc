import torch
import torch.nn as nn
from .attack import Attack


class PointDrop(Attack):
    
    def __init__(self, model, device, num_points=200, steps=10, seed=3):
        super().__init__("PointDrop", model, device, seed)
        # Attack Vals
        self.num_points = num_points
        self.steps = steps
        self.supported_mode = ['default']
        # self.supported_mode = ['default', 'targeted']
        self.device = device
        self.targeted = False
        
    
    def attack(self, data, labels):
        # data: (B, N, d); labels: (B, 1)
        data = data.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        
        loss = nn.CrossEntropyLoss()
        alpha = self.num_points // self.steps
        
        adv_data = data.clone().detach()
        
        for _ in range(self.steps):
            adv_data.requires_grad = True
            self.model.zero_grad()
            outputs = self.get_logits(adv_data)
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

            # Saliency drop
            with torch.no_grad():
                sphere_core,_ = torch.median(adv_data, dim=1, keepdim=True)
                sphere_r = torch.sqrt(torch.sum(torch.square(adv_data - sphere_core), dim=2))
                sphere_axis = adv_data - sphere_core

                sphere_map = - torch.mul(torch.sum(torch.mul(grad, sphere_axis), dim=2), torch.pow(sphere_r, 2))
                # print(adv_data.shape[1])
                _,indice = torch.topk(sphere_map, k=adv_data.shape[1] - alpha, dim=-1, largest=False)
                tmp = torch.zeros((adv_data.shape[0], adv_data.shape[1] - alpha, 3))
                for i in range(adv_data.shape[0]):
                    tmp[i] = adv_data[i][indice[i],:]
                adv_data = tmp.clone()
        return adv_data
    
    
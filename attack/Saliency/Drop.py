"""Implementation of saliency map based attack,
    dropping points with high importance scores.
Based on ICCV'19: PointCloud Saliency Maps.
"""

import numpy as np

import torch
import torch.nn.functional as F


class SaliencyDrop:
    """Class for saliency map based points dropping.
    """

    def __init__(self, model, num_drop, alpha=1, k=5):

        self.model = model.cuda()
        self.model.eval()

        self.num_drop = num_drop
        self.alpha = alpha
        self.k = k

    def get_gradient(self, data, target):
        """Calculate gradient on data.

        Args:
            data (torch.FloatTensor): victim data, [B, 3, K]
            target (torch.LongTensor): target output, [B]
        """
        input_data = data.clone().detach().float().cuda()
        input_data.requires_grad_()
        target = target.long().cuda()

        # forward pass
        logits = self.model(input_data)
        if isinstance(logits, tuple):  # PoitnNet
            logits = logits[0]
        loss = F.cross_entropy(logits, target)
        loss.backward()
        with torch.no_grad():
            grad = input_data.grad.detach()  # [B, 3, K]
            # success num
            pred = torch.argmax(logits, dim=-1)  # [B]
            num = (pred == target).sum().detach().cpu().item()
        return grad, num

    def attack(self, data, target, *args, **kwargs):
        """Attack on given data to target.

        Args:
            data (torch.FloatTensor): victim data, [B, num_points, 3]
            target (torch.LongTensor): target output, [B]
        """
        B, K_ = data.shape[:2]
        data = data.float().cuda().detach()
        data = data.transpose(1, 2).contiguous()  # [B, 3, K]
        target = target.long().cuda().detach()  # [B]
        num_rounds = int(np.ceil(float(self.num_drop) / float(self.k)))
        for i in range(num_rounds):
            K = data.shape[2]

            # number of points to drop in this round
            k = min(self.k, self.num_drop - i * self.k)

            # calculate gradient of loss
            grad, success_num = \
                self.get_gradient(data, target)  # [B, 3, K]
            if i % (num_rounds // 5) == 0:
                print('Iteration {}/{}, success {}/{}\n'
                      'Point num: {}/{}'.
                      format(i, num_rounds, success_num, B,
                             K, K_))

            with torch.no_grad():
                # compute center point
                center = torch.median(data, dim=-1)[0].\
                    clone().detach()  # [B, 3]

                # compute r_i as l2 distance
                r = torch.sum((data - center[:, :, None]) ** 2,
                              dim=1) ** 0.5  # [B, K]

                # compute saliency score
                saliency = -1. * (r ** self.alpha) * \
                    torch.sum((data - center[:, :, None]) * grad,
                              dim=1)  # [B, K]

                # drop points with highest saliency scores w.r.t. gt labels
                # note that this is for untarget attack!
                _, idx = (-saliency).topk(k=K - k, dim=-1)  # [B, K - k]
                data = torch.stack([
                    data[j, :, idx[j]] for j in range(B)
                ], dim=0)  # [B, 3, K - k]

        # end of dropping
        # now data is [B, 3, K - self.num_drop]
        with torch.no_grad():
            logits = self.model(data)
            if isinstance(logits, tuple):  # PointNet
                logits = logits[0]
            pred = torch.argmax(logits, dim=-1)
            success_num = (pred == target).\
                sum().detach().cpu().item()
        print('Final success: {}/{}, point num: {}/{}'.
              format(success_num, B, data.shape[2], K_))
        return data.transpose(1, 2).contiguous().detach(), \
            success_num


class Drop_PointDP:
    def __init__(self, model, step= 10, num_points=200):
        self.model = model
        self.step = step
        self.num_points= num_points
    
    def attack(self, data, target_label=None, label=None, scale=None, shift=None):
        self.model.eval()
        data = data.contiguous()
        data_og = data.cuda()
        data = data_og.clone()
        scale = scale.cuda()
        shift = shift.cuda()
        
        
        adv_data=data.clone().cuda()
        adv_data.detach()
        adv_data_batch = {}
        alpha = self.num_points // self.step

        for i in range(self.step):
            adv_data.requires_grad=True
            adv_data_batch['pc'] = adv_data
            adv_data_batch['label'] = label.cuda()
            self.model.zero_grad()
            out = self.model(adv_data_batch['pc'].transpose(1, 2).contiguous())
            loss = F.cross_entropy(out, label, reduction='mean')
            loss.backward()

            with torch.no_grad():
                sphere_core,_ = torch.median(adv_data, dim=1, keepdim=True)
                sphere_r = torch.sqrt(torch.sum(torch.square(adv_data - sphere_core), dim=2))
                sphere_axis = adv_data - sphere_core

                sphere_map = - torch.mul(torch.sum(torch.mul(adv_data.grad, sphere_axis), dim=2), torch.pow(sphere_r, 2))
                # print(adv_data.shape[1])
                _,indice = torch.topk(sphere_map, k=adv_data.shape[1] - alpha, dim=-1, largest=False)
                tmp = torch.zeros((adv_data.shape[0], adv_data.shape[1] - alpha, 3))
                for i in range(adv_data.shape[0]):
                    tmp[i] = adv_data[i][indice[i],:]
                adv_data = tmp.clone()
                
        return adv_data.type(torch.cuda.FloatTensor), 0
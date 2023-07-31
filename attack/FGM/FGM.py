"""Implementation of gradient based attack methods, FGM, I-FGM, MI-FGM, PGD, etc.
Related paper: CVPR'20 GvG-P,
    https://openaccess.thecvf.com/content_CVPR_2020/papers/Dong_Self-Robust_3D_Point_Recognition_via_Gather-Vector_Guidance_CVPR_2020_paper.pdf
"""
#%%

import torch
import numpy as np
import torch.nn.functional as F

class Identity_Attack:
    """Class for FGM attack.
    """

    def __init__(self):
        pass
    
    def attack(self, data, target, *args, **kwargs):
        success_num = 0
        return data, success_num
    
    
class PGD:
    def __init__(self, model, step=7, eps=0.05, alpha=0.01, p=np.inf):
        self.model = model
        self.step = step
        self.eps = eps
        self.alpha = alpha
        self.p = p
    
    def attack(self, data, target_label=None, label=None, scale=None, shift=None):
        self.model.eval()
        # keep data_og as original
        data = data.transpose(1, 2).contiguous()
        data_og = data.cuda()
        data = data_og.clone()
        scale = scale.cuda()
        shift = shift.cuda()
        eps = (self.eps / scale).expand(-1,data.shape[1],data.shape[2])
        alpha = (self.alpha / scale).expand(-1,data.shape[1],data.shape[2])
        
        adv_data=data.clone().cuda()
        # initialize random perturbation, use 0.05 by default 
        # adv_data=adv_data+(torch.rand_like(adv_data)*0.05*2-0.05) 
        adv_data.detach()
        adv_data_batch = {}
        batchsize = data.shape[0]
        
        for i in range(self.step):
            if i % 10 == 0:
                print("Iter", i, "/", self.step)
            adv_data.requires_grad=True
            adv_data_batch['pc'] = adv_data
            adv_data_batch['label'] = label.cuda()
            self.model.zero_grad()
            
            out = self.model(adv_data_batch['pc'])
            loss = F.cross_entropy(out, adv_data_batch['label'], reduction='mean')
            loss.backward()
            with torch.no_grad():
                if self.p==np.inf:
                    adv_data = adv_data + alpha * adv_data.grad.sign()
                    delta = adv_data-data_og
                    delta = torch.clamp(delta, -eps,eps)
                else:
                    adv_data = adv_data + alpha * adv_data.grad
                    delta = adv_data-data_og
                    normVal = torch.norm(delta.view(batchsize, -1), self.p, 1).view(batchsize, 1, 1)
                    mask = normVal<=eps[:,0,0].view(batchsize, 1, 1)
                    scaling = eps[:,0,0].view(batchsize, 1, 1) / normVal
                    scaling[mask] = 1
                    delta = delta*scaling
                adv_data = (data+delta).detach_()
                
        return adv_data.type(torch.cuda.FloatTensor).transpose(1, 2).contiguous(), 0

#%%

        
if __name__ == "__main__":
    import os
    os.chdir("/home/robust/diffusion-point-cloud")

    from models.dgcnn import PointNet, DGCNN_cls
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from dataset.modelnet40 import ModelNet40
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from FGM_attack import PGD as pgd_ifdefense
    from attack.util.adv_utils import CrossEntropyAdvLoss, LogitsAdvLoss
    from attack.util.clip_utils import ClipPointsL2
    from attack.util.dist_utils import L2Dist
    from attack.CW.Perturb import CWPerturb

    class AttrDict(dict):
        def __init__(self, *args, **kwargs):
            super(AttrDict, self).__init__(*args, **kwargs)
            self.__dict__ = self
            
    num_points = 1024
    batch_size = 32
    cuda = True
    scale_mode = "none"
    test_loader = DataLoader(ModelNet40(partition='test', num_points=num_points, scale_mode=scale_mode),
                            batch_size=batch_size, shuffle=False, drop_last=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    budget = 0.05
    num_iter = 500
    attack_lr = 1e-2
    p = 2.
    adv_func = "logits"
    kappa = 0
    
    model_path = "pretrained/model.1024.t7"
    model = DGCNN_cls(args=AttrDict({"k":20, "dropout":0.5, "emb_dims":1024}))
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(model_path))
    model = model.eval()    
    
    attack_type = "ifdefense"
    
    if attack_type == "ifdefense":
        delta = budget
        budget = budget * \
            np.sqrt(num_points * 3)  # \delta * \sqrt(N * d)
        num_iter = int(num_iter)
        step_size = budget / float(num_iter)
        # which adv_func to use?
        if adv_func == 'logits':
            adv_func = LogitsAdvLoss(kappa=kappa)
        else:
            adv_func = CrossEntropyAdvLoss()
        clip_func = ClipPointsL2(budget=budget)
        dist_func = L2Dist()
        binary_step = 10
        # attacker = pgd_ifdefense(model, adv_func, clip_func, budget, step_size,
        #                          num_iter, "l2")
        
        attacker = CWPerturb(model, adv_func, dist_func,
                         attack_lr=attack_lr,
                         init_weight=10., max_weight=80.,
                         binary_step=binary_step,
                         num_iter=num_iter)
    else:
        attacker = PGD(model, num_iter, budget, attack_lr, p)

    for i, batch in enumerate(test_loader):
        print("Batch : ", i+1, "/", len(test_loader))
        
        # data, label, target_label = data.to(device), label.to(device).squeeze(), target_label.to(device).squeeze()
        
        data = batch['pointcloud'].to(device)
        shift = batch['shift']
        scale = batch['scale']
        label = batch["cate"].to(device).view(-1)
        
        random_labels = torch.randint(0, 40, label.size(), device=device)
        random_labels[random_labels == label] += 1
        target_label = torch.remainder(random_labels, 40)

        best_pc, _ = attacker.attack(data, target_label, label=label, scale=scale, shift=shift)
        break
    
# %%
    from dataset.modelnet40 import LABELS


    fig = plt.figure()

    ax = fig.add_subplot(projection="3d")

    i= 9

    rescaled_pc = (best_pc[i] * scale[i].numpy()) + shift[i].numpy()

    rescaled_data = (data[i].cpu().numpy() * scale[i].numpy()) + shift[i].numpy()

    print(scale[i])

    print(shift[i])
    # x, y, z = best_pc[i, :, 0].cpu().numpy(), best_pc[i, :, 1].cpu().numpy(), best_pc[i, :, 2].cpu().numpy()
    # ax.scatter3D(x, y, z)

    # x, y, z = data[i, :, 0].cpu().numpy(), data[i, :, 1].cpu().numpy(), data[i, :, 2].cpu().numpy()
    # ax.scatter3D(x, y, z)

    x, y, z = rescaled_pc[:, 0], rescaled_pc[:, 1], rescaled_pc[:, 2]
    ax.scatter3D(x, y, z)

    x, y, z = rescaled_data[:, 0], rescaled_data[:, 1], rescaled_data[:, 2]
    ax.scatter3D(x, y, z)

    print(LABELS[label[i].cpu().numpy()])
    plt.show()


# %%

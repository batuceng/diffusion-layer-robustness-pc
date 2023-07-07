"""Implementation of gradient based attack methods, FGM, I-FGM, MI-FGM, PGD, etc.
Related paper: CVPR'20 GvG-P,
    https://openaccess.thecvf.com/content_CVPR_2020/papers/Dong_Self-Robust_3D_Point_Recognition_via_Gather-Vector_Guidance_CVPR_2020_paper.pdf
"""

import torch
import numpy as np
import torch.nn.functional as F

class FGM:
    """Class for FGM attack.
    """

    def __init__(self, model, adv_func, budget,
                 dist_metric='l2'):
        """FGM attack.

        Args:
            model (torch.nn.Module): victim model
            adv_func (function): adversarial loss function
            budget (float): \epsilon ball for FGM attack
            dist_metric (str, optional): type of constraint. Defaults to 'l2'.
        """

        self.model = model.cuda()
        self.model.eval()

        self.adv_func = adv_func
        self.budget = budget
        self.dist_metric = dist_metric.lower()

    def get_norm(self, x):
        """Calculate the norm of a given data x.

        Args:
            x (torch.FloatTensor): [B, 3, K]
        """
        # use global l2 norm here!
        norm = torch.sum(x ** 2, dim=[1, 2]) ** 0.5
        return norm

    def get_gradient(self, data, target, normalize=True):
        """Generate one step gradient.

        Args:
            data (torch.FloatTensor): batch pc, [B, 3, K]
            target (torch.LongTensor): target label, [B]
            normalize (bool, optional): whether l2 normalize grad. Defaults to True.
        """
        data = data.float().cuda()
        data.requires_grad_()
        target = target.long().cuda()

        # forward pass
        logits = self.model(data)
        if isinstance(logits, tuple):
            logits = logits[0]  # [B, class]
        pred = torch.argmax(logits, dim=-1)  # [B]

        # backward pass
        loss = self.adv_func(logits, target).mean()
        loss.backward()
        with torch.no_grad():
            grad = data.grad.detach()  # [B, 3, K]
            if normalize:
                norm = self.get_norm(grad)
                grad = grad / (norm[:, None, None] + 1e-9)
        return grad, pred

    def attack(self, data, target, *args, **kwargs):
        """One step FGM attack.

        Args:
            data (torch.FloatTensor): batch pc, [B, K, 3]
            target (torch.LongTensor): target label, [B]
        """
        data = data.float().cuda().detach()
        pc = data.clone().detach().transpose(1, 2).contiguous()
        target = target.long().cuda()
        
        # gradient
        normalized_grad, _ = self.get_gradient(pc, target)  # [B, 3, K]
        perturbation = normalized_grad * self.budget
        with torch.no_grad():
            perturbation = perturbation.transpose(1, 2).contiguous()
            data = data - perturbation  # no need to clip

            # test attack performance
            logits = self.model(data.transpose(1, 2).contiguous())
            if isinstance(logits, tuple):
                logits = logits[0]
            pred = torch.argmax(logits, dim=-1)
            success_num = (pred == target).sum().item()

        print('Successfully attack {}/{}'.format(success_num, data.shape[0]))
        torch.cuda.empty_cache()
        return data.detach(), success_num


class IFGM(FGM):
    """Class for I-FGM attack.
    """

    def __init__(self, model, adv_func, clip_func,
                 budget, step_size, num_iter,
                 dist_metric='l2'):
        """Iterative FGM attack.

        Args:
            model (torch.nn.Module): victim model
            adv_func (function): adversarial loss function
            clip_func (function): clipping method
            budget (float): \epsilon ball for IFGM attack
            step_size (float): attack step length
            num_iter (int): number of iteration
            dist_metric (str, optional): type of constraint. Defaults to 'l2'.
        """
        super(IFGM, self).__init__(model, adv_func, budget, dist_metric)
        self.clip_func = clip_func
        self.step_size = step_size
        self.num_iter = num_iter

    def attack(self, data, target, *args, **kwargs):
        """Iterative FGM attack.

        Args:
            data (torch.FloatTensor): batch pc, [B, K, 3]
            target (torch.LongTensor): target label
        """
        B, K = data.shape[:2]
        data = data.float().cuda().detach()
        pc = data.clone().detach().transpose(1, 2).contiguous()
        pc = pc + torch.randn((B, 3, K)).cuda() * 1e-7
        ori_pc = pc.clone().detach()
        target = target.long().cuda()

        # start iteration
        for iteration in range(self.num_iter):
            # gradient
            normalized_grad, pred = self.get_gradient(pc, target)
            success_num = (pred == target).sum().item()
            if iteration % (self.num_iter // 5) == 0:
                print('iter {}/{}, success: {}/{}'.
                      format(iteration, self.num_iter,
                             success_num, B))
                torch.cuda.empty_cache()
            perturbation = self.step_size * normalized_grad

            # add perturbation and clip
            with torch.no_grad():
                pc = pc - perturbation
                pc = self.clip_func(pc, ori_pc)

        # end of iteration
        with torch.no_grad():
            logits = self.model(pc)
            if isinstance(logits, tuple):
                logits = logits[0]
            pred = torch.argmax(logits, dim=-1)
            success_num = (pred == target).sum().item()
        print('Final success: {}/{}'.format(success_num, B))
        return pc.transpose(1, 2).contiguous().detach(), \
            success_num


class MIFGM(FGM):
    """Class for MI-FGM attack.
    """

    def __init__(self, model, adv_func, clip_func,
                 budget, step_size, num_iter, mu=1.,
                 dist_metric='l2'):
        """Momentum enhanced iterative FGM attack.

        Args:
            model (torch.nn.Module): victim model
            adv_func (function): adversarial loss function
            clip_func (function): clipping method
            budget (float): \epsilon ball for IFGM attack
            step_size (float): attack step length
            num_iter (int): number of iteration
            mu (float): momentum factor
            dist_metric (str, optional): type of constraint. Defaults to 'l2'.
        """
        super(MIFGM, self).__init__(model, adv_func,
                                    budget, dist_metric)
        self.clip_func = clip_func
        self.step_size = step_size
        self.num_iter = num_iter
        self.mu = mu

    def attack(self, data, target, *args, **kwargs):
        """Momentum enhanced iterative FGM attack.

        Args:
            data (torch.FloatTensor): batch pc, [B, K, 3]
            target (torch.LongTensor): target label
        """
        B, K = data.shape[:2]
        data = data.float().cuda().detach()
        pc = data.clone().detach().transpose(1, 2).contiguous()
        pc = pc + torch.randn((B, 3, K)).cuda() * 1e-7
        ori_pc = pc.clone().detach()
        target = target.long().cuda()
        momentum_g = torch.tensor(0.).cuda()

        # start iteration
        for iteration in range(self.num_iter):
            # gradient
            grad, pred = self.get_gradient(pc, target, normalize=False)
            success_num = (pred == target).sum().item()
            if iteration % (self.num_iter // 5) == 0:
                print('iter {}/{}, success: {}/{}'.
                      format(iteration, self.num_iter,
                             success_num, B))
                torch.cuda.empty_cache()

            # grad is [B, 3, K]
            # normalized by l1 norm
            grad_l1_norm = torch.sum(torch.abs(grad), dim=[1, 2])  # [B]
            normalized_grad = grad / (grad_l1_norm[:, None, None] + 1e-9)
            momentum_g = self.mu * momentum_g + normalized_grad
            g_norm = self.get_norm(momentum_g)
            normalized_g = momentum_g / (g_norm[:, None, None] + 1e-9)
            perturbation = self.step_size * normalized_g

            # add perturbation and clip
            with torch.no_grad():
                print(self.get_norm(perturbation))
                pc = pc - perturbation
                pc = self.clip_func(pc, ori_pc)

        # end of iteration
        with torch.no_grad():
            logits = self.model(pc)
            if isinstance(logits, tuple):
                logits = logits[0]
            pred = torch.argmax(logits, dim=-1)
            success_num = (pred == target).sum().item()
        print('Final success: {}/{}'.format(success_num, B))
        return pc.transpose(1, 2).contiguous().detach(), \
            success_num


class PGD(IFGM):
    """Class for PGD attack.
    """

    def __init__(self, model, adv_func, clip_func,
                 budget, step_size, num_iter,
                 dist_metric='l2'):
        """PGD attack.

        Args:
            model (torch.nn.Module): victim model
            adv_func (function): adversarial loss function
            clip_func (function): clipping method
            budget (float): \epsilon ball for IFGM attack
            step_size (float): attack step length
            num_iter (int): number of iteration
            dist_metric (str, optional): type of constraint. Defaults to 'l2'.
        """
        super(PGD, self).__init__(model, adv_func, clip_func,
                                  budget, step_size, num_iter,
                                  dist_metric)

    def attack(self, data, target, *args, **kwargs):
        """PGD attack.

        Args:
            data (torch.FloatTensor): batch pc, [B, K, 3]
            target (torch.LongTensor): target label
        """
        # the only difference between IFGM and PGD is
        # the initialization of noise
        epsilon = self.budget / \
            ((data.shape[1] * data.shape[2]) ** 0.5)
        init_perturbation = \
            torch.empty_like(data).uniform_(-epsilon, epsilon)
        with torch.no_grad():
            init_data = data + init_perturbation
        return super(PGD, self).attack(init_data, target)


class Identity_Attack:
    """Class for FGM attack.
    """

    def __init__(self):
        pass
    
    def attack(self, data, target, *args, **kwargs):
        success_num = 0
        return data, success_num
    
    
class PGD_PointDP:
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
        adv_data=adv_data+(torch.rand_like(adv_data)*0.05*2-0.05) 
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
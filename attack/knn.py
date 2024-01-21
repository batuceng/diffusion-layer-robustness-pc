import torch
import torch.nn as nn
import torch.optim as optim
from .attack import Attack


class KNN(Attack):

    def __init__(self, model, device=None, kappa=0, steps=200, lr=0.01, seed=3):
        super().__init__('CW', model, device, seed)
        self.kappa = kappa
        self.steps = steps
        self.lr = lr
        self.supported_mode = ['default']
        # self.supported_mode = ['default1', 'targeted']
        self.targeted = False
    
    def normalize(self, data):
        pc_max, _ = data.max(dim=1, keepdim=True) # (1, 3)    
        pc_min, _ = data.min(dim=1, keepdim=True) # (1, 3)
        shift = ((pc_min + pc_max) / 2).view(-1, 1, 3)
        scale = (pc_max - pc_min).max().reshape(-1, 1, 1) / 2
        return (data-shift)/scale, shift, scale
    
    def denormalize(self, data, shift, scale):
        return (data * scale) + shift

    def attack(self, data, labels):
        # data: (B, N, d); labels: (B, 1)
        data = data.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device).squeeze(-1)

        # data, shift, scale = self.normalize(data)
        
        # w = torch.zeros_like(data).detach() # Requires 2x times
        w = self.inverse_tanh_space(data).detach()
        w.requires_grad = True

        best_adv_images = data.clone().detach()
        best_L2 = 1e10*torch.ones((len(data))).to(self.device)
        prev_cost = 1e10
        dim = len(data.shape)

        dist_func = ChamferkNNDist(chamfer_method='adv2ori',
                               knn_k=5, knn_alpha=1.05,
                               chamfer_weight=10., knn_weight=6.)

        optimizer = optim.Adam([w], lr=self.lr)

        for step in range(self.steps):
            # Get adversarial images
            adv_images = self.tanh_space(w)
            
            current_dis = dist_func(adv_images, data)
            dist_loss = current_dis.mean() * 1024
            
            # Calculate loss
            outputs = self.get_logits(adv_images)
            if self.targeted:
                # f_loss = self.f(outputs, target_labels).sum()
                raise NotImplementedError
            else:
                f_loss = self.f(outputs, labels).mean()

            cost = dist_loss + f_loss
            
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            # Update adversarial images
            pre = torch.argmax(outputs.detach(), 1)
            if self.targeted:
                # We want to let pre == target_labels in a targeted attack
                # condition = (pre == target_labels).float()
                raise NotImplementedError
            else:
                # If the attack is not targeted we simply make these two values unequal
                condition = (pre != labels).float()

            # Filter out images that get either correct predictions or non-decreasing loss,
            # i.e., only images that are both misclassified and loss-decreasing are left
            mask = condition*(best_L2 > dist_loss.detach())
            best_L2 = mask*dist_loss.detach() + (1-mask)*best_L2

            mask = mask.view([-1]+[1]*(dim-1))
            best_adv_images = mask*adv_images.detach() + (1-mask)*best_adv_images

            # Early stop when loss does not converge.
            # max(.,1) To prevent MODULO BY ZERO error in the next step.
            if step % max(self.steps//10, 1) == 0:
                if cost.item() > prev_cost:
                    return best_adv_images
                prev_cost = cost.item()

        return best_adv_images
        # return self.denormalize(best_adv_images, shift, scale)

    def tanh_space(self, x):
        return torch.tanh(x)
        # return 1/2*(torch.tanh(x) + 1)

    def inverse_tanh_space(self, x):
        # torch.atanh is only for torch >= 1.7.0
        # atanh is defined in the range -1 to 1
        # return self.atanh(torch.clamp(x*2-1, min=-1, max=1))
        return self.atanh(x)

    def atanh(self, x):
        return 0.5*torch.log((1+x)/(1-x))

    # f-function in the paper
    def f(self, outputs, labels):
        one_hot_labels = torch.eye(outputs.shape[1]).to(self.device)[labels]

        # find the max logit other than the target class
        other = torch.max((1-one_hot_labels)*outputs - one_hot_labels*10000, dim=1)[0]
        # get the target class's logit
        real = torch.masked_select(outputs, one_hot_labels.bool())

        if self.targeted:
            # return torch.clamp((other-real), min=-self.kappa)
            raise NotImplementedError
        else:
            return torch.clamp((real-other), min=-self.kappa)


class ChamferDist(nn.Module):

    def __init__(self, method='adv2ori'):
        super(ChamferDist, self).__init__()

        self.method = method

    def batch_pairwise_dist(self, x, y):
        bs, num_points_x, points_dim = x.size()
        _, num_points_y, _ = y.size()
        xx = torch.bmm(x, x.transpose(2, 1))  # [B, K, K]
        yy = torch.bmm(y, y.transpose(2, 1))
        zz = torch.bmm(x, y.transpose(2, 1))
        diag_ind_x = torch.arange(0, num_points_x)
        diag_ind_y = torch.arange(0, num_points_y)
        rx = xx[:, diag_ind_x, diag_ind_x].unsqueeze(
            1).expand_as(zz.transpose(2, 1))
        ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz)
        P = (rx.transpose(2, 1) + ry - 2 * zz)
        return P

    def chamfer(self,preds, gts):
        P = self.batch_pairwise_dist(gts, preds)  # [B, N2, N1]
        mins, _ = torch.min(P, 1)  # [B, N1], find preds' nearest points in gts
        loss1 = torch.mean(mins, dim=1)  # [B]
        mins, _ = torch.min(P, 2)  # [B, N2], find gts' nearest points in preds
        loss2 = torch.mean(mins, dim=1)  # [B]
        return loss1, loss2

    def forward(self, adv_pc, ori_pc,
                weights=None, batch_avg=True):
        B = adv_pc.shape[0]
        if weights is None:
            weights = torch.ones((B,))
        loss1, loss2 = self.chamfer(adv_pc, ori_pc)  # [B], adv2ori, ori2adv
        if self.method == 'adv2ori':
            loss = loss1
        elif self.method == 'ori2adv':
            loss = loss2
        else:
            loss = (loss1 + loss2) / 2.
        weights = weights.float().cuda()
        loss = loss.cuda() * weights
        if batch_avg:
            return loss.mean()
        return loss


class KNNDist(nn.Module):

    def __init__(self, k=5, alpha=1.05):
        super(KNNDist, self).__init__()

        self.k = k
        self.alpha = alpha

    def forward(self, pc, weights=None, batch_avg=True):
        # build kNN graph
        B, K = pc.shape[:2]
        pc = pc.transpose(2, 1)  # [B, 3, K]
        inner = -2. * torch.matmul(pc.transpose(2, 1), pc)  # [B, K, K]
        xx = torch.sum(pc ** 2, dim=1, keepdim=True)  # [B, 1, K]
        dist = xx + inner + xx.transpose(2, 1)  # [B, K, K], l2^2
        assert dist.min().item() >= -1e-6
        # the min is self so we take top (k + 1)
        neg_value, _ = (-dist).topk(k=self.k + 1, dim=-1)
        # [B, K, k + 1]
        value = -(neg_value[..., 1:])  # [B, K, k]
        value = torch.mean(value, dim=-1)  # d_p, [B, K]
        with torch.no_grad():
            mean = torch.mean(value, dim=-1)  # [B]
            std = torch.std(value, dim=-1)  # [B]
            # [B], penalty threshold for batch
            threshold = mean + self.alpha * std
            weight_mask = (value > threshold[:, None]).\
                float().detach()  # [B, K]
        loss = torch.mean(value * weight_mask, dim=1)  # [B]
        # accumulate loss
        if weights is None:
            weights = torch.ones((B,))
        weights = weights.float().cuda()
        loss = loss.cuda() * weights
        if batch_avg:
            return loss.mean()
        return loss


class ChamferkNNDist(nn.Module):

    def __init__(self, chamfer_method='adv2ori',
                 knn_k=5, knn_alpha=1.05,
                 chamfer_weight=5., knn_weight=3.):
        super(ChamferkNNDist, self).__init__()

        self.chamfer_dist = ChamferDist(method=chamfer_method)
        self.knn_dist = KNNDist(k=knn_k, alpha=knn_alpha)
        self.w1 = chamfer_weight
        self.w2 = knn_weight

    def forward(self, adv_pc, ori_pc,
                weights=None, batch_avg=True):
        chamfer_loss = self.chamfer_dist(
            adv_pc, ori_pc, weights=weights, batch_avg=batch_avg)
        knn_loss = self.knn_dist(
            adv_pc, weights=weights, batch_avg=batch_avg)
        loss = chamfer_loss * self.w1 + knn_loss * self.w2
        return loss
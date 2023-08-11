"""
Test layer denoisers on point cloud attacked data.
"""

from util.misc import seed_all, IOStream
from models.autoencoder import AutoEncoder

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import sklearn.metrics as metrics
from torch.utils.data import DataLoader
from dataset.modelnet40 import ModelNet40
from models import DGCNN_cls, PointNet2_cls
from models.denoiser import Identity, Layer_Denoiser
from attack import Identity_Attack, PGD, Drop_PointDP
from attack import CWKNN, CWAdd
from attack import CWPerturb
from attack import SaliencyDrop
from attack import CrossEntropyAdvLoss, LogitsAdvLoss
from attack import ClipPointsL2, L2Dist, ChamferkNNDist, ProjectInnerClipLinf

# Arguments
parser = argparse.ArgumentParser()
# parser.add_argument('--ckpt', type=str, default='logs/layer2/AE_2023_07_17__20_14_47/ckpt_1.611477_750000.pt')
# parser.add_argument('--t', type=int, default=0)
# parser.add_argument('--categories', type=str_list, default=['airplane'])
parser.add_argument('--save-dir', type=str, default='./attack_data')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--test-size', type=int, default=np.inf)

# Datasets and loaders
# parser.add_argument('--dataset-path', type=str, default='./data/shapenet.hdf5')
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--scale-mode', type=str, default='unit_sphere')
parser.add_argument('--num-points', type=int, default=1024,
                        help='num of points to use')

# Attack arguments
parser.add_argument('--attack-type', type=str, default="none")
parser.add_argument('--budget', type=float, default=0.05,
                        help='FGM attack budget')
parser.add_argument('--num-iter', type=int, default=200, # 50 for pgd, 500 for cw    
                    help='IFGM iterate step')
parser.add_argument('--mu', type=float, default=1.,
                    help='momentum factor for MIFGM attack')
parser.add_argument('--local-rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--adv-func', type=str, default='logits',
                        choices=['logits', 'cross_entropy'],
                        help='Adversarial loss function to use')
parser.add_argument('--kappa', type=float, default=0.,
                        help='min margin in logits adv loss')
parser.add_argument('--attack-lr', type=float, default=1e-2,
                        help='lr in CW optimization')
parser.add_argument('--binary-step', type=int, default=10, metavar='N', help='Binary search step')

# DGCNN arguments
parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
parser.add_argument('--model', type=str, default='dgcnn', metavar='N',
                    choices=['pointnet2', 'dgcnn'],
                    help='Model to use, [pointnet, dgcnn]')
parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N',
                    choices=['modelnet40'])
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--num-drop', type=int, default=200, metavar='N',
                        help='Number of dropping points')

args = parser.parse_args()
seed_all(args.seed)

def test(args, io):
    
    # test_dset = ModelNet40Attack(data_root="data/ModelNet40attack", num_points=args.num_points)

    # test_loader = DataLoader(test_dset, batch_size=args.batch_size, num_workers=0, shuffle=True)

    test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points, scale_mode=args.scale_mode),
                            batch_size=args.batch_size, shuffle=False, drop_last=False)
    device = torch.device("cuda" if args.cuda else "cpu")

    #Try to load models
    if args.model == 'pointnet2':
        model = PointNet2_cls().to(device)
    elif args.model == 'dgcnn':
        model = DGCNN_cls().to(device)
    else:
        raise Exception("Not implemented")

    model.eval()
        
    ###
    attacker = None
    if args.adv_func == 'logits':
        adv_func = LogitsAdvLoss(kappa=args.kappa)
    else:
        adv_func = CrossEntropyAdvLoss()
    clip_func = ClipPointsL2(budget=args.budget)
    dist_func = L2Dist()
        
    if args.attack_type == "pgd":
        dist = np.inf  # np.inf
        attacker = PGD(model, args.num_iter, eps=args.budget, alpha=args.attack_lr, p=dist)
    
    elif args.attack_type == "drop":
        attacker = SaliencyDrop(model, num_drop=args.num_drop,
                                alpha=1, k=5)
        
    elif args.attack_type == "drop_pointdp":
        attacker = Drop_PointDP(model, 10, args.num_drop)
    
    elif args.attack_type == "cw_perturb":
        attacker = CWPerturb(model, adv_func, dist_func,
                         attack_lr=args.attack_lr,
                         init_weight=10., max_weight=80.,
                         binary_step=args.binary_step,
                         num_iter=args.num_iter)
    
    elif args.attack_type == "knn":
        dist_func = ChamferkNNDist(chamfer_method='adv2ori',
                        knn_k=5, knn_alpha=1.05,
                        chamfer_weight=5., knn_weight=3.)
        clip_func = ProjectInnerClipLinf(budget=0.1)
        attacker = CWKNN(model, adv_func, dist_func, clip_func,
                        attack_lr=args.attack_lr,
                        num_iter=args.num_iter)
    
    elif args.attack_type.lower() == "none":
        attacker = Identity_Attack()
    
    test_acc = 0.0
    count = 0.0
    test_true = []
    test_target = []
    test_pred = []
    test_pre_attack_pred = []
    test_post_attack_pred = []
        
    # Number of point clouds to test the attack
    stop_iter = args.test_size // args.batch_size
    
    all_ref = []
    all_attack = []
    all_shift = []
    all_scale = []
    
    # Denoiser
    if ae == None:
        denoiser = Identity()
    else:
        denoiser = Layer_Denoiser(model=ae, t=args.t)
    
    for i, batch in enumerate(test_loader):
        
        # Stop at specified batch number
        if i == stop_iter: break
        
        print("Batch : ", i+1, "/", len(test_loader))
        
        # data, label, target_label = data.to(device), label.to(device).squeeze(), target_label.to(device).squeeze()
        
        data = batch['pointcloud'].to(args.device)
        shift = batch['shift']
        scale = batch['scale']
        label = batch["cate"].to(args.device).view(-1)
        
        # Generate random target labels unequal to ground truth labels
        random_labels = torch.randint(0, 40, label.size(), device=args.device)
        random_labels[random_labels == label] += 1
        target_label = torch.remainder(random_labels, 40)
        test_target.append(target_label.cpu().numpy())
        
        with torch.no_grad():
            # Pre-attack prediction
            input_data = data * scale.to(args.device) + shift.to(args.device)
            logits = model(input_data.clone().detach().permute(0, 2, 1))
            pre_attack_preds = logits.max(dim=1)[1]
            test_pre_attack_pred.append(pre_attack_preds.detach().cpu().numpy())
            
        # Attack
        best_pc, _ = attacker.attack(data, target_label, label=label, scale=scale, shift=shift)
            
            
        with torch.no_grad():
            # Post-attack prediction
            input_data = best_pc * scale.to(args.device) + shift.to(args.device)
            logits = model(input_data.clone().detach().permute(0, 2, 1))
            post_attack_preds = logits.max(dim=1)[1]
            test_post_attack_pred.append(post_attack_preds.detach().cpu().numpy())
            
            
        
        # Defended
        with torch.no_grad():
            input_data = best_pc * scale.to(args.device) + shift.to(args.device)
            logits, _ = model.module.forward_denoised(input_data.clone().detach().permute(0, 2, 1), denoiser=denoiser)
            defended_preds = logits.max(dim=1)[1]
            test_true.append(label.cpu().numpy())
            test_pred.append(defended_preds.detach().cpu().numpy())
        
        
        # Save data
        all_ref.append(data.detach().cpu().numpy() * scale.numpy() + shift.numpy())
        all_attack.append(best_pc.cpu().numpy() * scale.numpy() + shift.numpy())
        all_shift.append(shift.numpy())
        all_scale.append(scale.numpy())

        path = "test_attack_outputs/pgd_linf_0.05"
        
        # Save batch
        mode = False
        if mode:
            np.save(os.path.join(path, f'ref_{i}.npy'), np.concatenate(all_ref))
            np.save(os.path.join(path, f'attack_{i}.npy'), np.concatenate(all_attack))
            np.savetxt(os.path.join(path, f'true_label_{i}.txt'), np.concatenate(test_true))
            np.savetxt(os.path.join(path, f'target_label_{i}.txt'), np.concatenate(test_target))
            np.savetxt(os.path.join(path, f'test_pred_{i}.txt'), np.concatenate(test_pred))
            np.savetxt(os.path.join(path, f'test_pre_attack_pred_{i}.txt'), np.concatenate(test_pre_attack_pred))
            np.savetxt(os.path.join(path, f'test_post_attack_pred_{i}.txt'), np.concatenate(test_post_attack_pred)) 
            
    path = "test_attack_outputs/pgd_linf_0.05"
    
    test_true = np.concatenate(test_true)
    test_target = np.concatenate(test_target)
    test_pred = np.concatenate(test_pred)
    test_pre_attack_pred = np.concatenate(test_pre_attack_pred)
    test_post_attack_pred = np.concatenate(test_post_attack_pred)
    all_ref = np.concatenate(all_ref)
    all_attack = np.concatenate(all_attack)
    all_shift = np.concatenate(all_shift)
    all_scale = np.concatenate(all_scale)
    
    np.save(os.path.join(path, 'ref.npy'), all_ref)
    np.save(os.path.join(path, 'attack.npy'), all_attack)
    np.save(os.path.join(path, 'shift.npy'), all_shift)
    np.save(os.path.join(path, 'scale.npy'), all_scale)
    np.savetxt(os.path.join(path, 'true_label.txt'), test_true)
    np.savetxt(os.path.join(path, 'target_label.txt'), test_target)
    np.savetxt(os.path.join(path, 'test_pred.txt'), test_pred)
    np.savetxt(os.path.join(path, 'test_pre_attack_pred.txt'), test_pre_attack_pred)
    np.savetxt(os.path.join(path, 'test_post_attack_pred.txt'), test_post_attack_pred)    
    
    test_acc = metrics.accuracy_score(test_true, test_pre_attack_pred)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pre_attack_pred)
    outstr = 'Test pre attack :: test acc: %.6f, test avg acc: %.6f'%(test_acc, avg_per_class_acc)
    io.cprint(outstr)
    
    test_acc = metrics.accuracy_score(test_true, test_post_attack_pred)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_post_attack_pred)
    outstr = 'Test post attack :: test acc: %.6f, test avg acc: %.6f'%(test_acc, avg_per_class_acc)
    io.cprint(outstr)
    
    test_acc = metrics.accuracy_score(test_true, test_pred)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
    outstr = 'Test :: test acc: %.6f, test avg acc: %.6f'%(test_acc, avg_per_class_acc)
    io.cprint(outstr)
    

# # Logging
# save_dir = os.path.join(args.save_dir, 'AE_Ours_%s_%d' % ('_'.join(args.categories), int(time.time())) )
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)
# logger = get_logger('test', save_dir)
# for k, v in vars(args).items():
#     logger.info('[ARGS::%s] %s' % (k, repr(v)))

# # Checkpoint
# ckpt = torch.load(args.ckpt)


io = IOStream('outputs/' + args.exp_name + '/run.log')
io.cprint(str(args))

args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    io.cprint(
        'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
    torch.cuda.manual_seed(args.seed)
else:
    io.cprint('Using CPU')

test(args, io)

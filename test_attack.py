import os
import time
import argparse
import torch
from tqdm.auto import tqdm

from utils.dataset import *
from utils.misc import *
from utils.data import *
from models.autoencoder import *
from evaluation import EMD_CD


import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from data.modelnet40 import ModelNet40
from models.dgcnn import PointNet, DGCNN_cls
import numpy as np
from torch.utils.data import DataLoader
from utils.util import cal_loss, IOStream
import sklearn.metrics as metrics
from attack import FGM, IFGM, MIFGM, PGD, Identity_Attack, PGD_PointDP, Drop_PointDP
from attack import CWKNN, CWAdd
from attack import CWPerturb
from attack import SaliencyDrop
from attack import CrossEntropyAdvLoss, LogitsAdvLoss
from attack import ClipPointsL2, L2Dist, ChamferkNNDist, ProjectInnerClipLinf
from utils.dataset import ModelNet40Attack

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str, default='./logs_ae/input_denoising_modelnet_06_24/ckpt_0.002547_1601000.pt')
parser.add_argument('--t', type=int, default=0)
parser.add_argument('--categories', type=str_list, default=['airplane'])
parser.add_argument('--save_dir', type=str, default='./denoise_results')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--exp_logs', type=str, default='experiment_logs')
parser.add_argument('--test_size', type=int, default=np.inf)

# Datasets and loaders
parser.add_argument('--dataset_path', type=str, default='./data/shapenet.hdf5')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--scale_mode', type=str, default='shape_unit')

# Attack arguments
parser.add_argument('--attack_type', type=str, default="none")
parser.add_argument('--budget', type=float, default=1.25,
                        help='FGM attack budget')
parser.add_argument('--num_iter', type=int, default=200, # 50 for pgd, 500 for cw    
                    help='IFGM iterate step')
parser.add_argument('--mu', type=float, default=1.,
                    help='momentum factor for MIFGM attack')
parser.add_argument('--local_rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
parser.add_argument('--adv_func', type=str, default='logits',
                        choices=['logits', 'cross_entropy'],
                        help='Adversarial loss function to use')
parser.add_argument('--kappa', type=float, default=0.,
                        help='min margin in logits adv loss')
parser.add_argument('--attack_lr', type=float, default=1e-2,
                        help='lr in CW optimization')
parser.add_argument('--binary_step', type=int, default=10, metavar='N', help='Binary search step')

# DGCNN arguments
parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
parser.add_argument('--model', type=str, default='dgcnn', metavar='N',
                    choices=['pointnet', 'dgcnn'],
                    help='Model to use, [pointnet, dgcnn]')
parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N',
                    choices=['modelnet40'])
parser.add_argument('--epochs', type=int, default=250, metavar='N',
                    help='number of episode to train ')
parser.add_argument('--use_sgd', type=bool, default=True,
                    help='Use SGD')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001, 0.1 if using sgd)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--scheduler', type=str, default='cos', metavar='N',
                    choices=['cos', 'step'],
                    help='Scheduler to use, [cos, step]')
parser.add_argument('--no_cuda', type=bool, default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--eval', type=bool,  default=True,
                    help='evaluate the model')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='initial dropout rate')
parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                    help='Dimension of embeddings')
parser.add_argument('--k', type=int, default=20, metavar='N',
                    help='Num of nearest neighbors to use')
parser.add_argument('--model_path', type=str, default='pretrained/model_original.1024.t7', metavar='N',
                    help='Pretrained model path')
parser.add_argument('--num_drop', type=int, default=200, metavar='N',
                        help='Number of dropping points')

args = parser.parse_args()

def test(args, io):
    
    # test_dset = ModelNet40Attack(data_root="data/ModelNet40attack", num_points=args.num_points)

    # test_loader = DataLoader(test_dset, batch_size=args.batch_size, num_workers=0, shuffle=True)

    test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points, scale_mode=args.scale_mode),
                            batch_size=args.batch_size, shuffle=True, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")

    #Try to load models
    if args.model == 'pointnet':
        model = PointNet().to(device)
    elif args.model == 'dgcnn':
        model = DGCNN_cls(args).to(device)
    else:
        raise Exception("Not implemented")

    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(args.model_path))
    model = model.eval()
    
    
    # Autoencoder
    ae = None
    if args.t != 0:
        ckpt = torch.load(args.ckpt)
        ae = AutoEncoder(ckpt['args']).to(args.device)
        ae.load_state_dict(ckpt['state_dict'])
        ae.eval()

    
    ###
    attacker = None
    if args.adv_func == 'logits':
        adv_func = LogitsAdvLoss(kappa=args.kappa)
    else:
        adv_func = CrossEntropyAdvLoss()
    clip_func = ClipPointsL2(budget=args.budget)
    dist_func = L2Dist()
    
    if args.attack_type == "pgd":
        args.budget = args.budget * np.sqrt(args.num_points * 3)  # \delta * \sqrt(N * d)
        args.num_iter = int(args.num_iter)
        args.step_size = args.budget / float(args.num_iter)
        attacker = PGD(model, adv_func=adv_func,
                        clip_func=clip_func, budget=args.budget, step_size=args.step_size,
                        num_iter=args.num_iter, dist_metric='l2')
        
    elif args.attack_type == "pgd_pointdp":
        dist = 2  # np.inf
        attacker = PGD_PointDP(model, args.num_iter, eps=args.budget, alpha=args.attack_lr, p=dist)
    
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
    all_recons = []
    
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
        
        # Pre-attack prediction
        
        input_data = data * scale.to(args.device) + shift.to(args.device)
        logits = model(input_data.clone().detach().permute(0, 2, 1))
        pre_attack_preds = logits.max(dim=1)[1]
        test_pre_attack_pred.append(pre_attack_preds.detach().cpu().numpy())
        
        # Attack
        best_pc, _ = attacker.attack(data, target_label, label=label, scale=scale, shift=shift)
        
        
        # Post-attack prediction
        input_data = best_pc * scale.to(args.device) + shift.to(args.device)
        logits = model(input_data.clone().detach().permute(0, 2, 1))
        post_attack_preds = logits.max(dim=1)[1]
        test_post_attack_pred.append(post_attack_preds.detach().cpu().numpy())
        
        # Denoising
        if args.t != 0:
            with torch.no_grad():
                code = ae.encode(best_pc)
                # Changed Encode -> Denoiser to call truncated_sample()
                recons = ae.denoiser(best_pc, t=args.t, flexibility=ckpt['args'].flexibility, context=code).detach()
        else:
            recons = best_pc.detach()
        
        # Save data
        all_ref.append(data.detach().cpu().numpy() * scale.numpy() + shift.numpy())
        all_attack.append(best_pc.cpu().numpy() * scale.numpy() + shift.numpy())
        all_recons.append(recons.cpu().numpy() * scale.numpy() + shift.numpy())
        
        # Defended prediction
        input_data = recons * scale.to(args.device) + shift.to(args.device)
        logits = model(input_data.permute(0, 2, 1))
        defended_preds = logits.max(dim=1)[1]
        test_true.append(label.cpu().numpy())
        test_pred.append(defended_preds.detach().cpu().numpy())
        
        # Save batch
        mode = False
        if mode:
            np.save(os.path.join("test_attack_outputs", f'ref_{i}.npy'), np.concatenate(all_ref))
            np.save(os.path.join("test_attack_outputs", f'attack_{i}.npy'), np.concatenate(all_attack))
            np.save(os.path.join("test_attack_outputs", f'recons_{i}.npy'), np.concatenate(all_recons))
            np.savetxt(os.path.join("test_attack_outputs", f'true_label_{i}.txt'), np.concatenate(test_true))
            np.savetxt(os.path.join("test_attack_outputs", f'target_label_{i}.txt'), np.concatenate(test_target))
            np.savetxt(os.path.join("test_attack_outputs", f'test_pred_{i}.txt'), np.concatenate(test_pred))
            np.savetxt(os.path.join("test_attack_outputs", f'test_pre_attack_pred_{i}.txt'), np.concatenate(test_pre_attack_pred))
            np.savetxt(os.path.join("test_attack_outputs", f'test_post_attack_pred_{i}.txt'), np.concatenate(test_post_attack_pred)) 

    test_true = np.concatenate(test_true)
    test_target = np.concatenate(test_target)
    test_pred = np.concatenate(test_pred)
    test_pre_attack_pred = np.concatenate(test_pre_attack_pred)
    test_post_attack_pred = np.concatenate(test_post_attack_pred)
    all_ref = np.concatenate(all_ref)
    all_attack = np.concatenate(all_attack)
    all_recons = np.concatenate(all_recons)
    
    np.save(os.path.join("test_attack_outputs", 'ref.npy'), all_ref)
    np.save(os.path.join("test_attack_outputs", 'attack.npy'), all_attack)
    np.save(os.path.join("test_attack_outputs", 'recons.npy'), all_recons)
    np.savetxt(os.path.join("test_attack_outputs", 'true_label.txt'), test_true)
    np.savetxt(os.path.join("test_attack_outputs", 'target_label.txt'), test_target)
    np.savetxt(os.path.join("test_attack_outputs", 'test_pred.txt'), test_pred)
    np.savetxt(os.path.join("test_attack_outputs", 'test_pre_attack_pred.txt'), test_pre_attack_pred)
    np.savetxt(os.path.join("test_attack_outputs", 'test_post_attack_pred.txt'), test_post_attack_pred)    
    
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
# seed_all(ckpt['args'].seed)


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


## Model
# logger.info('Loading model...')
# model = AutoEncoder(ckpt['args']).to(args.device)
# model.load_state_dict(ckpt['state_dict'])

# all_ref = []
# all_recons = []
# for i, batch in enumerate(tqdm(test_loader)):
#     ref = batch['pointcloud'].to(args.device)
#     shift = batch['shift'].to(args.device)
#     scale = batch['scale'].to(args.device)
#     model.eval()
#     with torch.no_grad():
#         code = model.encode(ref)
#         # Changed Encode -> Denoiser to call truncated_sample()
#         recons = model.denoiser(ref, t=args.t, flexibility=ckpt['args'].flexibility, context=code).detach()

#     ref = ref * scale + shift
#     recons = recons * scale + shift

#     all_ref.append(ref.detach().cpu())
#     all_recons.append(recons.detach().cpu())

# all_ref = torch.cat(all_ref, dim=0)
# all_recons = torch.cat(all_recons, dim=0)

# logger.info('Saving point clouds...')
# np.save(os.path.join(save_dir, 'ref.npy'), all_ref.numpy())
# np.save(os.path.join(save_dir, 'out.npy'), all_recons.numpy())

# logger.info('Start computing metrics...')
# metrics = EMD_CD(all_recons.to(args.device), all_ref.to(args.device), batch_size=args.batch_size)
# cd, emd = metrics['MMD-CD'].item(), metrics['MMD-EMD'].item()
# logger.info('CD:  %.12f' % cd)
# logger.info('EMD: %.12f' % emd)

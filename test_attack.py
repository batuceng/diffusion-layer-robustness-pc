"""
Test denoising model on pre-saved PGD attacked data.
"""


import os
import argparse
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
import sklearn.metrics as metrics
from util.misc import IOStream, seed_all
from dataset.modelnet40 import ModelNet40Attack
from models.autoencoder import AutoEncoder
from models.dgcnn import DGCNN_cls
from models.pointnet2 import PointNet2_cls
from models.denoiser import Identity, Layer_Denoiser, Multiple_Layer_Denoiser

import warnings

warnings.filterwarnings("ignore")

layer0 = "logs/layer0/AE_2023_07_10__22_25_08/ckpt_0.003777_602000.pt"
layer1 = "logs/layer1/AE_2023_07_17__20_03_03/ckpt_0.585605_616000.pt"
layer2 = "logs/layer2/AE_2023_07_17__20_14_47/ckpt_1.611477_750000.pt"

available_models = [layer0, layer1, layer2]

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--layers', type=str, default="0", help="Layer no list in the string format : '0, 1, 2'")
parser.add_argument('--t-list', type=str, default="30")
# parser.add_argument('--categories', type=str_list, default=['airplane'])
parser.add_argument('--save-dir', type=str, default='./denoise_results')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--exp-logs', type=str, default='experiment_logs')
parser.add_argument('--test-size', type=int, default=np.inf)

# Datasets and loaders
# parser.add_argument('--dataset-path', type=str, default='./data/shapenet.hdf5')
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--scale-mode', type=str, default='shape_unit')

# Attack arguments
parser.add_argument('--attack-type', type=str, default="none")
parser.add_argument('--budget', type=float, default=1.25,
                        help='FGM attack budget')
parser.add_argument('--num-iter', type=int, default=200, # 50 for pgd, 500 for cw    
                    help='IFGM iterate step')
parser.add_argument('--mu', type=float, default=1.,
                    help='momentum factor for MIFGM attack')
parser.add_argument('--local-rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--num-points', type=int, default=1024,
                        help='num of points to use')
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
parser.add_argument('-model', type=str, metavar='N', choices=['pointnet2', 'dgcnn'],
                    help='Model to use, [pointnet2, dgcnn]')
parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N',
                    choices=['modelnet40'])
parser.add_argument('--epochs', type=int, default=250, metavar='N',
                    help='number of episode to train ')
parser.add_argument('--use-sgd', type=bool, default=True,
                    help='Use SGD')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001, 0.1 if using sgd)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--scheduler', type=str, default='cos', metavar='N',
                    choices=['cos', 'step'],
                    help='Scheduler to use, [cos, step]')
parser.add_argument('--no-cuda', type=bool, default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--eval', type=bool,  default=True,
                    help='evaluate the model')
parser.add_argument('--data-path', type=str ,default="test_attack_outputs/pgd_linf_0.05")

args = parser.parse_args()
seed_all(args.seed)

args.layers = [int(i.strip()) for i in args.layers.split(",")]
args.t_list = [int(i.strip()) for i in args.t_list.split(",")]
args.available_models = available_models


def test(args, io):
    
    # test_dset = ModelNet40Attack(data_root="data/ModelNet40attack", num_points=args.num_points)

    # test_loader = DataLoader(test_dset, batch_size=args.batch_size, num_workers=0, shuffle=True)

    test_loader = DataLoader(ModelNet40Attack(args.num_points, path = args.data_path),
                            batch_size=args.batch_size, shuffle=False, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")

    #Try to load models
    if args.model == 'pointnet2':
        model = PointNet2_cls().to(device)
    elif args.model == 'dgcnn':
        model = DGCNN_cls().to(device)
    else:
        raise Exception("Not implemented")

    model = model.eval()
    
    # Autoencoders
    ae_list = []
    if len(args.t_list) != 0:
        for i in args.layers:
            ckpt = torch.load(args.available_models[i])
            ae = AutoEncoder(ckpt['args']).to(args.device)
            ae.load_state_dict(ckpt['state_dict'])
            ae.eval()
            ae_list.append(ae)
        
    test_acc = 0.0
    count = 0.0
    test_true = []
    test_attack_pred = []
    test_defended_pred = []
    test_pre_attack_pred = []
    test_pre_attack_defense = []
        
    # Number of point clouds to test the attack
    stop_iter = args.test_size // args.batch_size
    
    all_attack = []
    
    # Denoiser
    if ae == None:
        denoiser = Identity()
    else:
        denoiser = Multiple_Layer_Denoiser(models=ae_list, t_list=args.t_list)
    
    for i, batch in enumerate(test_loader):
        
        # Stop at specified batch number
        if i == stop_iter: break
        
        # print("Batch : ", i+1, "/", len(test_loader))
        
        # data, label, target_label = data.to(device), label.to(device).squeeze(), target_label.to(device).squeeze()
        
        data = batch['pointcloud'].to(args.device)
        data_attack = batch['attack'].to(args.device)
        shift = batch['shift']
        scale = batch['scale']
        label = batch["cate"].to(args.device).view(-1)
        
        with torch.no_grad():
            # Prediction on original data
            input_data = data * scale.to(args.device) + shift.to(args.device)
            logits = model(input_data.clone().detach().permute(0, 2, 1))
            pre_attack_preds = logits.max(dim=1)[1]
            test_pre_attack_pred.append(pre_attack_preds.detach().cpu().numpy())
        
        with torch.no_grad():
            # Prediction on attacked data
            input_data = data_attack * scale.to(args.device) + shift.to(args.device)
            logits = model(input_data.clone().detach().permute(0, 2, 1))
            attack_preds = logits.max(dim=1)[1]
            test_attack_pred.append(attack_preds.detach().cpu().numpy())
            
        with torch.no_grad():
            # Prediction with layer denoising on pre-attacked data
            input_data = data * scale.to(args.device) + shift.to(args.device)
            logits, _ = model.forward_denoised(input_data.clone().detach().permute(0, 2, 1), denoiser=denoiser)
            pre_attack_defense_preds = logits.max(dim=1)[1]
            test_pre_attack_defense.append(pre_attack_defense_preds.detach().cpu().numpy())
        
        with torch.no_grad():
            # Prediction with layer denoising on attacked data
            input_data = data_attack * scale.to(args.device) + shift.to(args.device)
            logits, _ = model.forward_denoised(input_data.clone().detach().permute(0, 2, 1), denoiser=denoiser)
            defended_preds = logits.max(dim=1)[1]
            test_true.append(label.cpu().numpy())
            test_defended_pred.append(defended_preds.detach().cpu().numpy())
        
        # Save data
        all_attack.append(data_attack.cpu().numpy() * scale.numpy() + shift.numpy())


    test_true = np.concatenate(test_true)
    test_pre_attack_pred = np.concatenate(test_pre_attack_pred)
    test_attack_pred = np.concatenate(test_attack_pred)
    test_defended_pred = np.concatenate(test_defended_pred)
    test_pre_attack_defense = np.concatenate(test_pre_attack_defense)
    all_attack = np.concatenate(all_attack)
    
    save_mode = False
    if save_mode:
        try:
            os.mkdir("pgd_test_outputs")
        except Exception:
            pass
        np.save(os.path.join("pgd_test_outputs", 'attack.npy'), all_attack)
        np.savetxt(os.path.join("pgd_test_outputs", 'true_label.txt'), test_true)
        np.savetxt(os.path.join("pgd_test_outputs", 'test_attack_pred.txt'), test_attack_pred)
        np.savetxt(os.path.join("pgd_test_outputs", 'test_defended_pred.txt'), test_defended_pred)    
    
    io.cprint("Layers : " + str(args.layers))
    io.cprint("T list : " + str(args.t_list))
    
    test_acc = metrics.accuracy_score(test_true, test_pre_attack_pred)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pre_attack_pred)
    outstr = 'Test pre attack :: test acc: %.6f, test avg acc: %.6f'%(test_acc, avg_per_class_acc)
    io.cprint(outstr)
    
    test_acc = metrics.accuracy_score(test_true, test_pre_attack_defense)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pre_attack_defense)
    outstr = 'Test pre attack defense :: test acc: %.6f, test avg acc: %.6f'%(test_acc, avg_per_class_acc)
    io.cprint(outstr)
    
    test_acc = metrics.accuracy_score(test_true, test_attack_pred)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_attack_pred)
    outstr = 'Test attack :: test acc: %.6f, test avg acc: %.6f'%(test_acc, avg_per_class_acc)
    io.cprint(outstr)
    
    test_acc = metrics.accuracy_score(test_true, test_defended_pred)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_defended_pred)
    outstr = 'Test defense :: test acc: %.6f, test avg acc: %.6f'%(test_acc, avg_per_class_acc)
    io.cprint(outstr)
    io.cprint("\n")
    
    
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
# io.cprint(str(args))

args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
# if args.cuda:
#     io.cprint(
#         'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
#     torch.cuda.manual_seed(args.seed)
# else:
#     io.cprint('Using CPU')

test(args, io)

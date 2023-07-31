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
from models.dgcnn import PointNet, DGCNN_cls
from models.denoiser import Identity, Layer_Denoiser, Multiple_Layer_Denoiser

import warnings

warnings.filterwarnings("ignore")

layer0 = "logs/layer0/AE_2023_07_10__22_25_08/ckpt_0.003777_602000.pt"
layer1 = "logs/layer1/AE_2023_07_17__20_03_03/ckpt_0.585605_616000.pt"
layer2 = "logs/layer2/AE_2023_07_17__20_14_47/ckpt_1.611477_750000.pt"
layer3 = "logs/layer3/AE_2023_07_28__18_41_09/ckpt_3.507668_441000.pt"
layer4 = "Identity"

available_models = [layer0, layer1, layer2, layer3, layer4]

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--L0', type=str, default="0..0..0")
parser.add_argument('--L1', type=str, default="0..0..0")
parser.add_argument('--L2', type=str, default="0..0..0")
parser.add_argument('--L3', type=str, default="0..0..0")
parser.add_argument('--L4', type=str, default="0..0..0")

parser.add_argument('--save-dir', type=str, default='./denoise_results')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--exp-logs', type=str, default='experiment_logs')
parser.add_argument('--test-size', type=int, default=np.inf)

# Datasets and loaders
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--scale-mode', type=str, default='shape_unit')


# DGCNN arguments
parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')

parser.add_argument('--model', type=str, default='dgcnn', metavar='N',
                    choices=['pointnet', 'dgcnn'],
                    help='Model to use, [pointnet, dgcnn]')
parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N',
                    choices=['modelnet40'])
parser.add_argument('--num-points', type=int, default=1024,
                        help='num of points to use')

parser.add_argument('--no-cuda', type=bool, default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--eval', type=bool,  default=True,
                    help='evaluate the model')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='initial dropout rate')
parser.add_argument('--emb-dims', type=int, default=1024, metavar='N',
                    help='Dimension of embeddings')
parser.add_argument('--k', type=int, default=20, metavar='N',
                    help='Num of nearest neighbors to use')
parser.add_argument('--model-path', type=str, default='pretrained/model.1024.t7', metavar='N',
                    help='Pretrained model path')


args = parser.parse_args()
seed_all(args.seed)

args.t_list = [int(i.strip()) for i in args.t_list.split(",")]
args.available_models = available_models


def test(args, io):

    test_loader = DataLoader(ModelNet40Attack(args.num_points), path = "test_attack_outputs/pgd_linf_0.05",
                            batch_size=args.batch_size, shuffle=False, drop_last=False)

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
    
    # Autoencoders
    ae_list = []
    for lay in args.available_models:
        if lay is "Identity":
            ae = Identity()
        else:
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
            # Prediction with layer denoising on clean data
            input_data = data * scale.to(args.device) + shift.to(args.device)
            logits, _ = model.module.forward_denoised(input_data.clone().detach().permute(0, 2, 1), denoiser=denoiser)
            pre_attack_defense_preds = logits.max(dim=1)[1]
            test_pre_attack_defense.append(pre_attack_defense_preds.detach().cpu().numpy())
        
        with torch.no_grad():
            # Prediction with layer denoising on attacked data
            input_data = data_attack * scale.to(args.device) + shift.to(args.device)
            logits, _ = model.module.forward_denoised(input_data.clone().detach().permute(0, 2, 1), denoiser=denoiser)
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

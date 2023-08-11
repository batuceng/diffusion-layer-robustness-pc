"""
Test denoising model on pre-saved PGD attacked data.
"""


import os
import argparse
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from util.misc import IOStream, seed_all
from dataset.modelnet40attack import ModelNet40Attack
from models.autoencoder import AutoEncoder
from models import DGCNN_cls, PointNet2_cls
from models.denoiser import Identity, Layer_Denoiser, Multiple_Layer_Denoiser
from sklearn.metrics import accuracy_score
from util.evaluation_metrics import EMD_CD

import warnings

warnings.filterwarnings("ignore")

layer0 = "logs/layer0/AE_2023_07_10__22_25_08/ckpt_0.003777_602000.pt"
layer1 = "logs/layer1/AE_2023_07_17__20_03_03/ckpt_0.585605_616000.pt"
layer2 = "logs/layer2/AE_2023_07_17__20_14_47/ckpt_1.611477_750000.pt"
layer3 = "logs/layer3/AE_2023_07_28__18_41_09/ckpt_3.507668_441000.pt"

available_models = [layer0, layer1, layer2, layer3]

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('-t_list', type=str, default='5,0,0,0')

parser.add_argument('--save-dir', type=str, default='./denoise_results')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--exp-logs', type=str, default='experiment_logs')
parser.add_argument('--test_size', type=int, default=np.inf)

# Datasets and loaders
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--scale-mode', type=str, default='shape_unit')

# DGCNN arguments
parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
parser.add_argument('--model', type=str, default='dgcnn', metavar='N',
                    choices=['pointnet2', 'dgcnn'],
                    help='Model to use, [pointnet2, dgcnn]')
parser.add_argument('--dataset', type=str, default='modelnet40attack', metavar='N',
                    choices=['modelnet40attack'])
parser.add_argument('--num-points', type=int, default=1024,
                        help='num of points to use')

parser.add_argument('--no-cuda', type=bool, default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--data_attacked', type=str, default="/home/robust/diffusion-point-cloud/data_attacked/ModelNet40_DGCNN_cls_PGDLinf_eps_0.05.pt")


args = parser.parse_args()
seed_all(args.seed)

args.t_list = [int(i.strip()) for i in args.t_list.split(",")]
args.available_models = available_models

@torch.no_grad()
def get_logits(data, model, shift, scale, device, denoiser=Identity()):
    input = (data.to(device) * scale.view(-1,1,1).to(device) + shift.to(device)).detach().permute(0,2,1)
    model = model.to(device)
    logits, layers = model.forward_denoised(input, denoiser)
    return logits.clone().detach().cpu(), [l.clone().detach().cpu() for l in layers]

@torch.no_grad()
def get_stats(labels, logits):
    preds = logits.max(dim=1)[1]
    acc = accuracy_score(labels.squeeze().numpy(), preds.squeeze().numpy())
    loss = F.cross_entropy(logits.to(torch.float64), labels.squeeze().to(torch.long))
    return acc, loss.item()

@torch.no_grad()
def get_layer_distances(layerA, layerB):
    for la, lb in zip(layerA, layerB):
        get_distances(la, lb)
    pass

    # ObjA Clean Pc, ObjB Denoised PC
    def get_distances(objA, objB):
        # L2
        l2dist = torch.norm((objA-objB), p=2, dim=0)
        # Linf
        linfdist = torch.norm((objA-objB), torch.inf, dim=0)
        # Chamfer
        metrics = EMD_CD(layerB, layerA, batch_size=args.val_batch_size)
        cd, emd = metrics['MMD-CD'].item(), metrics['MMD-EMD'].item()
        return l2dist, linfdist, cd

def test(args, io):

    test_loader = DataLoader(ModelNet40Attack(path=args.data_attacked),
                            batch_size=args.batch_size, shuffle=False, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")

    #Try to load models
    if args.model == 'pointnet':
        model = PointNet2_cls().to(device)
    elif args.model == 'dgcnn':
        model = DGCNN_cls().to(device)
    else:
        raise Exception("Not implemented")
    model.eval()
    
    # Autoencoders
    ae_list = []
    for lay in args.available_models:
        if lay == "Identity":
            ae = Identity()
        else:
            ckpt = torch.load(lay)
            ae = AutoEncoder(ckpt['args']).to(args.device)
            ae.load_state_dict(ckpt['state_dict'])
            ae.eval()
        ae_list.append(ae)
    denoiser = Multiple_Layer_Denoiser(models=ae_list, t_list=args.t_list)
    
    num_classes = 40    
    clean_preds_list = torch.tensor([]).view(0,num_classes)
    attacked_preds_list = torch.tensor([]).view(0,40)
    denoised_preds_list = torch.tensor([]).view(0,40)
    defended_preds_list = torch.tensor([]).view(0,40)
    true_label_list = torch.tensor([]).view(0,1)
    
    # Number of point clouds to test the attack
    stop_iter = args.test_size // args.batch_size
    # stop_iter = 5
    
    for i, batch in enumerate(test_loader):
        # print(f"batch {i}/{len(test_loader)}")
        # Stop at specified batch number
        if i == stop_iter: break
        
        data = batch['pointcloud']
        data_attack = batch['attacked']
        shift = batch['shift']
        scale = batch['scale']
        label = batch["cate"]
        
        # Prediction on original data
        clean_logits, clean_layers = get_logits(batch['pointcloud'], model, shift, scale, device)
        clean_preds_list = torch.cat((clean_preds_list, clean_logits), dim=0)
        
        # Prediction on attacked data
        attacked_logits, attacked_layers = get_logits(batch['attacked'], model, shift, scale, device)
        attacked_preds_list = torch.cat((attacked_preds_list, attacked_logits), dim=0)
        
        # Defensive Prediction on original data
        denoised_logits, denoised_layers = get_logits(batch['pointcloud'], model, shift, scale, device, denoiser=denoiser)
        denoised_preds_list = torch.cat((denoised_preds_list, denoised_logits), dim=0)
        
        # Defensive Prediction on attacked data
        defended_logits, defended_layers = get_logits(batch['attacked'], model, shift, scale, device, denoiser=denoiser)
        defended_preds_list = torch.cat((defended_preds_list, defended_logits), dim=0)
        
        # Label
        true_label_list = torch.cat((true_label_list, label), dim=0)
    
    
    print(f"\n T_List: {args.t_list}")
    # Print out Results
    clean_acc, clean_loss = get_stats(true_label_list, clean_preds_list)
    print("Clean Acc:", clean_acc)
    print("Clean Loss:", clean_loss)
    
    attacked_acc, attacked_loss = get_stats(true_label_list, attacked_preds_list)
    print("Attacked Acc:", attacked_acc)
    print("Attacked Loss:", attacked_loss)
    
    denoised_acc, denoised_loss = get_stats(true_label_list, denoised_preds_list)
    print("Denoised Acc:", denoised_acc)
    print("Denoised Loss:", denoised_loss)
    
    defended_acc, defended_loss = get_stats(true_label_list, defended_preds_list)
    print("Defended Acc:", defended_acc)
    print("Defended Loss:", defended_loss)
    
    print("Geo-Mean Acc:", np.sqrt((denoised_acc**2 + defended_acc**2)/2)  )

    # test_acc = metrics.accuracy_score(test_true, test_pre_attack_pred)
    # avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pre_attack_pred)
    # outstr = 'Test pre attack :: test acc: %.6f, test avg acc: %.6f'%(test_acc, avg_per_class_acc)
    # io.cprint(outstr)
    
    # test_acc = metrics.accuracy_score(test_true, test_pre_attack_defense)
    # avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pre_attack_defense)
    # outstr = 'Test pre attack defense :: test acc: %.6f, test avg acc: %.6f'%(test_acc, avg_per_class_acc)
    # io.cprint(outstr)
    
    # test_acc = metrics.accuracy_score(test_true, test_attack_pred)
    # avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_attack_pred)
    # outstr = 'Test attack :: test acc: %.6f, test avg acc: %.6f'%(test_acc, avg_per_class_acc)
    # io.cprint(outstr)
    
    # test_acc = metrics.accuracy_score(test_true, test_defended_pred)
    # avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_defended_pred)
    # outstr = 'Test defense :: test acc: %.6f, test avg acc: %.6f'%(test_acc, avg_per_class_acc)
    # io.cprint(outstr)
    # io.cprint("\n")
    
    
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

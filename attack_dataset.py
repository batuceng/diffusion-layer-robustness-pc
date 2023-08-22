import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from datetime import datetime
import argparse
import os

from dataset.modelnet40 import ModelNet40
from models import PCT_cls, DGCNN_cls, CurveNet_cls, PointMLP_cls, PointNet_cls, PointNet2_cls
from attack import PGDLinf, PGDL2, VANILA, PointDrop

def none_or_str(value):
    if value == 'None':
        return None
    return value

# Arguments
parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--dataset', type=str, default='modelnet40', choices=['modelnet40'],
                    help='Dataset to use, [modelnet40]')
parser.add_argument('-model', type=str, metavar='N', choices=['pointnet', 'pointnet2', 'dgcnn', 'curvenet', 'pct', 'pointmlp'],
                    help='Model to use, [pointnet, pointnet2, dgcnn, curvenet, pct, pointmlp]')
parser.add_argument('--save_path', type=none_or_str, default="./data_attacked",
                    help="Path to save the attacked dataset. Give None to not save. Creates subdirs.") 
subparsers = parser.add_subparsers(dest='attack')
subparsers.required = True

# subparsers for PGD Attack
parser_pgd = subparsers.add_parser('pgd').add_argument_group()
parser_pgd.add_argument('--eps', type=float, default=0.05,
                        help='Linf eps bound')
parser_pgd.add_argument('--alpha', type=float, default=0.002,
                        help='Size of each step')
parser_pgd.add_argument('--steps', type=int, default=30,
                        help='Number of iteration steps for pgd')


# subparsers for PGDL2 Attack
parser_pgdl2 = subparsers.add_parser('pgdl2')
parser_pgdl2.add_argument('--eps', type=float, default=1.25,
                        help='L2 eps bound')
parser_pgdl2.add_argument('--alpha', type=float, default=0.05,
                        help='Size of each step')
parser_pgdl2.add_argument('--steps', type=int, default=30,
                        help='Number of iteration steps for pgdL2')

# subparsers for Vanila Attack
parser_vanila = subparsers.add_parser('vanila')

# subparsers for PointDrop Attack
parser_pgdl2 = subparsers.add_parser('drop')
parser_pgdl2.add_argument('--num_points', type=int, default=200,
                        help='Number of points to drop')
parser_pgdl2.add_argument('--steps', type=int, default=10,
                        help='Number of iteration steps for drop')
# Parse args
parser._action_groups.reverse()

args = parser.parse_args()

print(args)
# print(vars(args))

# Model Dictionary
model_dict = {
    'curvenet': CurveNet_cls, #0.938412
    'pct':      PCT_cls, #0.931524
    'pointmlp': PointMLP_cls, #0.940032
    'dgcnn':    DGCNN_cls, #0.928687
    'pointnet2':PointNet2_cls, #0.911264
    'pointnet': PointNet_cls, #0.896677
}
# Attack Dictionary
attack_dict = {
    'pgd':      PGDLinf,
    'pgdl2':    PGDL2,
    'drop':     PointDrop,
    'vanila':   VANILA,
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model_dict[args.model]().to(device)
model.eval()
dataset = ModelNet40(num_points=1024, partition="test", scale_mode='none')
test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

# Try to pass args via groups later
if args.attack == 'pgd':
    atk = PGDLinf(model=model, device=device, eps=args.eps, alpha=args.alpha, steps=args.steps)
elif args.attack == 'pgdl2':
    atk = PGDL2(model=model, device=device, eps=args.eps, alpha=args.alpha, steps=args.steps)
elif args.attack == 'vanila':
    atk = VANILA(model=model, device=device)
elif args.attack == 'drop':
    atk = PointDrop(model=model, device=device, num_points=args.num_points, steps=args.steps)
else:
    raise NotImplementedError

# Define subfolders to save
savedir = os.path.join(*[args.save_path, args.attack, args.model])
filename = os.path.join(savedir,datetime.today().strftime('ATK_%Y_%m_%d__%H_%M_%S'))
# Apply Attack & Save
atk.save(dataloader=test_loader, root=savedir, file_name=filename, args=vars(args))


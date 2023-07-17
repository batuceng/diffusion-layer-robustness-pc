import os
import argparse
import torch
import torch.utils.tensorboard
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm

from utils.dataset import *
from utils.misc import *
from utils.data import *
from utils.transform import *
from models.autoencoder import *
from evaluation import EMD_CD
from data.modelnet40 import ModelNet40
from models.dgcnn import PointNet, DGCNN_cls, Identity
import time

# Arguments
parser = argparse.ArgumentParser()
# Model arguments
parser.add_argument('--latent-dim', type=int, default=256)
parser.add_argument('--num_steps', type=int, default=200)
parser.add_argument('--beta-1', type=float, default=1e-4)
parser.add_argument('--beta-T', type=float, default=0.05)
parser.add_argument('--sched_mode', type=str, default='linear')
parser.add_argument('--flexibility', type=float, default=0.0)
parser.add_argument('--residual', type=eval, default=True, choices=[True, False])
parser.add_argument('--resume', type=str, default=None)

# Datasets and loaders
parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N',
                    choices=['modelnet40'])
parser.add_argument('--num-points', type=int, default=1024)
parser.add_argument('--scale-mode', type=str, default='none')  # Use none for DGCNN
parser.add_argument('--train-batch_size', type=int, default=128)
parser.add_argument('--val-batch_size', type=int, default=32)
parser.add_argument('--rotate', type=eval, default=False, choices=[True, False])

# Optimizer and scheduler
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight-decay', type=float, default=0)
parser.add_argument('--max-grad-norm', type=float, default=10)
parser.add_argument('--end-lr', type=float, default=1e-4)
parser.add_argument('--sched-start-epoch', type=int, default=150*THOUSAND)
parser.add_argument('--sched-end-epoch', type=int, default=300*THOUSAND)

# Training
parser.add_argument('--seed', type=int, default=2020)
parser.add_argument('--logging', type=eval, default=True, choices=[True, False])
parser.add_argument('--log-root', type=str, default='./logs/layer1')
parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
parser.add_argument('--max-iters', type=int, default=float('inf'))
parser.add_argument('--val-freq', type=float, default=1000)
parser.add_argument('--tag', type=str, default=None)
parser.add_argument('--num-val-batches', type=int, default=-1)
parser.add_argument('--num-inspect-batches', type=int, default=1)
parser.add_argument('--num-inspect-pointclouds', type=int, default=4)

# Classifier Args
parser.add_argument('--layer-no', type=int, default=0, choices=[0, 1, 2, 3, 4])
parser.add_argument('--model', type=str, default='dgcnn', metavar='N',
                    choices=['pointnet', 'dgcnn'],
                    help='Model to use, [pointnet, dgcnn]')
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

# Logging
if args.logging:
    log_dir = get_new_log_dir(args.log_root, prefix='AE_', postfix='_' + args.tag if args.tag is not None else '')
    logger = get_logger('train', log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    ckpt_mgr = CheckpointManager(log_dir)
else:
    logger = get_logger('train', None)
    writer = BlackHole()
    ckpt_mgr = BlackHole()
logger.info(args)

# Datasets and loaders
if args.dataset == "modelnet40":
    train_dset = ModelNet40(partition='train', scale_mode=args.scale_mode, num_points=args.num_points)
                            
    train_iter = get_data_iterator(DataLoader(
        train_dset,
        batch_size=args.train_batch_size,
        num_workers=0,
    ))

    test_loader = DataLoader(ModelNet40(partition='test', scale_mode=args.scale_mode, num_points=args.num_points), num_workers=8,
                                batch_size=args.val_batch_size, shuffle=True, drop_last=False)
else:
    raise Exception("Unavailable Dataset!")

# Device
device = torch.device("cuda" if (torch.cuda.is_available() and args.device=="cuda") else "cpu")

# Classification Model
if args.model == 'pointnet':
    cls_model = PointNet().to(device)
elif args.model == 'dgcnn':
    cls_model = DGCNN_cls(args).to(device)
    input_dim_dict = {0:3, 1:64, 2:64, 3:128, 4:256}
else:
    raise Exception("Not implemented")
args.input_dim = input_dim_dict[args.layer_no]

cls_model = nn.DataParallel(cls_model)
cls_model.load_state_dict(torch.load(args.model_path))
cls_model = cls_model.eval()

# Denoiser
denoiser = Identity()

# Model
logger.info('Building model...')
if args.resume is not None:
    logger.info('Resuming from checkpoint...')
    ckpt = torch.load(args.resume)
    model = AutoEncoder(ckpt['args']).to(args.device)
    model.load_state_dict(ckpt['state_dict'])
else:
    model = AutoEncoder(args).to(args.device)
logger.info(repr(model))


# Optimizer and scheduler
optimizer = torch.optim.Adam(model.parameters(), 
    lr=args.lr, 
    weight_decay=args.weight_decay
)
scheduler = get_linear_scheduler(
    optimizer,
    start_epoch=args.sched_start_epoch,
    end_epoch=args.sched_end_epoch,
    start_lr=args.lr,
    end_lr=args.end_lr
)

def normalize_layer_data(data, normalization="unit_shape"):
    # if normalization == "unit_shape":
        # data.shape = [B, N, D]
    shift = data.mean(dim=1, keepdim=True) # (B, 1, D)
    scale = data.std(dim=[1,2], keepdim=True) # (B, 1, 1)
    data = (data - shift) / scale # Normalize layer data
    return data, shift, scale

def get_layer_data(batch, layer_no):
    data = batch['pointcloud'].to(args.device)
    
    cls_model.eval()
    with torch.no_grad():
        # CLS forward
        _, data = cls_model.module.forward_denoised(data.permute(0,2,1), denoiser)
        data = data[layer_no]
        data = data.permute(0,2,1).to(device)
        # Normalize
        _ = normalize_layer_data(data)
        return _
    

# Train, validate 
def train(it):
    # Load data
    batch = next(train_iter)
    
    # Get Layer Data
    x, shift, scale = get_layer_data(batch, layer_no= args.layer_no)

    # Reset grad and model state
    optimizer.zero_grad()
    model.train()

    # Forward
    loss = model.get_loss(x)
    
    # Backward and optimize
    loss.backward()
    
    orig_grad_norm = clip_grad_norm_(model.parameters(), args.max_grad_norm)
    optimizer.step()
    scheduler.step()
    
    
    logger.info('[Train] Iter %04d | Loss %.6f | Grad %.4f ' % (it, loss.item(), orig_grad_norm))
    writer.add_scalar('train/loss', loss, it)
    writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], it)
    writer.add_scalar('train/grad_norm', orig_grad_norm, it)
    writer.flush()

def validate_loss(it):

    all_refs = []
    all_recons = []
    for i, batch in enumerate(tqdm(test_loader, desc='Validate')):
        if args.num_val_batches > 0 and i >= args.num_val_batches:
            break
        # Get Layer Data
        ref, shift, scale = get_layer_data(batch, layer_no= args.layer_no)
        
        with torch.no_grad():
            model.eval()
            code = model.encode(ref)
            recons = model.decode(code, ref.size(1), flexibility=args.flexibility)
        all_refs.append(ref * scale + shift)
        all_recons.append(recons * scale + shift)

    all_refs = torch.cat(all_refs, dim=0)
    all_recons = torch.cat(all_recons, dim=0)
    metrics = EMD_CD(all_recons, all_refs, batch_size=args.val_batch_size)
    cd, emd = metrics['MMD-CD'].item(), metrics['MMD-EMD'].item()
    
    logger.info('[Val] Iter %04d | CD %.6f | EMD %.6f  ' % (it, cd, emd))
    writer.add_scalar('val/cd', cd, it)
    writer.add_scalar('val/emd', emd, it)
    writer.flush()

    return cd

def validate_inspect(it):
    sum_n = 0
    sum_chamfer = 0
    for i, batch in enumerate(tqdm(test_loader, desc='Inspect')):
        x, shift, scale = get_layer_data(batch, layer_no= args.layer_no)
        model.eval()
        code = model.encode(x)
        recons = model.decode(code, x.size(1), flexibility=args.flexibility).detach()

        sum_n += x.size(0)
        if i >= args.num_inspect_batches:
            break   # Inspect only 5 batch

    writer.add_mesh('val/pointcloud', recons[:args.num_inspect_pointclouds], global_step=it)
    writer.flush()

# Main loop
logger.info('Start training...')
try:
    it = 1
    while it <= args.max_iters:
        train(it)
        if it % args.val_freq == 0 or it == args.max_iters:
            with torch.no_grad():
                cd_loss = validate_loss(it)
                validate_inspect(it)
            opt_states = {
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }
            ckpt_mgr.save(model, args, cd_loss, opt_states, step=it)
        it += 1

except KeyboardInterrupt:
    logger.info('Terminating...')

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

from models.dgcnn import DGCNN

# Arguments
parser = argparse.ArgumentParser()
# Model arguments
parser.add_argument('--latent_dim', type=int, default=512)
parser.add_argument('--num_steps', type=int, default=256)
parser.add_argument('--beta_1', type=float, default=1e-4)
parser.add_argument('--beta_T', type=float, default=0.05)
parser.add_argument('--sched_mode', type=str, default='linear')
parser.add_argument('--flexibility', type=float, default=0.0)
parser.add_argument('--residual', type=eval, default=True, choices=[True, False])
parser.add_argument('--resume', type=str, default=None)

# Datasets and loaders
parser.add_argument('--dataset_path', type=str, default='./data/shapenet.hdf5')
parser.add_argument('--categories', type=str_list, default=['airplane'])
parser.add_argument('--scale_mode', type=str, default='shape_unit')
parser.add_argument('--train_batch_size', type=int, default=64)
parser.add_argument('--val_batch_size', type=int, default=64)
parser.add_argument('--rotate', type=eval, default=False, choices=[True, False])
parser.add_argument('--train_witerator', type=bool, default=True) # Whether use iterator or epoch based training

# Optimizer and scheduler
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--max_grad_norm', type=float, default=10)
parser.add_argument('--end_lr', type=float, default=1e-4)
parser.add_argument('--sched_start_epoch', type=int, default=128*THOUSAND)
parser.add_argument('--sched_end_epoch', type=int, default=256*THOUSAND)

# Training
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--logging', type=eval, default=False, choices=[True, False])
parser.add_argument('--log_root', type=str, default='./logs_ae')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--max_iters', type=int, default=float('inf'))
parser.add_argument('--val_freq', type=float, default=1000) # default=1000
parser.add_argument('--tag', type=str, default=None)
parser.add_argument('--num_val_batches', type=int, default=-1)
parser.add_argument('--num_inspect_batches', type=int, default=1)
parser.add_argument('--num_inspect_pointclouds', type=int, default=4)
parser.add_argument('--input_type', type=str, default='x1', choices=["x1", "x2", "x3", "x4", "original"])  # "x1", "x2", "x3", "x4", "original"
parser.add_argument('--random_input_layer', type=bool, default=False, choices=[True, False])
parser.add_argument('--train_mode', type=str, default='train_ae_DGCNN_layers')# 'train_ae_default','train_ae_DGCNN_layers', 'train_layers'


args = parser.parse_args()
seed_all(args.seed)
# torch.backends.cudnn.enabled = False

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
transform = None
if args.rotate:
    transform = RandomRotate(180, ['pointcloud'], axis=1)
logger.info('Transform: %s' % repr(transform))
logger.info('Loading datasets...')


if args.train_mode == "train_ae_default":
    train_dset = ShapeNetCore(
        path=args.dataset_path,
        cates=args.categories,
        split='train',
        scale_mode=args.scale_mode,
        transform=transform,
    )
    val_dset = ShapeNetCore(
        path=args.dataset_path,
        cates=args.categories,
        split='val',
        scale_mode=args.scale_mode,
        transform=transform,
    )
    train_iter = get_data_iterator(DataLoader(
        train_dset,
        batch_size=args.train_batch_size,
        num_workers=0,
    ))
    val_loader = DataLoader(val_dset, batch_size=args.val_batch_size, num_workers=0)
    
elif args.train_mode == "train_ae_DGCNN_layers":
    layer_name = args.input_type
    if layer_name == "original":
        standardize = False
    else:
        standardize = True

    train_dset = ModelNet40(num_points=1024, partition='train')
    val_dset = ModelNet40(num_points=1024, partition='val')

    if args.train_witerator:
        train_iter = get_data_iterator(DataLoader(
            train_dset,
            batch_size=args.train_batch_size,
            num_workers=0,
        ))
    else:
        train_iter = DataLoader(train_dset, batch_size=args.val_batch_size, num_workers=0)
    val_loader = DataLoader(val_dset, batch_size=args.val_batch_size, num_workers=0)
    
    dgcnn_model =  DGCNN(mode="return_layers")
    model_dict = torch.load("model_best_test.pth")["model_state"]
    model_keys = list(model_dict.keys())
    for key in model_keys:
        if key.startswith("model."):
            keyname = key[6:]
            model_dict[keyname] = model_dict.pop(key)
        else:
            _ = model_dict.pop(key)

    dgcnn_model.load_state_dict(model_dict)
    dgcnn_model.cuda()
    dgcnn_model.eval()
    
else:
    layer_name = args.input_name
    if layer_name == "original":
        standardize = False
    else:
        standardize = True

    train_dset = Layer1(layer_name, standardize=standardize)
    val_dset = Layer1(layer_name, split="val", standardize=standardize)

    train_iter = get_data_iterator(DataLoader(
        train_dset,
        batch_size=args.train_batch_size,
        num_workers=0,
    ))

    val_loader = DataLoader(val_dset, batch_size=args.val_batch_size, num_workers=0)
    
    # val_data = next(iter(val_loader))



# Model
layer_dict = {"x1":64, "x2":64, "x3":128, "x4":256, "original":3}
layer_dim = layer_dict[layer_name]
if args.random_input_layer: layer_dim=256

logger.info('Building model...')
if args.resume is not None:
    logger.info('Resuming from checkpoint...')
    ckpt = torch.load(args.resume)
    model = AutoEncoder(ckpt['args'], layer_dim=layer_dim).to(args.device)
    model.load_state_dict(ckpt['state_dict'])
else:
    model = AutoEncoder(args, layer_dim=layer_dim).to(args.device)
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


class Identity_c(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, t=5, layer_name="original"):
        return x
    def denoise_layer(self, x, t=5, layer_name="original"):
        return x

# denoiser = Denoiser(args=args, layer_dim=layer_dim)
denoiser_list = nn.ModuleList([
    Identity_c(),                  # Input
    Identity_c(),                  # Layer1
    Identity_c(),                  # Layer2
    Identity_c(),                  # Layer3
    Identity_c(),                  # Layer4
])

# Train, validate
def train(it):
    # Load data
    batch = next(train_iter)
    x = batch[0].to(args.device)
    if args.train_mode == "train_ae_DGCNN_layers":
        with torch.no_grad():
            if args.random_input_layer: 
                global layer_name
                layer_name = np.random.choice(["x1","x2","x3","x4"], p=[0.25,0.25,0.25,0.25])
            _, layers = dgcnn_model.forward(x.permute((0,2,1)), denoiser=denoiser_list, t=1, layer_name=layer_name)
            x = layers[layer_name]
            #Zeropad
            if args.random_input_layer: x = F.pad(x, (0,0, 0,layer_dim-x.size(1)))
            if standardize:
                x -= torch.mean(x, axis=(2), keepdim=True)
                x /= torch.std(x, axis=(1,2), keepdim=True)
    
    # Reset grad and model state
    optimizer.zero_grad()
    model.train()

    # Forward
    loss = model.get_loss(x, layer_name=layer_name)

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
    return loss



def validate_loss(it):

    all_refs = []
    all_recons = []
    for i, batch in enumerate(tqdm(val_loader, desc='Validate')):
        if args.num_val_batches > 0 and i >= args.num_val_batches:
            break
        ref = batch[0].to(args.device).permute((0,2,1))
        if args.train_mode == "train_ae_DGCNN_layers":
            if args.random_input_layer: 
                global layer_name
                layer_name = np.random.choice(["x1","x2","x3","x4"], p=[0.25,0.25,0.25,0.25])
            with torch.no_grad():
                _, layers = dgcnn_model.forward(ref, denoiser=denoiser_list, t=1, layer_name=layer_name)
                ref = layers[layer_name]
            #Zeropad
            if args.random_input_layer: ref = F.pad(ref, (0,0, 0,layer_dim-ref.size(1)))
            if standardize:
                ref -= torch.mean(ref, axis=(2), keepdim=True)
                ref /= torch.std(ref, axis=(1,2), keepdim=True)
            # shift = batch['shift'].to(args.device)
        # scale = batch['scale'].to(args.device)
        with torch.no_grad():
            model.eval()
            code = model.encode(ref)
            ref = ref.permute((0,2,1))
            recons = model.decode(code, ref.size(1), flexibility=args.flexibility, layer_name=layer_name)
        # all_refs.append(ref * scale + shift)
        all_refs.append(ref)
        # all_recons.append(recons * scale + shift)
        all_recons.append(recons)
    
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
    for i, batch in enumerate(tqdm(val_loader, desc='Inspect')):
        x = batch[0].to(args.device).permute((0,2,1))
        if args.train_mode == "train_ae_DGCNN_layers":
            if args.random_input_layer: 
                global layer_name
                layer_name = np.random.choice(["x1","x2","x3","x4"], p=[0.25,0.25,0.25,0.25])
            with torch.no_grad():
                _, layers = dgcnn_model.forward(x, denoiser=denoiser_list, t=1, layer_name=layer_name)
                x = layers[layer_name]
            #Zeropad
            if args.random_input_layer: x = F.pad(x, (0,0, 0,layer_dim-x.size(1)))
            if standardize:
                x -= torch.mean(x, axis=(2), keepdim=True)
                x /= torch.std(x, axis=(1,2), keepdim=True)
        model.eval()
        code = model.encode(x)
        x = x.permute((0,2,1))
        recons = model.decode(code, x.size(1), flexibility=args.flexibility, layer_name=layer_name).detach()

        sum_n += x.size(0)
        if i >= args.num_inspect_batches:
            break   # Inspect only 5 batch

    writer.add_mesh('val/pointcloud', recons[:args.num_inspect_pointclouds], global_step=it)
    writer.flush()

# Main loop
logger.info('Start training...')
try:
    it = 1
    epoch_no = 0
    epoch_train_loss_list = []
    epoch_train_loss = 0
    while it <= args.max_iters:
        epoch_train_loss += train(it)
        
        # Epoch loop
        if (it-1) % (len(train_dset) // args.train_batch_size) and it!=1:
            epoch_train_loss_list.append(epoch_train_loss)
            logger.info('------ [Train] Epoch %04d | Loss %.6f ------' % (epoch_no, epoch_train_loss))
            epoch_no += 1
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

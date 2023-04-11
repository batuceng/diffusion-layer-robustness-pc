import torch
from torch.nn import Module

from .encoders import *
from .diffusion import *


class AutoEncoder(Module):

    def __init__(self, args, layer_dim=3):
        super().__init__()
        self.args = args
        self.layer_dim = layer_dim
        self.encoder = PointNetEncoder(zdim=args.latent_dim, input_dim=self.layer_dim)
        self.diffusion = DiffusionPoint(
            net = PointwiseNet(point_dim=self.layer_dim, context_dim=args.latent_dim, residual=args.residual),
            var_sched = VarianceSchedule(
                num_steps=args.num_steps,
                beta_1=args.beta_1,
                beta_T=args.beta_T,
                mode=args.sched_mode
            )
        )

    def encode(self, x):
        """
        Args:
            x:  Point clouds to be encoded, (B, N, d).
        """
        code, _ = self.encoder(x)
        return code

    def decode(self, code, num_points, flexibility=0.0, ret_traj=False, layer_name="original"):
        return self.diffusion.sample(num_points, code, self.layer_dim, flexibility=flexibility, ret_traj=ret_traj, layer_name=layer_name)

    def get_loss(self, x, layer_name):
        code = self.encode(x)
        loss = self.diffusion.get_loss(x, code, t=None, layer_name=layer_name)
        return loss

    def denoiser(self, pointcloud, t, flexibility=0.0, ret_traj=False, context=None, layer_name="original"):
        return self.diffusion.truncated_sample(pointcloud, t, context=context, layer_name=layer_name)
    
    # Do the denoising
    def denoise_layer(self, x, t=5, layer_name="original"):
        #Normalize
        x = x.transpose(1, 2)
        mean = torch.mean(x, axis=1, keepdim=True)
        x -= mean
        std = torch.std(x, axis=(1,2), keepdim=True)
        x /= std
        
        # Denoise
        code = self.encode(x.transpose(1,2))
        x = self.denoiser(x, t, context=code, layer_name="original")
        
        # Denormalize
        x *= std
        x += mean
        x = x.transpose(1, 2)
        return x
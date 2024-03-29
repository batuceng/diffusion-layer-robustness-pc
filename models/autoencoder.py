import torch
from torch.nn import Module

from .encoders import *
from .diffusion import *
import time

class AutoEncoder(Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = PointNetEncoder(zdim=args.latent_dim, input_dim=args.input_dim)
        self.diffusion = DiffusionPoint(
            net = PointwiseNet(point_dim=args.input_dim, context_dim=args.latent_dim, residual=args.residual),
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

    def decode(self, code, num_points, flexibility=0.0, ret_traj=False):
        return self.diffusion.sample(num_points, code, flexibility=flexibility, 
                                     ret_traj=ret_traj, point_dim=self.args.input_dim)
    
    # Denoising function, N-step backward diffusion
    def denoiser(self, pointcloud, t, flexibility=0.0, ret_traj=False, context=None):
        return self.diffusion.truncated_sample(pointcloud, t, context=context)

    def get_loss(self, x):
        # start = time.time()
        code = self.encode(x)
        # encoding = time.time()
        # print(f"Encoding: {encoding-start}")
        loss = self.diffusion.get_loss(x, code)
        # getting_loss = time.time()
        # print(f"Getting loss: {getting_loss-encoding}")
        
        return loss
    
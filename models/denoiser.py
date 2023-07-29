import torch
import torch.nn as nn
from typing import List
from autoencoder import AutoEncoder

class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()
        pass
    
    def forward(self, data, *args, **kwargs):
        return data

class Layer_Denoiser(nn.Module):
    def __init__(self, model:AutoEncoder, t:int):
        super(Layer_Denoiser, self).__init__()
        self.model = model
        self.t = t
    
    def forward(self, data:torch.tensor, layer:int) -> torch.tensor:
        if layer == self.model.args.layer_no:
            with torch.no_grad():
                data = data.permute((0,2,1))
                data, shift, scale = self.normalize_layer_data(data)
                code = self.model.encode(data)
                # Changed Encode -> Denoiser to call truncated_sample()
                recons = self.model.denoiser(data, t=self.t, flexibility=self.model.args.flexibility, context=code).detach()
                recons = recons * scale + shift
            return recons.permute((0,2,1))
        else:
            return data
    
    @staticmethod
    def normalize_layer_data(data:torch.tensor, normalization="unit_shape"):
        # if normalization == "unit_shape":
            # data.shape = [B, N, D]
        shift = data.mean(dim=1, keepdim=True) # (B, 1, D)
        scale = data.std(dim=[1,2], keepdim=True) # (B, 1, 1)
        data = (data - shift) / scale # Normalize layer data
        return data, shift, scale


class Multiple_Layer_Denoiser(nn.Module):
    def __init__(self, models:List[AutoEncoder], t_list:List[int]):
        super(Multiple_Layer_Denoiser, self).__init__()
        self.models = models
        self.layers = [m.args.layer_no for m in models]
        self.t_list = t_list
    
    def forward(self, data:torch.tensor, layer:int) -> torch.tensor:
        if layer in self.layers:
            idx = self.layers.index(layer)
            model = self.models[idx]
            if self.t_list[idx] != 0:
                with torch.no_grad():
                    data = data.permute((0,2,1))
                    data, shift, scale = self.normalize_layer_data(data)
                    code = model.encode(data)
                    # Changed Encode -> Denoiser to call truncated_sample()
                    recons = model.denoiser(data, t=self.t_list[idx], flexibility=model.args.flexibility, context=code).detach()
                    recons = recons * scale + shift
                return recons.permute((0,2,1))
            else:
                return data
        else:
            return data
    
    @staticmethod
    def normalize_layer_data(data:torch.tensor, normalization="unit_shape"):
        # if normalization == "unit_shape":
            # data.shape = [B, N, D]
        shift = data.mean(dim=1, keepdim=True) # (B, 1, D)
        scale = data.std(dim=[1,2], keepdim=True) # (B, 1, 1)
        data = (data - shift) / scale # Normalize layer data
        return data, shift, scale
        
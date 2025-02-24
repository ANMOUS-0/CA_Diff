import torch, math
import torch.nn as nn
from einops import rearrange
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
from typing import Union, Type, List, Tuple


def zero_module(module, flag:bool=True):
    """
    Zero out the parameters of a module and return it.
    """
    if flag:
        for p in module.parameters():
            p.detach().zero_()
    return module



def make_zero_conv(channels, conv_op:Type[_ConvNd], use_zero_init:bool):
    return zero_module(conv_op(channels, channels, 1, padding=0), 
                        flag=use_zero_init)


class LearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with learned sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered
    


class Embedding(nn.Module):

    def __init__(self,**kwargs) -> None:
        super().__init__()

        self.embedding = nn.Embedding(**kwargs)

    def forward(self,x):
        x = torch.squeeze(x, dim=1)
        x = self.embedding(x).permute(0,-1,1,2,3).contiguous()
        return x


class TimeStepEmbedding(nn.Module):

    def __init__(self, dim, max_period=10000):
        super().__init__()
        self.dim = dim
        self.half = dim // 2
        self.freqs = torch.exp(
            -math.log(max_period) * \
                torch.arange(start=0, end=self.half, dtype=torch.float32) / self.half)
        

    def forward(self, timesteps):
        self.freqs = self.freqs.to(timesteps.device)
        args = timesteps[:, None].float() * self.freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding
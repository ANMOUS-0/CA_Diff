import torch
from torch import nn
from torch.nn.parameter import Parameter
from functools import partial
from itertools import repeat as it_repeat
import collections.abc
from .dynamic_conv import attention1d
import torch.nn.functional as F
import numpy as np



class eca_layer(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)
        


class ECA_Dyn_Time(nn.Module):
    def __init__(self, feat_dim, k_size:int=3, time_dim:int=None,
                 dilate_rate=None,dyn_K:int=None):
        super().__init__()

        # eca modules;
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.eca_conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.eca_sigmoid = nn.Sigmoid()

        dyn_parameters = {'kernel_size':3, 'in_planes':1, 'out_planes':1}
        self.dyn_conv = ASPP_DynamicConv1d_Refer(
            reference_dim=time_dim,
            dilate_rate=dilate_rate,
            K=dyn_K,
            **dyn_parameters,
        )
        self.final_conv = nn.Sequential(
            nn.InstanceNorm1d(int(1+len(dilate_rate))),
            nn.Conv1d(int(1+len(dilate_rate)),1,1)
        )

    def forward(self, x, time):
        y = self.avg_pool(x)
        y = y.squeeze(-1).squeeze(-1).transpose(-1,-2)

        y_eca = self.eca_conv(y)
        y_dyn = self.dyn_conv(time, y)

        y = self.final_conv(torch.concat([y_eca, y_dyn], dim=1))
        
        y = (y).transpose(-1, -2).unsqueeze(-1).unsqueeze(-1)
        y = self.eca_sigmoid(y)
        y = y.expand_as(x)

        result = x + x * y
        return result



class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x



# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(it_repeat(x, n))
    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple




class Dynamic_conv1d_refer(nn.Module):
    def __init__(self, reference_planes,
                 in_planes, out_planes, kernel_size, ratio=0.25, stride=1, padding=0, dilation=1, groups=1, bias=True, K=4,temperature=34, init_weight=True):
        super().__init__()
        assert in_planes%groups==0
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.K = K
        self.attention = attention1d(reference_planes, ratio, K, temperature)

        self.weight = nn.Parameter(torch.randn(K, out_planes, in_planes//groups, kernel_size), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.zeros(K, out_planes))
        else:
            self.bias = None
        if init_weight:
            self._initialize_weights()

        #TODO 初始化
    def _initialize_weights(self):
        for i in range(self.K):
            nn.init.kaiming_uniform_(self.weight[i])


    def update_temperature(self):
        self.attention.updata_temperature()

    def forward(self, reference, x):#将batch视作维度变量，进行组卷积，因为组卷积的权重是不同的，动态卷积的权重也是不同的
        softmax_attention = self.attention(reference)
        batch_size, in_planes, height = x.size()
        x = x.view(1, -1, height, )# 变化成一个维度进行组卷积
        weight = self.weight.view(self.K, -1)

        # 动态卷积的权重的生成， 生成的是batch_size个卷积参数（每个参数不同）
        aggregate_weight = torch.mm(softmax_attention, weight).view(batch_size*self.out_planes, self.in_planes//self.groups, self.kernel_size,)
        if self.bias is not None:
            aggregate_bias = torch.mm(softmax_attention, self.bias).view(-1)
            output = F.conv1d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups*batch_size)
        else:
            output = F.conv1d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * batch_size)

        output = output.view(batch_size, self.out_planes, output.size(-1))
        return output
    

class ASPP_DynamicConv1d_Refer(nn.Module):
    def __init__(self,
                 reference_dim,
                 dilate_rate,
                 K,
                 kernel_size,
                 in_planes,
                 out_planes):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * len(dilate_rate)
        pads = tuple(same_padding(k, d) for k, d in zip(kernel_size, dilate_rate))
        self.convs = nn.ModuleList()
        for k, d, p in zip(kernel_size, dilate_rate, pads):
            _conv = Dynamic_conv1d_refer(
                reference_planes=reference_dim,
                in_planes=in_planes,
                out_planes=out_planes,
                kernel_size=k,
                padding=p,
                dilation=d,
                K=K
            )
            self.convs.append(_conv)

        # self.final_conv = nn.Sequential(
        #     nn.InstanceNorm1d(int(out_planes*3)),
        #     nn.Conv1d(int(out_planes*3), out_channels=out_planes,
        #               kernel_size=1, stride=1)
        # )

    def forward(self, reference, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: in shape (batch, channel, spatial_1[, spatial_2, ...]).
        """
        reference = reference.unsqueeze(dim=-1)
        x_out = torch.cat([conv(reference, x) for conv in self.convs], dim=1)
        # x_out = self.final_conv(x_out)
        return x_out


from typing import List, Optional, Sequence, Tuple, Union



def same_padding(
    kernel_size: Union[Sequence[int], int], dilation: Union[Sequence[int], int] = 1
) -> Union[Tuple[int, ...], int]:
    """
    Return the padding value needed to ensure a convolution using the given kernel size produces an output of the same
    shape as the input for a stride of 1, otherwise ensure a shape of the input divided by the stride rounded down.

    Raises:
        NotImplementedError: When ``np.any((kernel_size - 1) * dilation % 2 == 1)``.

    """

    kernel_size_np = np.atleast_1d(kernel_size)
    dilation_np = np.atleast_1d(dilation)

    if np.any((kernel_size_np - 1) * dilation % 2 == 1):
        raise NotImplementedError(
            f"Same padding not available for kernel_size={kernel_size_np} and dilation={dilation_np}."
        )

    padding_np = (kernel_size_np - 1) / 2 * dilation_np
    padding = tuple(int(p) for p in padding_np)

    return padding if len(padding) > 1 else padding[0]
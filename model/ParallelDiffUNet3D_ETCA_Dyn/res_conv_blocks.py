from typing import Tuple, List, Union, Type

import numpy as np
import torch.nn
from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
import torch.nn.functional as F

from ...building_blocks.helper import maybe_convert_scalar_to_list


from .util import zero_module

from copy import deepcopy

from .attention_blocks import DiTBlock_CA
from .eca_blocks import ECA_Dyn_Time


def normalization(channels):
    """
    Make a standard normalization layer.
    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return torch.nn.InstanceNorm3d(channels,affine=True)


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, 
                 channels, 
                 conv_op: Type[_ConvNd],
                 use_conv:bool=False, 
                 out_channels=None, 
                 dims=3,
                 padding=1,
                 stride=1,
                 kernel=3):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        self.stride = stride
        if use_conv:
            self.conv = conv_op(self.channels, self.out_channels, kernel, padding=padding)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2] * self.stride[0], 
                    x.shape[3] * self.stride[1], 
                    x.shape[4] * self.stride[2]), 
                    mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, 
                 channels, 
                 conv_op: Type[_ConvNd],
                 use_conv:bool=False, 
                 out_channels=None,
                 padding=1,
                 stride=1,
                 kernel=3):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.stride = stride
        if use_conv:
            self.op = conv_op(
                self.channels, self.out_channels, kernel, stride=stride, padding=padding
            )
        else:
            assert self.channels == self.out_channels
            self.op = nn.AvgPool3d(kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)




class ResConvBlock(nn.Module):
    def __init__(self,
                 conv_op: Type[_ConvNd],
                 input_channels: int,
                 output_channels: Union[int, List[int], Tuple[int, ...]],
                 kernel_size: Union[int, List[int], Tuple[int, ...]],
                 initial_stride: Union[int, List[int], Tuple[int, ...]],
                 time_embedding: bool = False,
                 time_embedding_channel: int = None,
                 use_skip_conv: bool = False,
                 up: bool = False,
                 down: bool = False,
                 zero_conv: bool=True,
                 use_scale_shift_norm:bool=False,
                 use_coord:bool=False
                 ):
        """

        :param conv_op:
        :param input_channels:
        :param output_channels: can be int or a list/tuple of int. If list/tuple are provided, each entry is for
        one conv. The length of the list/tuple must then naturally be num_convs
        :param kernel_size:
        :param initial_stride:
        :param conv_bias:
        :param norm_op:
        :param norm_op_kwargs:
        :param dropout_op:
        :param dropout_op_kwargs:
        :param nonlin:
        :param nonlin_kwargs:
        :param zero_conv: use zero conv in the out layer of residual conv.
        """
        super().__init__()
        self.time_embedding = time_embedding
        # if not isinstance(output_channels, (tuple, list)):
        #     output_channels = [output_channels] * num_convs

        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(input_channels),
            nn.SiLU(),
            conv_op(input_channels,output_channels,3,padding=1)
        )

        self.updown = up or down

        if not all(np.unique(initial_stride)==1) or len(np.unique(initial_stride))>1:
            if up:
                self.h_upd = Upsample(
                    channels=input_channels, 
                    conv_op=conv_op,
                    use_conv=False,
                    stride=initial_stride,
                    kernel=kernel_size)
                self.x_upd = Upsample(
                    channels=input_channels, 
                    conv_op=conv_op,
                    use_conv=False,
                    stride=initial_stride,
                    kernel=kernel_size)
            elif down:
                self.h_upd = Downsample(
                    channels=input_channels, 
                    conv_op=conv_op,
                    use_conv=False,
                    stride=initial_stride,
                    kernel=kernel_size)
                self.x_upd = Downsample(
                    channels=input_channels, 
                    conv_op=conv_op,
                    use_conv=False,
                    stride=initial_stride,
                    kernel=kernel_size)
        else:
            self.h_upd = self.x_upd = nn.Identity()
        
        self.use_coord = use_coord
        if time_embedding:
            time_embedding_output_channel = output_channels * 2 if self.use_scale_shift_norm else output_channels
            if use_coord:
                time_embedding_output_channel = int(time_embedding_output_channel // 2)
                self.emb_layers_1 = nn.Sequential(
                    torch.nn.SiLU(),
                    torch.nn.Linear(time_embedding_channel, 
                                    time_embedding_output_channel)
                )
                # self.fuse_layer = torch.nn.Conv3d(int(2*time_embedding_output_channel),
                #                                   int(2*time_embedding_output_channel),
                #                                   kernel_size=1)
            self.emb_layers = nn.Sequential(
                torch.nn.SiLU(),
                torch.nn.Linear(time_embedding_channel, 
                                time_embedding_output_channel)
            )

        # 可以使用zero_module；
        self.out_layers = nn.Sequential(
            normalization(output_channels),
            nn.SiLU(),
            # nn.Dropout(p=dropout_op_kwargs),
            zero_module(
                conv_op(output_channels, output_channels, 3, padding=1)
            ) if zero_conv else \
                conv_op(output_channels, output_channels, 3, padding=1),
        )

        if output_channels == input_channels:
            self.skip_connection = nn.Identity()
        elif use_skip_conv:
            self.skip_connection = conv_op(
                input_channels, output_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_op(input_channels, output_channels, 1)

    def forward(self, x, label_time=None, coord_time=None):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)

        if self.time_embedding:
            assert label_time is not None, "Label_time to be embedded is None."
            embedded_label_t = self.emb_layers(label_time)
            while len(embedded_label_t.shape) < len(x.shape):
                embedded_label_t = embedded_label_t[..., None]
            if self.use_coord:
                assert coord_time is not None, f"Missing coordinate time!"
                embedded_coord_t = self.emb_layers_1(coord_time)
                while len(embedded_coord_t.shape) < len(x.shape):
                    embedded_coord_t = embedded_coord_t[..., None]

            if self.use_scale_shift_norm:
                out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
                scale, shift = torch.chunk(embedded_label_t, 2, dim=1)
                if self.use_coord:
                    scale_1, shift1 = torch.chunk(embedded_coord_t, 2, dim=1)
                    scale = torch.cat([scale, scale_1], dim=1)
                    shift = torch.cat([shift, shift1], dim=1)
                h = out_norm(h) * (1 + scale) + shift
                h = out_rest(h)
            else:
                embedded_t = embedded_label_t
                if self.use_coord:
                    embedded_t = torch.cat([embedded_label_t, embedded_coord_t], dim=1)
                h = h + embedded_t
                h = self.out_layers(h)

        else:
            h = self.out_layers(h)

        return self.skip_connection(x) + h




class ResConvEncoder(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 return_skips: bool = False,
                 nonlin_first: bool = False,
                 pool: str = None,
                 time_embedding: bool = False,
                 time_embedding_channel: int = None,
                 fuse_method=None,
                 contain_middle:bool=True,
                 conv_resample:bool = True,
                 use_zero_module: bool=False,
                 use_zero_module_in_res_block: bool=True,
                 zero_module_func=None,
                 use_scale_shift_norm:bool=False,
                 use_coord:bool=False,
                 ):
        '''
        Changes in this version:
        The zero_conv module is called after both conv and downsample, which lead to a shared module.
        ControlNet use separate zero_conv module for conv and downsample. If we don't want so many
        layers like ControlNet, we should use sigle conv module after down sample, which is:
        input --> conv --> downsample --> zero_conv.
        Actually, this is our first design but we wrongly wrote the code, shit! 

        :param use_zero_module: use zero conv at the end of each layer module.
        :param use_zero_module_in_res_block: use zero conv inside each ResConvBlock.
        
        '''

        super().__init__()
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * n_stages
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(strides, int):
            strides = [strides] * n_stages
        assert len(kernel_sizes) == n_stages, "kernel_sizes must have as many entries as we have resolution stages (n_stages)"
        assert len(n_conv_per_stage) == n_stages, "n_conv_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert len(features_per_stage) == n_stages, "features_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert len(strides) == n_stages, "strides must have as many entries as we have resolution stages (n_stages). " \
                                             "Important: first entry is recommended to be 1, else we run strided conv drectly on the input"
        # time embedding settings;
        if time_embedding:
            assert time_embedding_channel is not None, \
                "If you want to use time embedding, you should offer time embedding channel."

        self.fuse_method = fuse_method

        self.use_zero_module = use_zero_module
        self.zero_module_func = zero_module_func
        if self.use_zero_module and (self.zero_module_func is None):
            ValueError('If you want to apply zero convolution, you should pass the related function!')


        stages = []
        zero_conv_list = []
        for s in range(n_stages):
            stage_modules = []
            conv_stride = strides[s]
            downsample_flag = False

            if np.cumprod(conv_stride)[-1] != 1:
                downsample_flag = True

            stage_modules.append(
                ResConvBlock(
                conv_op=conv_op,
                input_channels=input_channels,
                output_channels=features_per_stage[s],
                kernel_size=kernel_sizes[s],
                initial_stride=[1,1,1],
                time_embedding=time_embedding,
                time_embedding_channel=time_embedding_channel,
                use_skip_conv=False,
                zero_conv=use_zero_module_in_res_block,
                use_scale_shift_norm=use_scale_shift_norm,
                use_coord=use_coord
            )
            )

            input_channels = features_per_stage[s]

            for conv_num in range(1,n_conv_per_stage[s]):
                stage_modules.append(
                ResConvBlock(
                conv_op=conv_op,
                input_channels=input_channels,
                output_channels=input_channels,
                kernel_size=kernel_sizes[s],
                initial_stride=[1,1,1],
                time_embedding=time_embedding,
                time_embedding_channel=time_embedding_channel,
                use_skip_conv=False,
                zero_conv=use_zero_module_in_res_block,
                use_scale_shift_norm=use_scale_shift_norm,
                use_coord=use_coord
                )
                )

            if downsample_flag:
                stage_modules.append(
                    ResConvBlock(
                    conv_op=conv_op,
                    input_channels=input_channels,
                    output_channels=input_channels,
                    kernel_size=kernel_sizes[s],
                    initial_stride=conv_stride,
                    time_embedding=time_embedding,
                    time_embedding_channel=time_embedding_channel,
                    use_skip_conv=False,
                    down=True,
                    zero_conv=use_zero_module_in_res_block,
                    use_scale_shift_norm=use_scale_shift_norm,
                    use_coord=use_coord
                    ) if pool == 'res' else
                    Downsample(
                    conv_op=conv_op,
                    channels=input_channels,
                    stride=conv_stride,
                    use_conv=conv_resample
                    )
                )

            stages.append(nn.ModuleList(stage_modules))

            # The third parameter in the function will decide whether the conv is initialized as zero.
            # If not, it will return a normal conv block which is initialized randomly.
            zero_conv_list.append(
                self.zero_module_func(features_per_stage[s], conv_op, self.use_zero_module)
                )


        if contain_middle:
            middle_blocks = []
            for i in range(n_conv_per_stage[-1]):
                middle_blocks.append(
                    ResConvBlock(
                    conv_op=conv_op,
                    input_channels=features_per_stage[-1],
                    output_channels=features_per_stage[-1],
                    kernel_size=kernel_sizes[-1],
                    initial_stride=[1,1,1],
                    time_embedding=time_embedding,
                    time_embedding_channel=time_embedding_channel,
                    use_skip_conv=False,
                    zero_conv=use_zero_module_in_res_block,
                    use_scale_shift_norm=use_scale_shift_norm
                    )
                )
            stages.append(nn.ModuleList(middle_blocks))
            # The third parameter in the function will decide whether the conv is initialized as zero.
            # If not, it will return a normal conv block which is initialized randomly.
            zero_conv_list.append(
                self.zero_module_func(features_per_stage[-1], conv_op, self.use_zero_module)
                )


        self.stages = nn.ModuleList(stages)
        self.zero_convs = nn.ModuleList(zero_conv_list)
        self.output_channels = deepcopy(features_per_stage)
        self.strides = [maybe_convert_scalar_to_list(conv_op, i) for i in strides]

        self.return_skips = return_skips

        # we store some things that a potential decoder needs
        self.conv_op = deepcopy(conv_op)
        self.norm_op = deepcopy(norm_op)
        self.norm_op_kwargs = deepcopy(norm_op_kwargs)
        self.nonlin = deepcopy(nonlin)
        self.nonlin_kwargs = deepcopy(nonlin_kwargs)
        self.dropout_op = deepcopy(dropout_op)
        self.dropout_op_kwargs = deepcopy(dropout_op_kwargs)
        self.conv_bias = deepcopy(conv_bias)
        self.kernel_sizes = deepcopy(kernel_sizes)


        if contain_middle:
            if len(kernel_sizes)==1:
                kernel_sizes = [kernel_sizes]*3
            self.kernel_sizes = self.kernel_sizes+kernel_sizes[-1]
            self.output_channels = self.output_channels + [features_per_stage[-1]]
            self.strides = self.strides + [[1,1,1]]

    def forward(self, x, label_t=None, coord_t=None):
        # TODO: Operations related to control features.
        ret = []

        for i,(s,zero_conv) in enumerate(zip(self.stages, self.zero_convs)):
            for sub_module in s:
                if isinstance(sub_module, ResConvBlock):
                    x = sub_module(x, label_time=label_t, coord_time=coord_t)
                else:
                    x = sub_module(x)
            # The zero conv module is called at the end of each layer.
            x = zero_conv(x)
            ret.append(x)

        if self.return_skips:
            return ret
        else:
            return ret[-1]




class ResConvDecoder(nn.Module):
    def __init__(self,
                 encoder: ResConvEncoder,
                 num_classes: int,
                 n_conv_per_stage: Union[int, Tuple[int, ...], List[int]],
                 deep_supervision, 
                 nonlin_first: bool = False,
                 time_embedding: bool = False,
                 time_embedding_channel: int = None,
                 contain_middle: bool = False,
                 upsample:str = None,
                 conv_resample: bool = True,
                 use_zero_module_in_res_block:bool=True,
                 pred_noise:bool=False,
                 use_scale_shift_norm:bool=False,
                 use_coord:bool=False,
                 coord_dim:int=None,
                 eca_time:bool=True,
                 eca_dyn_dilate_rate=None,
                 eca_dyn_K=None):
        """
        This class needs the skips of the encoder as input in its forward.

        the encoder goes all the way to the bottleneck, so that's where the decoder picks up. stages in the decoder
        are sorted by order of computation, so the first stage has the lowest resolution and takes the bottleneck
        features and the lowest skip as inputs
        the decoder has two (three) parts in each stage:
        1) conv transpose to upsample the feature maps of the stage below it (or the bottleneck in case of the first stage)
        2) n_conv_per_stage conv blocks to let the two inputs get to know each other and merge
        3) (optional if deep_supervision=True) a segmentation output Todo: enable upsample logits?
        :param encoder:
        :param num_classes:
        :param n_conv_per_stage:
        :param deep_supervision:
        """
        super().__init__()
        self.deep_supervision = deep_supervision
        self.encoder = encoder
        self.num_classes = num_classes
        n_stages_encoder = len(encoder.output_channels)
        if contain_middle:
            n_conv_per_stage = n_conv_per_stage + [n_conv_per_stage[-1]]
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * (n_stages_encoder - 1)
        assert len(n_conv_per_stage) == n_stages_encoder - 1, "n_conv_per_stage must have as many entries as we have " \
                                                          "resolution stages - 1 (n_stages in encoder - 1), " \
                                                          "here: %d" % n_stages_encoder


        # we start with the bottleneck and work out way up
        stages = []
        transpconvs = []
        seg_layers = []
        for s in range(1, n_stages_encoder):
            stage_modules = []
            input_features_below = encoder.output_channels[-s]
            input_features_skip = encoder.output_channels[-(s + 1)]
            stride_for_transpconv = encoder.strides[-(s + 1)]

            upsample_flag = False
            if np.cumprod(stride_for_transpconv)[-1] != 1:
                upsample_flag = True
            # transpconvs.append(transpconv_op(
            #     input_features_below, input_features_skip, stride_for_transpconv, stride_for_transpconv,
            #     bias=encoder.conv_bias
            # ))
            # input features to conv is 2x input_features_skip (concat input_features_skip with transpconv output)
            if s==1:
                stage_modules.append(
                    DiTBlock_CA(
                        hidden_size=input_features_skip+input_features_below,
                        num_heads=1,attn_mode='softmax-xformers',
                        context_dim=time_embedding_channel * (2 if use_coord else 1)
                    )
                )
            else:
                stage_modules.append(
                    ECA_Dyn_Time(
                        feat_dim=input_features_skip+input_features_below,
                        k_size=3,
                        time_dim=time_embedding_channel * (2 if use_coord else 1),
                        dilate_rate=eca_dyn_dilate_rate, dyn_K=eca_dyn_K
                    )
                )
            stage_modules.append(
                ResConvBlock(
                conv_op=encoder.conv_op,
                input_channels=input_features_skip+input_features_below,
                output_channels=input_features_skip,
                kernel_size=encoder.kernel_sizes[-(s + 1)],
                initial_stride=[1,1,1],
                time_embedding=time_embedding,
                time_embedding_channel=time_embedding_channel,
                use_skip_conv=False,
                zero_conv=use_zero_module_in_res_block,
                use_scale_shift_norm=use_scale_shift_norm,
                use_coord=use_coord
            )
            )

            input_channels = input_features_skip

            for conv_num in range(1,n_conv_per_stage[s-1]):
                stage_modules.append(
                ResConvBlock(
                conv_op=encoder.conv_op,
                input_channels=input_channels,
                output_channels=input_channels,
                kernel_size=encoder.kernel_sizes[-(s + 1)],
                initial_stride=[1,1,1],
                time_embedding=time_embedding,
                time_embedding_channel=time_embedding_channel,
                use_skip_conv=False,
                zero_conv=use_zero_module_in_res_block,
                use_scale_shift_norm=use_scale_shift_norm,
                use_coord=use_coord
                )
                )

            if upsample_flag:
                stage_modules.append(
                    ResConvBlock(
                    conv_op=encoder.conv_op,
                    input_channels=input_channels,
                    output_channels=input_channels,
                    kernel_size=encoder.kernel_sizes[-(s + 1)],
                    initial_stride=stride_for_transpconv,
                    time_embedding=time_embedding,
                    time_embedding_channel=time_embedding_channel,
                    use_skip_conv=False,
                    zero_conv=use_zero_module_in_res_block,
                    use_scale_shift_norm=use_scale_shift_norm,
                    use_coord=use_coord
                    ) if upsample == 'res' else
                    Upsample(
                    conv_op=encoder.conv_op,
                    channels=input_channels,
                    stride=stride_for_transpconv,
                    use_conv=conv_resample
                    )
                )

            stages.append(nn.ModuleList(stage_modules))

            # we always build the deep supervision outputs so that we can always load parameters. If we don't do this
            # then a model trained with deep_supervision=True could not easily be loaded at inference time where
            # deep supervision is not needed. It's just a convenience thing
            if self.deep_supervision:
                seg_layers.append(encoder.conv_op(input_features_skip, num_classes, 1, 1, 0, bias=True))
            elif s == n_stages_encoder-1:
                if pred_noise:
                    final_seg_module = nn.Sequential(
                        normalization(input_features_skip),
                        nn.SiLU(),
                        encoder.conv_op(input_features_skip, num_classes, 3, 1, 1, bias=True)
                    )
                else:
                    final_seg_module = nn.Sequential(
                        normalization(input_features_skip),
                        encoder.conv_op(input_features_skip, num_classes, 1, 1, 0, bias=True)
                    )
                seg_layers.append(final_seg_module)

            if use_coord:
                final_reg_module = nn.Sequential(
                    normalization(input_features_skip),
                    nn.SiLU(),
                    encoder.conv_op(input_features_skip, coord_dim, 3, 1, 1, bias=True))
                self.reg_layers = nn.ModuleList([final_reg_module])

        self.stages = nn.ModuleList(stages)
        self.transpconvs = nn.ModuleList(transpconvs)
        self.seg_layers = nn.ModuleList(seg_layers)
        self.use_coord = use_coord

    def forward(self, skips, label_t=None, coord_t=None):
        """
        we expect to get the skips in the order they were computed, so the bottleneck should be the last entry
        :param skips:
        :return:
        """
        lres_input = skips[-1]
        seg_outputs = []
        reg_outputs = []
        for s in range(len(self.stages)):
            x = lres_input
            x = torch.cat((x, skips[-(s+2)]), 1)
            for sub_module in self.stages[s]:
                if isinstance(sub_module, ResConvBlock):
                    x = sub_module(x, label_t, coord_t)
                elif isinstance(sub_module, ECA_Dyn_Time) or isinstance(sub_module, DiTBlock_CA):
                    x = sub_module(x, torch.cat([label_t, coord_t], dim=1))
                else:
                    x = sub_module(x)
            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))
            elif s == (len(self.stages) - 1):
                seg_outputs.append(self.seg_layers[-1](x))
                if self.use_coord:
                    reg_outputs.append(self.reg_layers[-1](x))
            lres_input = x

        # invert seg outputs so that the largest segmentation prediction is returned first
        seg_outputs = seg_outputs[::-1]
        # reg_outputs = reg_outputs[::-1]

        if not self.deep_supervision:
            r = seg_outputs[0]
        else:
            r = seg_outputs
        if self.use_coord: return r, reg_outputs[0]
        return r


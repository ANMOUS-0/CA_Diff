from .res_conv_blocks import *
from .util import *
from typing import Union, Type, List, Tuple
import torch
from torch.nn.modules.conv import _ConvNd
from torch import nn
from torch.nn.modules.dropout import _DropoutNd



class ParallelDiffUNet3D(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 time_embedding: bool = False,
                 time_embedding_channel: int=None,
                 fuse_method = None,
                 deep_supervision: bool = False,
                 nonlin_first: bool = False,
                 contain_middle: bool = False,
                 conv_resample: bool = False,
                 time_emb_learn: bool=False,
                 use_zero_module: bool = False,
                 use_zero_module_in_res_block:bool=False,
                 pred_noise:bool=None,
                 use_scale_shift_norm:bool=False,
                 use_coord:bool=True,
                 coord_dim:float=3,
                 eca_time:bool=True,
                 eca_dyn_dilate_rate=None,
                 eca_dyn_K=None
                 ):
        super().__init__()
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        assert len(n_conv_per_stage) == n_stages, "n_conv_per_stage must have as many entries as we have " \
                                                  f"resolution stages. here: {n_stages}. " \
                                                  f"n_conv_per_stage: {n_conv_per_stage}"
        assert len(n_conv_per_stage_decoder) == (n_stages - 1), "n_conv_per_stage_decoder must have one less entries " \
                                                                f"as we have resolution stages. here: {n_stages} " \
                                                                f"stages, so it should have {n_stages - 1} entries. " \
                                                                f"n_conv_per_stage_decoder: {n_conv_per_stage_decoder}"
        
        # initial time embedding module;
        self.time_embedding = time_embedding
        if time_embedding:
            self.model_base_channel = features_per_stage[0]
            self.time_embed = nn.Sequential(
                LearnedSinusoidalPosEmb(dim=self.model_base_channel) \
                    if time_emb_learn else TimeStepEmbedding(dim=self.model_base_channel),
                # nn.Linear(self.model_base_channel+1 if time_emb_learn \
                #           else self.model_base_channel, 
                #           time_embedding_channel),
                # nn.SiLU(),
                # nn.Linear(time_embedding_channel, time_embedding_channel),
            )

            if use_coord:
                time_embedding_channel = int(time_embedding_channel // 2)
                self.noisy_coord_time_embed = nn.Sequential(
                    nn.Linear(self.model_base_channel+1 if time_emb_learn \
                              else self.model_base_channel, 
                              time_embedding_channel),
                    nn.SiLU(),
                    nn.Linear(time_embedding_channel, time_embedding_channel),
                    )
                
            self.noisy_label_time_embed = nn.Sequential(
                nn.Linear(self.model_base_channel+1 if time_emb_learn \
                          else self.model_base_channel, 
                          time_embedding_channel),
                nn.SiLU(),
                nn.Linear(time_embedding_channel, time_embedding_channel),
            )

        # input layer for concat([image, noisy_label]);
        self.noisy_label_input_layer = conv_op(
            input_channels + num_classes,
            features_per_stage[0],
            3,
            1,
            padding=1,
            )
            
        # input layer for concat([image, noisy_coord]);
        self.use_coord = use_coord
        if use_coord == True:
            assert coord_dim is not None, f"Missing coordinate map input dimension: {coord_dim}."
            self.noisy_coord_input_layer = conv_op(
                input_channels + coord_dim,
                features_per_stage[0],
                3,
                1,
                padding=1
            )
        features_per_stage[0] = int(features_per_stage[0] * 2)

        # unet model part;
        self.encoder = ResConvEncoder(
            features_per_stage[0], 
            n_stages, 
            features_per_stage, 
            conv_op, 
            kernel_sizes, 
            strides,
            n_conv_per_stage, 
            conv_bias, 
            norm_op, 
            norm_op_kwargs, 
            dropout_op,
            dropout_op_kwargs, nonlin, nonlin_kwargs, return_skips=True,
            nonlin_first=nonlin_first, 
            time_embedding=time_embedding,
            time_embedding_channel=time_embedding_channel,
            fuse_method=fuse_method,
            contain_middle=contain_middle,
            conv_resample=conv_resample,
            use_zero_module=False,
            use_zero_module_in_res_block=use_zero_module_in_res_block,
            zero_module_func=make_zero_conv,
            use_scale_shift_norm=use_scale_shift_norm,
            use_coord=use_coord
            )
            
        self.decoder = ResConvDecoder(
            self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision,
            nonlin_first=nonlin_first,
            time_embedding=time_embedding,
            time_embedding_channel=time_embedding_channel,
            contain_middle=contain_middle,conv_resample=conv_resample,
            use_zero_module_in_res_block=use_zero_module_in_res_block,
            pred_noise=pred_noise,use_scale_shift_norm=use_scale_shift_norm,
            use_coord=use_coord, coord_dim=coord_dim, eca_time=eca_time,
            eca_dyn_dilate_rate=eca_dyn_dilate_rate, eca_dyn_K=eca_dyn_K
            )

    def forward(self, 
                noisy_label,
                label_time, 
                image, 
                coord_time=None,
                noisy_coord=None):
        ### time embedding ###
        if self.time_embedding:
            label_time_emb = self.noisy_label_time_embed(self.time_embed(label_time))
            coord_time_emb = None
            if self.use_coord:
                coord_time_emb = self.noisy_coord_time_embed(self.time_embed(coord_time))

        ### main branch related ###
        # initial encode;
        label_related_feature = self.noisy_label_input_layer(torch.cat([image, noisy_label], dim=1))
        input_feature = label_related_feature
        if self.use_coord:
            coord_related_feature = self.noisy_coord_input_layer(torch.cat([image, noisy_coord], dim=1))
            input_feature = torch.cat([label_related_feature, coord_related_feature], dim=1)

        # main branch encoder forward
        skips = self.encoder(input_feature, label_time_emb, coord_time_emb)
        output = self.decoder(skips, label_time_emb, coord_time_emb)

        if isinstance(output, tuple):
            result_dict = {'seg':output[0], 'reg':output[1]}
            return result_dict
        else:
            return output
        
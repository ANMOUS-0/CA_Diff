import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
# from .basic_unet import BasicUNetEncoder
# from .basic_unet_denose import BasicUNetDe
from .improved_diffusion_V1.gaussian_diffusion import get_named_beta_schedule, ModelMeanType, ModelVarType,LossType
from .improved_diffusion_V1.respace import SpacedDiffusion, space_timesteps
from .improved_diffusion_V1.resample import UniformSampler, ScheduleSampler

from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from .get_network_from_plans_for_DiffControlUnetV6_V3 import get_network_from_plans
from collections import OrderedDict

class StepBasedUniformSampler(ScheduleSampler):
    def __init__(self, step:int):
        self._weights = np.ones([step])

    def weights(self):
        return self._weights
    



class DenoiseFn(object):
    '''
    Process function for pred_xstart. 
    Supports using bit scale.
    '''

    def __init__(self, cls_num=None, fn_type='argmax_onehot', bit_scale:float=None):
        self.cls_num = cls_num
        self.fn_type = fn_type
        self.bit_scale = bit_scale
        assert fn_type in ['argmax_onehot', 'softmax',None], \
            NotImplementedError(f'Not implemented denoise method: {fn_type}')
        if self.fn_type == 'argmax_onehot':
            assert self.cls_num is not None,\
            'You shoud give class number for onehot operation in denoise function.'

    def __call__(self, logits):
        if self.fn_type == 'softmax':
            x = torch.softmax(logits, dim=1)
            # x = x * 2.0 - 1.0 

        elif self.fn_type == 'argmax_onehot':
            x = torch.argmax(logits, dim=1).long()
            x = F.one_hot(x, num_classes=self.cls_num).permute(0,4,1,2,3)
            # x = x * 2.0 - 1.0

        else:
            x = logits

        if self.bit_scale is not None:
            x *= self.bit_scale

        return x




class DiffControlUNet(nn.Module):
    '''
    NO CFG.
    '''

    def __init__(self,
                 plans_manager:PlansManager, 
                 dataset_json:dict, 
                 configuration_manager:ConfigurationManager,
                 deep_supervision:bool=False,
                 conv_per_stage=1,
                 sample_step=10,
                 clip_range=[-1,1],
                 model_mean_type=ModelMeanType.EPSILON,
                 model_var_type=ModelVarType.FIXED_LARGE,
                 loss_type=LossType.MSE,
                 p2_weight_gamma=0,
                 bit_scale:float=0.1,
                 denoise_fn='argmax_onehot',
                 network_class:str='ControlUNetZeroConv',
                 context_kwargs:dict=None,
                 control_kwargs:dict=None,
                 specific_kwargs:dict=None,
                 uncond_prob:float=0.1,
                 guidance_scale:float=1,
                 forward_type:str='simple',
                 return_x_start_mean:bool=False,
                 ddim:bool=True,
                 diffusion_indictor:str=None,
                 seg_indictor:str=None,
                 ensemble:int=1,
                 input_channnels:int=None
                 ) -> None:
        '''
        :param guidance_scale: The guidance scale used in classifier free guidance method. 1 indicates no uncond.
        '''
        super().__init__()
        self.plans_manager = plans_manager
        self.dataset_json = dataset_json
        self.configuration_manager = configuration_manager
        self.num_output_channels = plans_manager.get_label_manager(dataset_json).num_segmentation_heads
        self.deep_supervision = deep_supervision
        self.clip_range = clip_range
        self.model_mean_type=model_mean_type
        self.model_var_type = model_var_type
        self.loss_type=loss_type
        self.bit_scale = bit_scale
        self.sample_step = sample_step
        self.uncond_prob = uncond_prob
        self.guidance_scale = guidance_scale
        self.forward_type = forward_type
        self.return_x_start_mean = return_x_start_mean
        self.ddim = ddim
        self.diffusion_indictor = diffusion_indictor
        self.seg_indictor = seg_indictor
        self.ensemble = ensemble

        self.model = get_network_from_plans(
            plans_manager=plans_manager,
            dataset_json=dataset_json,
            configuration_manager=configuration_manager,
            num_input_channels=input_channnels,
            num_output_channels=self.num_output_channels * 2 if self.model_var_type==ModelVarType.LEARNED_RANGE else None,
            network_class_name=network_class,
            deep_supervision=self.deep_supervision,
            conv_per_stage=conv_per_stage,
            context_kwargs=context_kwargs,
            control_kwargs=control_kwargs,
            specific_kwargs=specific_kwargs
        )


        betas = get_named_beta_schedule("linear", 1000)
        self.diffusion = SpacedDiffusion(use_timesteps=space_timesteps(1000, [1000]),
                                         betas=betas,
                                         model_mean_type=self.model_mean_type,
                                         model_var_type=self.model_var_type,
                                         loss_type=self.loss_type,
                                         p2_gamma=p2_weight_gamma
                                         )

        if self.sample_step > 1:
            self.sample_diffusion = SpacedDiffusion(use_timesteps=space_timesteps(1000, [sample_step]),
                                                    betas=betas,
                                                    model_mean_type=self.model_mean_type,
                                                    model_var_type=self.model_var_type,
                                                    loss_type=self.loss_type,
                                                    )
        self.sampler = UniformSampler(self.diffusion)

        self.denoise_fn = DenoiseFn(cls_num=self.num_output_channels, fn_type=denoise_fn, bit_scale=self.bit_scale)



    def q_sample(self, x, noise=None, t=None):

        # assert (noise is None and t is None) or (noise is not None and t is not None), \
        # f"You should give noise and t for q_sample. Or you can ignore both of them to use automatic q_sample."

        if noise is None:
            noise = torch.randn_like(x).to(x.device)
        if t is None:
            t, weight = self.sampler.sample(x.shape[0], x.device)
        return self.diffusion.q_sample(x, t, noise=noise), t, noise


    def forward_train(self, 
                      image=None, 
                      label:dict=None, 
                      coord=None,
                      manual_noise=None,
                      manual_t=None):

        noisy_label, label_t, label_noise = self.q_sample(label, noise=manual_noise, t=manual_t)
        if coord is not None:
            noisy_coord, coord_t, coord_noise = self.q_sample(coord, noise=manual_noise, t=manual_t)
        model_output = self.model(image=image, label_time=label_t, coord_time=coord_t, noisy_label=noisy_label, noisy_coord=noisy_coord)
        time_dict = {'label_time':label_t, 'coord_time':coord_t}
        noise_dict = {'label_noise':label_noise, 'coord_noise':coord_noise}
        return model_output, time_dict, noise_dict
    

    def forward_train_val(self, 
                      image=None, 
                      label=None, 
                      coord=None,
                      manual_noise=None,
                      manual_t=None):

        max_t_ = torch.ones((label.shape[0],), dtype=torch.int, device=label.device) * 999
        noisy_label, label_t, label_noise = self.q_sample(label, noise=manual_noise, t=max_t_)
        noisy_label = torch.randn_like(noisy_label)
        if coord is not None:
            coord_t = torch.zeros((image.shape[0],), dtype=torch.int, device=image.device)
            noisy_coord = coord
            # noisy_coord, coord_t, coord_noise = self.q_sample(coord, noise=manual_noise, t=manual_t)
        model_output = self.model(image=image, label_time=label_t, coord_time=coord_t, noisy_label=noisy_label, noisy_coord=noisy_coord)
        time_dict = {'label_time':label_t, 'coord_time':coord_t}
        noise_dict = {'label_noise':label_noise, 'coord_noise':None}
        return model_output, time_dict, noise_dict    


    def forward_simple(self,
                       image=None,
                       model_kwargs=None):
        
        temp = torch.zeros((image.shape[0], self.num_output_channels, *image.shape[2:])).to(next(self.model.parameters()).device)
        
        assert not self.training, "The forward function should only be called in inference or validation!"
        for _ in range(self.ensemble):
            if self.sample_step > 1:
                if self.ddim:
                    sample_out = self.sample_diffusion.ddim_sample_loop(self.model,
                                                                        (image.shape[0],
                                                                        self.num_output_channels, *image.shape[2:]),
                                                                        clip_range=self.clip_range,
                                                                        model_kwargs=model_kwargs,
                                                                        diffusion_kwargs={"diffusion_indicator":self.diffusion_indictor},
                                                                        clip_denoised=False,
                                                                        denoised_fn=self.denoise_fn,
                                                                        device=next(self.model.parameters()).device
                                                                        )

                    sample_out = sample_out['sample'][self.seg_indictor]
                else:
                    sample_out = self.sample_diffusion.p_sample_loop(self.model, 
                                                                    (image.shape[0],
                                                                        self.num_output_channels, *image.shape[2:]),
                                                                    clip_denoised=True,
                                                                    denoised_fn=self.denoise_fn,
                                                                    model_kwargs={"context": context, "control":control},
                                                                    device=next(self.model.parameters()).device
                                                                    )
            else:
                pure_noise = torch.randn(
                    *(image.shape[0],self.num_output_channels, *image.shape[2:]),
                    device=next(self.model.parameters()).device,
                )
                t = torch.tensor([999.]*image.shape[0], device=pure_noise.device)
                sample_out = self.model(x=pure_noise, time=t, context=context, control=control)

            temp += sample_out
        return temp / self.ensemble     


    def forward_parallel(self,
                       image=None,
                       model_kwargs=None):
        
        temp = torch.zeros((image.shape[0], self.num_output_channels, *image.shape[2:])).to(next(self.model.parameters()).device)
        
        assert not self.training, "The forward function should only be called in inference or validation!"
        for _ in range(self.ensemble):
            if self.sample_step > 1:
                if self.ddim:
                    sample_out = self.sample_diffusion.ddim_sample_loop(self.model,
                                                                        (image.shape[0],
                                                                        self.num_output_channels, *image.shape[2:]),
                                                                        clip_range=self.clip_range,
                                                                        model_kwargs=model_kwargs,
                                                                        diffusion_kwargs={"diffusion_indicator":self.diffusion_indictor},
                                                                        clip_denoised=False,
                                                                        denoised_fn=self.denoise_fn,
                                                                        device=next(self.model.parameters()).device
                                                                        )

                    sample_out = sample_out['sample'][self.seg_indictor]
                else:
                    sample_out = self.sample_diffusion.p_sample_loop(self.model, 
                                                                    (image.shape[0],
                                                                        self.num_output_channels, *image.shape[2:]),
                                                                    clip_denoised=True,
                                                                    denoised_fn=self.denoise_fn,
                                                                    model_kwargs={"context": context, "control":control},
                                                                    device=next(self.model.parameters()).device
                                                                    )
            else:
                pure_noise = torch.randn(
                    *(image.shape[0],self.num_output_channels, *image.shape[2:]),
                    device=next(self.model.parameters()).device,
                )
                t = torch.tensor([999.]*image.shape[0], device=pure_noise.device)
                sample_out = self.model(x=pure_noise, time=t, context=context, control=control)

            temp += sample_out
        return temp / self.ensemble       


    def forward_simple_cond(self,
                       image=None,
                       model_kwargs=None):
        
        temp = torch.zeros((image.shape[0], self.num_output_channels, *image.shape[2:])).to(next(self.model.parameters()).device)
        model_kwargs['image'] = image
        assert not self.training, "The forward function should only be called in inference or validation!"
        for _ in range(self.ensemble):
            if self.sample_step > 1:
                if self.ddim:
                    sample_out = self.sample_diffusion.ddim_sample_loop(self.model,
                                                                        (image.shape[0],
                                                                        self.num_output_channels, *image.shape[2:]),
                                                                        clip_range=self.clip_range,
                                                                        model_kwargs=model_kwargs,
                                                                        diffusion_kwargs={"diffusion_indicator":self.diffusion_indictor},
                                                                        clip_denoised=False,
                                                                        denoised_fn=self.denoise_fn,
                                                                        device=next(self.model.parameters()).device
                                                                        )

                    if self.model_mean_type == ModelMeanType.START_X:
                        if not self.return_x_start_mean:
                            sample_out = sample_out['sample']["sample"]
                        else:
                            sample_out = sample_out['xstart_sum'] / self.sample_step
                    elif self.model_mean_type == ModelMeanType.EPSILON:
                        sample_out = sample_out['sample']['sample']
                else:
                    sample_out = self.sample_diffusion.p_sample_loop(self.model, 
                                                                    (image.shape[0],
                                                                        self.num_output_channels, *image.shape[2:]),
                                                                    clip_denoised=True,
                                                                    denoised_fn=self.denoise_fn,
                                                                    model_kwargs={"context": context, "control":control},
                                                                    device=next(self.model.parameters()).device
                                                                    )
            else:
                pure_noise = torch.randn(
                    *(image.shape[0],self.num_output_channels, *image.shape[2:]),
                    device=next(self.model.parameters()).device,
                )
                t = torch.tensor([999.]*image.shape[0], device=pure_noise.device)
                sample_out = self.model(noisy_label=pure_noise, label_time=t, **model_kwargs)
                if isinstance(sample_out, dict):
                    sample_out = sample_out[self.diffusion_indictor]
            temp += sample_out
        return temp / self.ensemble 
    
    
    def forward(self, 
                image=None,
                model_kwargs=None):
        
        if self.forward_type == 'simple':
            return self.forward_simple(image, model_kwargs)
        elif self.forward_type == 'parallel':
            return self.forward_parallel(image, model_kwargs)
        elif self.forward_type == 'simple_cond':
            return self.forward_simple_cond(image, model_kwargs)
        
        else:
            NotImplementedError(f"Wrong forward type: {self.forward_type}")




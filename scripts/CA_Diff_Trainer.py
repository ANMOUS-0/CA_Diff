import inspect
import multiprocessing
import os
import shutil
import sys
from copy import deepcopy
from datetime import datetime
from time import time, sleep
from typing import Union, Tuple, List
import SimpleITK as sitk
import numpy as np
import torch
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, \
    ContrastAugmentationTransform, GammaTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from batchgenerators.transforms.utility_transforms import RemoveLabelTransform, RenameTransform
from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile, save_json, maybe_mkdir_p
from torch._dynamo import OptimizedModule

from nnunetv2.configuration import ANISO_THRESHOLD, default_num_processes
# from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder
from nnunetv2.inference.export_prediction import export_prediction_from_logits, resample_and_save
from nnunetv2.inference.sliding_window_prediction import compute_gaussian
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_results
from nnunetv2.training.data_augmentation.compute_initial_patch_size import get_patch_size
from nnunetv2.training.data_augmentation.custom_transforms.cascade_transforms import MoveSegAsOneHotToData, \
    ApplyRandomBinaryOperatorTransform, RemoveRandomConnectedComponentFromOneHotEncodingTransform
from nnunetv2.training.data_augmentation.custom_transforms.deep_supervision_donwsampling import \
    DownsampleSegForDSTransform2
from nnunetv2.training.data_augmentation.custom_transforms.limited_length_multithreaded_augmenter import \
    LimitedLenWrapper,LimitedLenWrapper_nondet
from nnunetv2.training.data_augmentation.custom_transforms.region_based_training import \
    ConvertSegmentationToRegionsTransform
# from nnunetv2.training.data_augmentation.custom_transforms.transforms_for_dummy_2d import Convert2DTo3DTransform, \
#     Convert3DTo2DTransform
from nnunetv2.training.dataloading.data_loader_2d import nnUNetDataLoader2D
# from nnunetv2.training.dataloading.data_loader_3d import nnUNetDataLoader3D
# from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset
from nnunetv2.training.dataloading.utils import get_case_identifiers, unpack_dataset
# from nnunetv2.training.logging.nnunet_logger import nnUNetLogger
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss, DC_and_BCE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn, MemoryEfficientSoftDiceLoss
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.utilities.collate_outputs import collate_outputs
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from nnunetv2.utilities.file_path_utilities import check_workers_busy
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.helpers import empty_cache, dummy_context
from nnunetv2.utilities.label_handling.label_handling import convert_labelmap_to_one_hot, determine_num_input_channels
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from sklearn.model_selection import KFold
from torch import autocast, nn
from torch import distributed as dist
from torch.cuda import device_count
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from DiffUNetModules.DiffControlUnet import DiffControlUNet
from training.functions.loss import DiffUNet_loss_V1
from training.functions.deep_supervision import DeepSupervisionWrapper_V1
from training.functions.lr_schedulers import LinearWarmupCosineAnnealingLR
import torch.nn.functional as F

from training.functions.logger import nnUNetLogger as nnUNetLogger
from training.functions.evaluate_predictions import compute_metrics_on_folder

from training.functions.aux_V1 import (
    nnUNetDataset, nnUNetDataLoader3D,
    SpatialTransform, MaskTransform, MirrorTransform,
    Convert2DTo3DTransform, Convert3DTo2DTransform,
    NumpyToTensor, SpatialTransformIntraBatchSame,
    predict_sliding_window_return_logits
)
from collections import OrderedDict
from DiffUNetModules.improved_diffusion_V1.gaussian_diffusion import get_named_beta_schedule, ModelMeanType, ModelVarType,LossType

TRAINED_MODULE = ['decoder', 'control_branch']






class CA_Diff_Trainer(object):


    def __init__(self, plans: dict,
                 configuration: str,
                 fold: int,
                 dataset_json: dict,
                 unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda'),
                 num_epochs=5,
                 deep_supervision=False,
                 num_iterations_per_epoch=5,
                 num_val_iterations_per_epoch=1,
                 n_proc=12,
                 batch_size=2,
                 step=5,
                 eval_bg:bool=True,
                 get_mse: bool=False,
                 reg_loss_weight:float=1,
                 seg_loss_weight: float=1,
                 pair_loss_weight:float=1,
                 denoise_fn:str='argmax_onehot',
                 infer_mirror: bool=False,
                 aux_indicator:List[str]=['Colin27CoordMasked'],
                 aux_type: List[str]=['data'],
                 aux_border_cval: List[float]=[0],
                 aux_resample_order: List[int] = [None,],
                 aux_channels: List[int] = [3],
                 control_indicator:List[str]=[],
                 context_indicator:List[str]=['image'],
                 training_stage:str=None,
                 former_stage:str=None,
                 pred_noise:bool=False,
                 bit_scale:float=1,
                 val_mannual_name:str='cluster',
                 specific_ckpt=None,
                 diff_ensemble:int=5,
                 ):
        """
        Args:
            configuration: e.g. "3d_fullres".
            num_epochs: Used to control max epoch in training.
            eval_bg: Set True to compute evaluate the model containing background.
        """
        self.is_ddp = dist.is_available() and dist.is_initialized()
        self.local_rank = 0 if not self.is_ddp else dist.get_rank()
        self.n_proc = n_proc
        self.device = device

        self.step = step
        self.get_mse = get_mse

        # print what device we are using
        if self.is_ddp:  # implicitly it's clear that we use cuda in this case
            print(f"I am local rank {self.local_rank}. {device_count()} GPUs are available. The world size is "
                  f"{dist.get_world_size()}."
                  f"Setting device to {self.device}")
            self.device = torch.device(type='cuda', index=self.local_rank)
        else:
            if self.device.type == 'cuda':
                # we might want to let the user pick this but for now please pick the correct GPU with CUDA_VISIBLE_DEVICES=X
                self.device = torch.device(type='cuda', index=0)
            print(f"Using device: {self.device}")

        # loading and saving this class for continuing from checkpoint should not happen based on pickling. This
        # would also pickle the network etc. Bad, bad. Instead we just reinstantiate and then load the checkpoint we
        # need. So let's save the init args
        self.my_init_kwargs = {}
        for k in inspect.signature(self.__init__).parameters.keys():
            self.my_init_kwargs[k] = locals()[k]

        ###  Saving all the init args into class variables for later access
        self.plans_manager = PlansManager(plans)
        self.configuration_manager = self.plans_manager.get_configuration(configuration)
        self.configuration_name = configuration
        self.dataset_json = dataset_json
        self.fold = fold
        self.unpack_dataset = unpack_dataset

        ### Setting all the folder names. We need to make sure things don't crash in case we are just running
        # inference and some of the folders may not be defined!
        self.preprocessed_dataset_folder_base = join(nnUNet_preprocessed, self.plans_manager.dataset_name) \
            if nnUNet_preprocessed is not None else None
        self.output_folder_base = join(nnUNet_results, self.plans_manager.dataset_name,
                                       self.__class__.__name__ + '__' +
                                       self.plans_manager.plans_name + "__" + configuration) \
            if nnUNet_results is not None else None
        self.output_folder = join(self.output_folder_base, training_stage, f'fold_{fold}')

        self.preprocessed_dataset_folder = join(self.preprocessed_dataset_folder_base,
                                                self.configuration_manager.data_identifier)
        # unlike the previous nnunet folder_with_segs_from_previous_stage is now part of the plans. For now it has to
        # be a different configuration in the same plans
        # IMPORTANT! the mapping must be bijective, so lowres must point to fullres and vice versa (using
        # "previous_stage" and "next_stage"). Otherwise it won't work!
        self.is_cascaded = self.configuration_manager.previous_stage_name is not None
        self.folder_with_segs_from_previous_stage = \
            join(nnUNet_results, self.plans_manager.dataset_name,
                 self.__class__.__name__ + '__' + self.plans_manager.plans_name + "__" +
                 self.configuration_manager.previous_stage_name, 'predicted_next_stage', self.configuration_name) \
                if self.is_cascaded else None

        ### Some hyperparameters for you to fiddle with

        # lr dict for multi stage training;
        lr_dict = dict(stage1=1e-4, stage2=1e-4)
        assert training_stage.startswith('stage'),"The training stage flag need to start with 'stage', e.g., stage3"
        self.training_stage = training_stage
        self.initial_lr = lr_dict[training_stage]

        self.weight_decay = 1e-4
        self.oversample_foreground_percent = 0.33
        self.num_iterations_per_epoch = num_iterations_per_epoch
        self.num_val_iterations_per_epoch = num_val_iterations_per_epoch
        self.num_epochs = num_epochs
        self.current_epoch = 0

        ### Dealing with labels/regions
        self.label_manager = self.plans_manager.get_label_manager(dataset_json)
        # labels can either be a list of int (regular training) or a list of tuples of int (region-based training)
        # needed for predictions. We do sigmoid in case of (overlapping) regions

        self.num_input_channels = None  # -> self.initialize()
        self.network = None  # -> self._get_network()
        self.optimizer = self.lr_scheduler = None  # -> self.initialize
        self.grad_scaler = GradScaler() if self.device.type == 'cuda' else None
        self.loss = None  # -> self.initialize

        ### Simple logging. Don't take that away from me!
        # initialize log file. This is just our log for the print statements etc. Not to be confused with lightning
        # logging
        timestamp = datetime.now()
        maybe_mkdir_p(self.output_folder)
        self.log_file = join(self.output_folder, 
                             "stage_%s_training_log_%d_%d_%d_%02.0d_%02.0d_%02.0d.txt" %
                             (training_stage, timestamp.year, timestamp.month, timestamp.day, timestamp.hour, timestamp.minute,
                              timestamp.second))
        
        additional_items = ['reg_loss', 'seg_loss', 'consistence_loss', 'loss']
        val_additional_items = [f"{i}_val" for i in additional_items]
        self.logger = nnUNetLogger(aux_metrics=additional_items+val_additional_items)

        ### placeholders
        self.dataloader_train = self.dataloader_val = None  # see on_train_start

        ### initializing stuff for remembering things and such
        self._best_ema = None

        ### inference things
        self.inference_allowed_mirroring_axes = None  # this variable is set in
        # self.configure_rotation_dummyDA_mirroring_and_inital_patch_size and will be saved in checkpoints

        ### checkpoint saving stuff
        self.save_every = 10
        self.disable_checkpointing = False

        ## DDP batch size and oversampling can differ between workers and needs adaptation
        # we need to change the batch size in DDP because we don't use any of those distributed samplers
        self.manual_batch_size = batch_size
        self._set_batch_size_and_oversample()   

        self.was_initialized = False

        self.deep_supervision = deep_supervision

        # loss weight;
        loss_weight_dict = dict()
        loss_weight_dict['stage1'] = dict(reg=1,seg=1,pair=1)
        loss_weight_dict['stage2'] = dict(reg=0,seg=1,pair=0)
        self.reg_loss_weight = loss_weight_dict[training_stage]['reg']
        self.seg_loss_weight = loss_weight_dict[training_stage]['seg']
        self.pair_loss_weight = loss_weight_dict[training_stage]['pair']

        # use infer mirror
        self.infer_mirror = infer_mirror

        # denoise function used to transform logits to start
        self.denoise_fn = denoise_fn

        self.diff_ensemble = diff_ensemble

        # remove aux input if it is not needed in current stage;
        if training_stage == 'stage1':
            control_indicator = []
        aux_indicator = list(tuple(control_indicator + context_indicator + aux_indicator))
        if 'image' in aux_indicator:
            aux_indicator.remove('image')

        # set aux indicator to select which aux is used;
        self.aux_indicator = aux_indicator
        # set type of aux, can be image or label;
        self.aux_type = aux_type
        self.aux_type_dict = dict(zip(aux_indicator, aux_type))
        # set border value of aux input;
        self.aux_border_cval = aux_border_cval
        self.aux_border_cval_dict = dict(zip(aux_indicator, aux_border_cval))
        # mannualy set aux resample order;
        self.aux_resample_order = aux_resample_order
        self.aux_resample_order_dict = dict(zip(aux_indicator, aux_resample_order))
        # aux feature channels;
        self.aux_channels = aux_channels
        self.aux_channels_dict = dict(zip(aux_indicator, aux_channels))
        # seg control indicator to select the item to be used in control branch;
        self.control_indicator = control_indicator
        # context indicator. context is the item used in main branch, like stable diffusion branch.
        self.context_indicator = context_indicator

        # contain background as a class in final validation;
        self.eval_bg = eval_bg

        # load former stage model if former stage is not None.
        self.former_stage = former_stage
        self.bit_scale = bit_scale
        self.pred_noise = pred_noise

        self.specific_ckpt = specific_ckpt

        self.val_mannual_name = val_mannual_name + f'_ensemble{diff_ensemble}'
        if self.val_mannual_name is not None and self.specific_ckpt is not None:
            self.val_mannual_name = f"{self.val_mannual_name}_{str.split(self.specific_ckpt,'.')[0]}"

        # self.save_multi_epoch_ckpt = save_multi_epoch_ckpt

        



    def initialize(self):
        if not self.was_initialized:
            self.num_input_channels = determine_num_input_channels(self.plans_manager, self.configuration_manager,
                                                                   self.dataset_json)

            self.loss = self._build_loss()


            ### set context and control dict ###
            self.context_type, self.control_type = OrderedDict(), OrderedDict()
            self.context_channels, self.control_channels = OrderedDict(), OrderedDict()
            for i in self.context_indicator:
                if i in self.aux_type_dict:
                    self.context_type[i] = self.aux_type_dict[i]
                    self.context_channels[i] = self.aux_channels_dict[i]
                elif i == 'image':
                    self.context_type[i] = 'data'
                    self.context_channels[i] = 1
                else:
                    NotImplementedError(f"Not support context type: {i}")
            for i in self.control_indicator:
                if i in self.aux_type_dict:
                    self.control_type[i] = self.aux_type_dict[i]
                    self.control_channels[i] = self.aux_channels_dict[i]
                elif i == 'image':
                    self.control_type[i] = 'data'
                    self.control_channels[i] = 1
                else:
                    NotImplementedError(f"Not support control type: {i}")
            

            self.network = DiffControlUNet(
                self.plans_manager, 
                self.dataset_json,
                self.configuration_manager,
                deep_supervision=self.deep_supervision,
                sample_step=self.step,
                clip_range=[-1,1],
                bit_scale=self.bit_scale,
                denoise_fn=self.denoise_fn,
                network_class='ParallelDiffUNet3D_ETCA_Dyn',
                input_channnels=self.num_input_channels,
                specific_kwargs=dict(
                    use_zero_module_in_res_block=False,
                    pred_noise=False,
                    use_coord=True,
                    coord_dim=3,
                    eca_dyn_dilate_rate=[7,11,15],
                    eca_dyn_K=3,
                ),
                return_x_start_mean=False,
                ddim=True,
                model_mean_type=ModelMeanType.EPSILON if self.pred_noise else ModelMeanType.START_X,
                ensemble=self.diff_ensemble,
                forward_type='simple_cond',
                diffusion_indictor='seg'
                ).to(self.device)
            

            if self.former_stage is not None:
                for name, param in self.network.model.named_parameters():
                    param.requires_grad = False
                for name, param in self.network.model.named_parameters():
                    if any(i in name for i in TRAINED_MODULE):
                        param.requires_grad = True                

            model_param_num = sum(p.numel() for p in self.network.model.parameters() if p.requires_grad)
            self.print_to_log_file(f'The model parameter num is :{model_param_num}')

            def count_parameters(model):
                return sum(p.numel() for p in model.parameters() if p.requires_grad)

            for name, module in self.network.model.named_children():
                print(f"{name}: {count_parameters(module)} parameters")

            # Example usage:
            # model = YourModelHere()
            # print_parameters(model)

            # compile network for free speedup
            if ('nnUNet_compile' in os.environ.keys()) and (
                    os.environ['nnUNet_compile'].lower() in ('true', '1', 't')):
                self.print_to_log_file('Compiling network...')
                self.network = torch.compile(self.network)

            self.optimizer, self.lr_scheduler = self.configure_optimizers()
            # if ddp, wrap in DDP wrapper
            if self.is_ddp:
                self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
                self.network = DDP(self.network, device_ids=[self.local_rank])


            self.was_initialized = True

            if self.former_stage is not None:
                former_stage_model_path = join(self.output_folder_base, self.former_stage, f"fold_{self.fold}",
                                               'checkpoint_final.pth')
                assert os.path.exists(former_stage_model_path), \
                f"The former stage {self.former_stage} seems not finish training."\
                f"Missing required file:\n{former_stage_model_path}"
                self.simple_load_checkpoint(former_stage_model_path)
                if self.local_rank == 0:
                    self.print_to_log_file(f"Load former stage model:{former_stage_model_path}")

        else:
            raise RuntimeError("You have called self.initialize even though the trainer was already initialized. "
                               "That should not happen.")
        


    def _save_debug_information(self):
        # saving some debug information
        if self.local_rank == 0:
            dct = {}
            for k in self.__dir__():
                if not k.startswith("__"):
                    if not callable(getattr(self, k)) or k in ['loss', ]:
                        dct[k] = str(getattr(self, k))
                    elif k in ['network', ]:
                        dct[k] = str(getattr(self, k).__class__.__name__)
                    else:
                        # print(k)
                        pass
                if k in ['dataloader_train', 'dataloader_val']:
                    if hasattr(getattr(self, k), 'generator'):
                        dct[k + '.generator'] = str(getattr(self, k).generator)
                    if hasattr(getattr(self, k), 'num_processes'):
                        dct[k + '.num_processes'] = str(getattr(self, k).num_processes)
                    if hasattr(getattr(self, k), 'transform'):
                        dct[k + '.transform'] = str(getattr(self, k).transform)
            import subprocess
            hostname = subprocess.getoutput(['hostname'])
            dct['hostname'] = hostname
            torch_version = torch.__version__
            if self.device.type == 'cuda':
                gpu_name = torch.cuda.get_device_name()
                dct['gpu_name'] = gpu_name
                cudnn_version = torch.backends.cudnn.version()
            else:
                cudnn_version = 'None'
            dct['device'] = str(self.device)
            dct['torch_version'] = torch_version
            dct['cudnn_version'] = cudnn_version
            save_json(dct, join(self.output_folder, "debug.json"))

    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        """
        his is where you build the architecture according to the plans. There is no obligation to use
        get_network_from_plans, this is just a utility we use for the nnU-Net default architectures. You can do what
        you want. Even ignore the plans and just return something static (as long as it can process the requested
        patch size)
        but don't bug us with your bugs arising from fiddling with this :-P
        This is the function that is called in inference as well! This is needed so that all network architecture
        variants can be loaded at inference time (inference will use the same nnUNetTrainer that was used for
        training, so if you change the network architecture during training by deriving a new trainer class then
        inference will know about it).

        If you need to know how many segmentation outputs your custom architecture needs to have, use the following snippet:
        > label_manager = plans_manager.get_label_manager(dataset_json)
        > label_manager.num_segmentation_heads
        (why so complicated? -> We can have either classical training (classes) or regions. If we have regions,
        the number of outputs is != the number of classes. Also there is the ignore label for which no output
        should be generated. label_manager takes care of all that for you.)

        """
        return get_network_from_plans(plans_manager, dataset_json, configuration_manager,
                                      num_input_channels, deep_supervision=enable_deep_supervision)

    def _get_deep_supervision_scales(self):

        deep_supervision_scales = [[1,1,1]]

        return deep_supervision_scales

    def _set_batch_size_and_oversample(self):
        if not self.is_ddp:
            # set batch size to what the plan says, leave oversample untouched
            self.batch_size = self.manual_batch_size or self.configuration_manager.batch_size
        else:
            # batch size is distributed over DDP workers and we need to change oversample_percent for each worker
            batch_sizes = []
            oversample_percents = []

            world_size = dist.get_world_size()
            my_rank = dist.get_rank()

            global_batch_size = self.manual_batch_size or self.configuration_manager.batch_size
            assert global_batch_size >= world_size, \
                'Cannot run DDP if the batch size is smaller than the number of GPUs... Duh.'

            batch_size_per_GPU = np.ceil(global_batch_size / world_size).astype(int)

            for rank in range(world_size):
                if (rank + 1) * batch_size_per_GPU > global_batch_size:
                    batch_size = batch_size_per_GPU - ((rank + 1) * batch_size_per_GPU - global_batch_size)
                else:
                    batch_size = batch_size_per_GPU

                batch_sizes.append(batch_size)

                sample_id_low = 0 if len(batch_sizes) == 0 else np.sum(batch_sizes[:-1])
                sample_id_high = np.sum(batch_sizes)

                if sample_id_high / global_batch_size < (1 - self.oversample_foreground_percent):
                    oversample_percents.append(0.0)
                elif sample_id_low / global_batch_size > (1 - self.oversample_foreground_percent):
                    oversample_percents.append(1.0)
                else:
                    percent_covered_by_this_rank = sample_id_high / global_batch_size - \
                                                   sample_id_low / global_batch_size
                    oversample_percent_here = 1 - (((1 - self.oversample_foreground_percent) -
                                                    sample_id_low / global_batch_size) / percent_covered_by_this_rank)
                    oversample_percents.append(oversample_percent_here)

            print("worker", my_rank, "oversample", oversample_percents[my_rank])
            print("worker", my_rank, "batch_size", batch_sizes[my_rank])
            # self.print_to_log_file("worker", my_rank, "oversample", oversample_percents[my_rank])
            # self.print_to_log_file("worker", my_rank, "batch_size", batch_sizes[my_rank])

            self.batch_size = batch_sizes[my_rank]
            self.oversample_foreground_percent = oversample_percents[my_rank]

    def _build_loss(self):
        if self.label_manager.has_regions:
            loss = DC_and_BCE_loss({},
                                   {'batch_dice': self.configuration_manager.batch_dice,
                                    'do_bg': True, 'smooth': 1e-5, 'ddp': self.is_ddp},
                                   use_ignore_label=self.label_manager.ignore_label is not None,
                                   dice_class=MemoryEfficientSoftDiceLoss)
        else:
            loss = DC_and_CE_loss({'batch_dice': self.configuration_manager.batch_dice,
                                   'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, {}, weight_ce=1, weight_dice=1,
                                  ignore_label=self.label_manager.ignore_label, dice_class=MemoryEfficientSoftDiceLoss)

        deep_supervision_scales = self._get_deep_supervision_scales()


    #     # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
    #     # this gives higher resolution outputs more weight in the loss
        if deep_supervision_scales is not None:
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
        else:
            weights = [1]

        # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
        weights = weights / weights.sum()
        # now wrap the loss
        loss = DeepSupervisionWrapper_V1(loss, weights)
        return loss

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        """
        This function is stupid and certainly one of the weakest spots of this implementation. Not entirely sure how we can fix it.
        """
        patch_size = self.configuration_manager.patch_size
        dim = len(patch_size)
        # todo rotation should be defined dynamically based on patch size (more isotropic patch sizes = more rotation)
        if dim == 2:
            do_dummy_2d_data_aug = False
            # todo revisit this parametrization
            if max(patch_size) / min(patch_size) > 1.5:
                rotation_for_DA = {
                    'x': (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
                    'y': (0, 0),
                    'z': (0, 0)
                }
            else:
                rotation_for_DA = {
                    'x': (-180. / 360 * 2. * np.pi, 180. / 360 * 2. * np.pi),
                    'y': (0, 0),
                    'z': (0, 0)
                }
            mirror_axes = (0, 1)
        elif dim == 3:
            # todo this is not ideal. We could also have patch_size (64, 16, 128) in which case a full 180deg 2d rot would be bad
            # order of the axes is determined by spacing, not image size
            do_dummy_2d_data_aug = (max(patch_size) / patch_size[0]) > ANISO_THRESHOLD
            if do_dummy_2d_data_aug:
                # why do we rotate 180 deg here all the time? We should also restrict it
                rotation_for_DA = {
                    'x': (-180. / 360 * 2. * np.pi, 180. / 360 * 2. * np.pi),
                    'y': (0, 0),
                    'z': (0, 0)
                }
            else:
                rotation_for_DA = {
                    'x': (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
                    'y': (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
                    'z': (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
                }
            mirror_axes = (0, 1, 2)
        else:
            raise RuntimeError()

        # todo this function is stupid. It doesn't even use the correct scale range (we keep things as they were in the
        #  old nnunet for now)
        initial_patch_size = get_patch_size(patch_size[-dim:],
                                            *rotation_for_DA.values(),
                                            (0.85, 1.25))
        if do_dummy_2d_data_aug:
            initial_patch_size[0] = patch_size[0]

        self.print_to_log_file(f'do_dummy_2d_data_aug: {do_dummy_2d_data_aug}')
        self.inference_allowed_mirroring_axes = mirror_axes

        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes

    def print_to_log_file(self, *args, also_print_to_console=True, add_timestamp=True):
        if self.local_rank == 0:
            timestamp = time()
            dt_object = datetime.fromtimestamp(timestamp)

            if add_timestamp:
                args = ("%s:" % dt_object, *args)

            successful = False
            max_attempts = 5
            ctr = 0
            while not successful and ctr < max_attempts:
                try:
                    with open(self.log_file, 'a+') as f:
                        for a in args:
                            f.write(str(a))
                            f.write(" ")
                        f.write("\n")
                    successful = True
                except IOError:
                    print("%s: failed to log: " % datetime.fromtimestamp(timestamp), sys.exc_info())
                    sleep(0.5)
                    ctr += 1
            if also_print_to_console:
                print(*args)
        elif also_print_to_console:
            print(*args)

    def print_plans(self):
        if self.local_rank == 0:
            dct = deepcopy(self.plans_manager.plans)
            del dct['configurations']
            self.print_to_log_file(f"\nThis is the configuration used by this "
                                   f"training:\nConfiguration name: {self.configuration_name}\n",
                                   self.configuration_manager, '\n', add_timestamp=False)
            self.print_to_log_file('These are the global plan.json settings:\n', dct, '\n', add_timestamp=False)

    def configure_optimizers(self):
        trainable_parameters = filter(lambda p: p.requires_grad, self.network.parameters())
        optimizer = torch.optim.AdamW(trainable_parameters, lr=self.initial_lr, weight_decay=self.weight_decay, amsgrad=True)
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
        return optimizer, lr_scheduler

    def plot_network_architecture(self):
        if self.local_rank == 0:
            try:
                # raise NotImplementedError('hiddenlayer no longer works and we do not have a viable alternative :-(')
                # pip install git+https://github.com/saugatkandel/hiddenlayer.git

                # from torchviz import make_dot
                # # not viable.
                # make_dot(tuple(self.network(torch.rand((1, self.num_input_channels,
                #                                         *self.configuration_manager.patch_size),
                #                                        device=self.device)))).render(
                #     join(self.output_folder, "network_architecture.pdf"), format='pdf')
                # self.optimizer.zero_grad()

                # broken.

                import hiddenlayer as hl
                g = hl.build_graph(self.network,
                                   torch.rand((1, self.num_input_channels,
                                               *self.configuration_manager.patch_size),
                                              device=self.device),
                                   transforms=None)
                g.save(join(self.output_folder, "network_architecture.pdf"))
                del g
            except Exception as e:
                self.print_to_log_file("Unable to plot network architecture:")
                self.print_to_log_file(e)

                # self.print_to_log_file("\nprinting the network instead:\n")
                # self.print_to_log_file(self.network)
                # self.print_to_log_file("\n")
            finally:
                empty_cache(self.device)


    def do_split(self):

        splits_file = join(self.preprocessed_dataset_folder_base, "malc_datasplit.json")

        self.print_to_log_file("Using splits from existing split file:", splits_file)
        splits = load_json(splits_file)
        self.print_to_log_file("The split file contains %d splits." % len(splits))

        self.print_to_log_file("Desired fold for training: %d" % self.fold)
        assert self.fold < len(splits), f"Your datasplit file has {len(splits)} fold as you required fold {self.fold}."
        tr_keys = splits[self.fold]['train']
        val_keys = splits[self.fold]['val']
        self.print_to_log_file("This split has %d training and %d validation cases."
                                % (len(tr_keys), len(val_keys)))

        if any([i in val_keys for i in tr_keys]):
            self.print_to_log_file('WARNING: Some validation cases are also in the training set. Please check the '
                                    'splits.json or ignore if this is intentional.')
        return tr_keys, val_keys

    def get_tr_and_val_datasets(self):
        # create dataset split
        tr_keys, val_keys = self.do_split()

        # load the datasets for training and validation. Note that we always draw random samples so we really don't
        # care about distributing training cases across GPUs.
        dataset_tr = nnUNetDataset(self.preprocessed_dataset_folder, tr_keys,
                                   folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
                                   num_images_properties_loading_threshold=0,
                                   aux_indicator=self.aux_indicator,
                                   aux_type=self.aux_type)
        dataset_val = nnUNetDataset(self.preprocessed_dataset_folder, val_keys,
                                    folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
                                    num_images_properties_loading_threshold=0,
                                    aux_indicator=self.aux_indicator,
                                    aux_type=self.aux_type)
        return dataset_tr, dataset_val

    def get_dataloaders(self):
        # we use the patch size to determine whether we need 2D or 3D dataloaders. We also use it to determine whether
        # we need to use dummy 2D augmentation (in case of 3D training) and what our initial patch size should be
        patch_size = self.configuration_manager.patch_size
        dim = len(patch_size)

        # needed for deep supervision: how much do we need to downscale the segmentation targets for the different
        # outputs?
        deep_supervision_scales = self._get_deep_supervision_scales()

        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes = \
            self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        
        # not use mirror augmentation;
        mirror_axes = None

        # training pipeline
        tr_transforms = self.get_training_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
            order_resampling_data=3, order_resampling_seg=0,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded, foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label)

        # validation pipeline
        val_transforms = self.get_validation_transforms(deep_supervision_scales,
                                                        is_cascaded=self.is_cascaded,
                                                        foreground_labels=self.label_manager.foreground_labels,
                                                        regions=self.label_manager.foreground_regions if
                                                        self.label_manager.has_regions else None,
                                                        ignore_label=self.label_manager.ignore_label)

        dl_tr, dl_val = self.get_plain_dataloaders(initial_patch_size, dim)

        allowed_num_processes = int(min(self.n_proc, os.cpu_count()))

        if self.is_ddp:
            train_seed = list(
                np.arange(int(dist.get_rank()+1)*100,int(dist.get_rank()+1)*100+allowed_num_processes)
            )
            val_seed = list(
                np.arange(int(dist.get_rank()+1)*1000,int(dist.get_rank()+1)*1000+max(1, allowed_num_processes // 2))
            )
        else:
            train_seed = list(np.arange(100,100+allowed_num_processes))
            val_seed = list(np.arange(1000,1000+max(1, allowed_num_processes // 2)))

        if allowed_num_processes == 0:
            mt_gen_train = SingleThreadedAugmenter(dl_tr, tr_transforms)
            mt_gen_val = SingleThreadedAugmenter(dl_val, val_transforms)
        else:
            # mt_gen_train = LimitedLenWrapper_nondet(self.num_iterations_per_epoch, data_loader=dl_tr, transform=tr_transforms,
            #                                  num_processes=allowed_num_processes, num_cached=6, seeds=train_seed,
            #                                  pin_memory=self.device.type == 'cuda', wait_time=0.02)
            # mt_gen_val = LimitedLenWrapper_nondet(self.num_val_iterations_per_epoch, data_loader=dl_val,
            #                                transform=val_transforms, num_processes=max(1, allowed_num_processes // 2),
            #                                num_cached=3, seeds=val_seed, pin_memory=self.device.type == 'cuda',
            #                                wait_time=0.02)

            mt_gen_train = LimitedLenWrapper(self.num_iterations_per_epoch, data_loader=dl_tr, transform=tr_transforms,
                                             num_processes=allowed_num_processes, seeds=train_seed,
                                             pin_memory=self.device.type == 'cuda', wait_time=0.02)
            mt_gen_val = LimitedLenWrapper(self.num_val_iterations_per_epoch, data_loader=dl_val,
                                           transform=val_transforms, num_processes=max(1, allowed_num_processes // 2),
                                           seeds=val_seed, pin_memory=self.device.type == 'cuda',
                                           wait_time=0.02)
        return mt_gen_train, mt_gen_val

    def get_plain_dataloaders(self, initial_patch_size: Tuple[int, ...], dim: int):
        dataset_tr, dataset_val = self.get_tr_and_val_datasets()

        # if dim == 2:
        #     dl_tr = nnUNetDataLoader2D(dataset_tr, self.batch_size,
        #                                initial_patch_size,
        #                                self.configuration_manager.patch_size,
        #                                self.label_manager,
        #                                oversample_foreground_percent=self.oversample_foreground_percent,
        #                                sampling_probabilities=None, pad_sides=None)
        #     dl_val = nnUNetDataLoader2D(dataset_val, self.batch_size,
        #                                 self.configuration_manager.patch_size,
        #                                 self.configuration_manager.patch_size,
        #                                 self.label_manager,
        #                                 oversample_foreground_percent=self.oversample_foreground_percent,
        #                                 sampling_probabilities=None, pad_sides=None)
        assert dim==3, f'aux information dataset currently not support dim : {dim}'
        dl_tr = nnUNetDataLoader3D(dataset_tr, self.batch_size,
                                    initial_patch_size,
                                    self.configuration_manager.patch_size,
                                    self.label_manager,
                                    oversample_foreground_percent=self.oversample_foreground_percent,
                                    sampling_probabilities=None, pad_sides=None,
                                    aux_border_cval=self.aux_border_cval)
        dl_val = nnUNetDataLoader3D(dataset_val, self.batch_size,
                                    self.configuration_manager.patch_size,
                                    self.configuration_manager.patch_size,
                                    self.label_manager,
                                    oversample_foreground_percent=self.oversample_foreground_percent,
                                    sampling_probabilities=None, pad_sides=None,
                                    aux_border_cval=self.aux_border_cval)
        return dl_tr, dl_val

    # @staticmethod
    def get_training_transforms(self, patch_size: Union[np.ndarray, Tuple[int]],
                                rotation_for_DA: dict,
                                deep_supervision_scales: Union[List, Tuple],
                                mirror_axes: Tuple[int, ...],
                                do_dummy_2d_data_aug: bool,
                                order_resampling_data: int = 3,
                                order_resampling_seg: int = 1,
                                border_val_seg: int = -1,
                                use_mask_for_norm: List[bool] = None,
                                is_cascaded: bool = False,
                                foreground_labels: Union[Tuple[int, ...], List[int]] = None,
                                regions: List[Union[List[int], Tuple[int, ...], int]] = None,
                                ignore_label: int = None) -> AbstractTransform:
        tr_transforms = []
        if do_dummy_2d_data_aug:
            ignore_axes = (0,)
            tr_transforms.append(Convert3DTo2DTransform(apply_to_keys=('data', 'seg', 'aux')))  # modified for control;
            patch_size_spatial = patch_size[1:]
        else:
            patch_size_spatial = patch_size
            ignore_axes = None

        # for same batch, use same spatial transformation;
        tr_transforms.append(SpatialTransformIntraBatchSame(
            patch_size_spatial, patch_center_dist_from_border=None,
            do_elastic_deform=False, alpha=(0, 0), sigma=(0, 0),
            do_rotation=True, angle_x=rotation_for_DA['x'], angle_y=rotation_for_DA['y'], angle_z=rotation_for_DA['z'],
            p_rot_per_axis=1,  # todo experiment with this
            do_scale=True, scale=(0.7, 1.4),
            border_mode_data="constant", border_cval_data=0, order_data=order_resampling_data,
            border_mode_seg="constant", border_cval_seg=border_val_seg, order_seg=order_resampling_seg,
            random_crop=False,  # random cropping is part of our dataloaders
            p_el_per_sample=0, p_scale_per_sample=0.2, p_rot_per_sample=0.2,
            independent_scale_for_each_axis=False,  # todo experiment with this,
            aux_key='aux',
            aux_type=self.aux_type,
            border_cval_aux=self.aux_border_cval, # TODO: this value should be changed according to the aux input.
            order_aux=self.aux_resample_order
        ))

        if do_dummy_2d_data_aug:
            tr_transforms.append(Convert2DTo3DTransform(apply_to_keys=('data', 'seg', 'aux')))   # modified for aux;

        tr_transforms.append(GaussianNoiseTransform(p_per_sample=0.1))


        if mirror_axes is not None and len(mirror_axes) > 0:
            tr_transforms.append(MirrorTransform(mirror_axes))

        if use_mask_for_norm is not None and any(use_mask_for_norm):
            tr_transforms.append(MaskTransform([i for i in range(len(use_mask_for_norm)) if use_mask_for_norm[i]],
                                               mask_idx_in_seg=0, set_outside_to=0))
            
            # set all values outside the image to a certain value; be careful with this!
            # tr_transforms.append(MaskTransform([i for i in range(len(use_mask_for_norm)) if use_mask_for_norm[i]],
            #                                    mask_idx_in_seg=0, set_outside_to=self.control_border_cval, data_key='cond'))

        tr_transforms.append(RemoveLabelTransform(-1, 0))

        if is_cascaded:
            assert foreground_labels is not None, 'We need foreground_labels for cascade augmentations'
            tr_transforms.append(MoveSegAsOneHotToData(1, foreground_labels, 'seg', 'data'))
            tr_transforms.append(ApplyRandomBinaryOperatorTransform(
                channel_idx=list(range(-len(foreground_labels), 0)),
                p_per_sample=0.4,
                key="data",
                strel_size=(1, 8),
                p_per_label=1))
            tr_transforms.append(
                RemoveRandomConnectedComponentFromOneHotEncodingTransform(
                    channel_idx=list(range(-len(foreground_labels), 0)),
                    key="data",
                    p_per_sample=0.2,
                    fill_with_other_class_p=0,
                    dont_do_if_covers_more_than_x_percent=0.15))

        tr_transforms.append(RenameTransform('seg', 'target', True))

        if regions is not None:
            # the ignore label must also be converted
            tr_transforms.append(ConvertSegmentationToRegionsTransform(list(regions) + [ignore_label]
                                                                       if ignore_label is not None else regions,
                                                                       'target', 'target'))

        if deep_supervision_scales is not None:
            tr_transforms.append(DownsampleSegForDSTransform2(deep_supervision_scales, 0, input_key='target',
                                                              output_key='target'))
        tr_transforms.append(NumpyToTensor(['data', 'target', 'aux'], 'float'))
        tr_transforms = Compose(tr_transforms)
        return tr_transforms

    # @staticmethod
    def get_validation_transforms(self, deep_supervision_scales: Union[List, Tuple],
                                  is_cascaded: bool = False,
                                  foreground_labels: Union[Tuple[int, ...], List[int]] = None,
                                  regions: List[Union[List[int], Tuple[int, ...], int]] = None,
                                  ignore_label: int = None) -> AbstractTransform:
        val_transforms = []
        val_transforms.append(RemoveLabelTransform(-1, 0))

        if is_cascaded:
            val_transforms.append(MoveSegAsOneHotToData(1, foreground_labels, 'seg', 'data'))

        val_transforms.append(RenameTransform('seg', 'target', True))

        if regions is not None:
            # the ignore label must also be converted
            val_transforms.append(ConvertSegmentationToRegionsTransform(list(regions) + [ignore_label]
                                                                        if ignore_label is not None else regions,
                                                                        'target', 'target'))

        if deep_supervision_scales is not None:
            val_transforms.append(DownsampleSegForDSTransform2(deep_supervision_scales, 0, input_key='target',
                                                               output_key='target'))

        val_transforms.append(NumpyToTensor(['data', 'target', 'aux'], 'float'))
        val_transforms = Compose(val_transforms)
        return val_transforms

    def set_deep_supervision_enabled(self, enabled: bool):
        """
        This function is specific for the default architecture in nnU-Net. If you change the architecture, there are
        chances you need to change this as well!
        """
        if self.is_ddp:
            self.network.module.train_ds = enabled
        else:
            self.network.train_ds = enabled

    def on_train_start(self):
        if not self.was_initialized:
            self.initialize()

        maybe_mkdir_p(self.output_folder)

        # make sure deep supervision is on in the network
        self.set_deep_supervision_enabled(self.deep_supervision)

        self.print_plans()
        empty_cache(self.device)

        # maybe unpack
        if self.unpack_dataset and self.local_rank == 0:
            self.print_to_log_file('unpacking dataset...')
            unpack_dataset(self.preprocessed_dataset_folder, unpack_segmentation=True, overwrite_existing=False,
                           num_processes=max(1, round(get_allowed_n_proc_DA() // 2)))
            self.print_to_log_file('unpacking done...')

        if self.is_ddp:
            dist.barrier()

        # dataloaders must be instantiated here because they need access to the training data which may not be present
        # when doing inference
        self.dataloader_train, self.dataloader_val = self.get_dataloaders()

        # copy plans and dataset.json so that they can be used for restoring everything we need for inference
        save_json(self.plans_manager.plans, join(self.output_folder_base, 'plans.json'), sort_keys=False)
        save_json(self.dataset_json, join(self.output_folder_base, 'dataset.json'), sort_keys=False)

        # we don't really need the fingerprint, but it still handy to have it with the others
        shutil.copy(join(self.preprocessed_dataset_folder_base, 'dataset_fingerprint.json'),
                    join(self.output_folder_base, 'dataset_fingerprint.json'))

        # produces a pdf in output folder
        self.plot_network_architecture()

        self._save_debug_information()

        # print(f"batch size: {self.batch_size}")
        # print(f"oversample: {self.oversample_foreground_percent}")

    def on_train_end(self):
        self.save_checkpoint(join(self.output_folder, "checkpoint_final.pth"))
        # now we can delete latest
        if self.local_rank == 0 and isfile(join(self.output_folder, "checkpoint_latest.pth")):
            os.remove(join(self.output_folder, "checkpoint_latest.pth"))

        # shut down dataloaders
        old_stdout = sys.stdout
        with open(os.devnull, 'w') as f:
            sys.stdout = f
            if self.dataloader_train is not None:
                self.dataloader_train._finish()
            if self.dataloader_val is not None:
                self.dataloader_val._finish()
            sys.stdout = old_stdout

        empty_cache(self.device)
        self.print_to_log_file("Training done.")

    def on_train_epoch_start(self):
        self.network.train()
        self.lr_scheduler.step(self.current_epoch)
        self.print_to_log_file('')
        self.print_to_log_file(f'Epoch {self.current_epoch}')
        self.print_to_log_file(
            f"Current learning rate: {np.round(self.optimizer.param_groups[0]['lr'], decimals=10)}")
        # lrs are the same for all workers so we don't need to gather them in case of DDP training
        self.logger.log('lrs', self.optimizer.param_groups[0]['lr'], self.current_epoch)

    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']
        aux = batch['aux']

        # contains_neg_one = (aux[self.aux_indicator[1]].eq(-1).any()).item()
        # assert not contains_neg_one, \
        # "The label like aux should not include -1, something wrong."

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)
        if isinstance(aux, list):
            aux = [i.to(self.device, non_blocking=True) for i in aux]
        elif isinstance(aux, dict):
            for i,(k,v) in enumerate(aux.items()):
                aux[k] = v.to(self.device, non_blocking=True)
                if self.aux_type[i]=='seg':
                    aux[k] = aux[k].long()
        else:
            aux = [aux.to(self.device, non_blocking=True)]


        # context, control = OrderedDict(), OrderedDict()
        # for i in self.context_indicator:
        #     if i in aux:
        #         context[i] = aux[i]
        #     elif i=='image':
        #         context[i] = data
        #     else:
        #         NotImplementedError(f"Not support context indicator: {i}")

        # for i in self.control_indicator:
        #     if i in aux:
        #         control[i] = aux[i]
        #     elif i=='image':
        #         control[i] = data
        #     else:
        #         NotImplementedError(f"Not support control indicator: {i}")
        # del aux

        # self.print_to_log_file("Data loaded for current batch.")

        
        self.optimizer.zero_grad()
        # Autocast is a little bitch.
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            label_ = target[0].squeeze(dim=1).long()
            label_ = F.one_hot(label_, num_classes=self.label_manager.num_segmentation_heads).permute(0, 4, 1, 2, 3).float()
            if self.is_ddp:
                output, time, noise = self.network.module.forward_train(image=data, label=label_, coord=aux[self.aux_indicator[0]])
            else:
                output, time, noise = self.network.forward_train(image=data, label=label_, coord=aux[self.aux_indicator[0]])
            
            del data

            # segmentation loss;
            if isinstance(output, dict):
                seg_output = output['seg']
            else:
                seg_output = output
            segmentation_loss = self.loss([seg_output], [target[0]])
            segmentation_loss *= self.seg_loss_weight
            
            # coordinate regression loss;
            if self.reg_loss_weight > 0:
                reg_loss = F.mse_loss(output['reg'], aux[self.aux_indicator[0]])
            else:
                reg_loss = torch.zeros_like(segmentation_loss)
            reg_loss *= self.reg_loss_weight

            # seg and coord consistence loss;
            # TODO: this consistence loss need to be improved; this is just a simple version;
            if self.pair_loss_weight > 0:
                assert seg_output.shape[0] > 1, f"Pair-sample loss at least need to use two samples but only one sample detected."
                # TODO: not sure if this loss is ok; 
                consistence_loss = ConsistenceBCE(seg_output, output['reg'])
            else:
                consistence_loss = torch.zeros_like(segmentation_loss)
            consistence_loss *= self.pair_loss_weight

            l = segmentation_loss + reg_loss + consistence_loss
            

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        # self.print_to_log_file("Comutation done for current batch")
        

        result_dict=({'reg_loss': reg_loss.detach().cpu().numpy(),
                        'seg_loss': segmentation_loss.detach().cpu().numpy(),
                        'consistence_loss': consistence_loss.detach().cpu().numpy(),
                        'loss': l.detach().cpu().numpy()})
        return result_dict
  

    def train_step_stage2(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']
        aux = batch['aux']


        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)
        if isinstance(aux, list):
            aux = [i.to(self.device, non_blocking=True) for i in aux]
        elif isinstance(aux, dict):
            for i,(k,v) in enumerate(aux.items()):
                aux[k] = v.to(self.device, non_blocking=True)
                if self.aux_type[i]=='seg':
                    aux[k] = aux[k].long()
        else:
            aux = [aux.to(self.device, non_blocking=True)]

        
        self.optimizer.zero_grad()
        # Autocast is a little bitch.
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            label_ = target[0].squeeze(dim=1).long()
            label_ = F.one_hot(label_, num_classes=self.label_manager.num_segmentation_heads).permute(0, 4, 1, 2, 3).float()
            if self.is_ddp:
                output, time, noise = self.network.module.forward_train_val(image=data, label=label_, coord=aux[self.aux_indicator[0]])
            else:
                output, time, noise = self.network.forward_train_val(image=data, label=label_, coord=aux[self.aux_indicator[0]])
            
            del data

            # segmentation loss;
            if isinstance(output, dict):
                seg_output = output['seg']
            else:
                seg_output = output
            segmentation_loss = self.loss([seg_output], [target[0]])
            segmentation_loss *= self.seg_loss_weight
            
            # coordinate regression loss;
            if self.reg_loss_weight > 0:
                reg_loss = F.mse_loss(output['reg'], aux[self.aux_indicator[0]])
            else:
                reg_loss = torch.zeros_like(segmentation_loss)
            reg_loss *= self.reg_loss_weight

            # seg and coord consistence loss;
            # TODO: this consistence loss need to be improved; this is just a simple version;
            if self.pair_loss_weight > 0:
                assert seg_output.shape[0] > 1, f"Pair-sample loss at least need to use two samples but only one sample detected."
                # TODO: not sure if this loss is ok; 
                consistence_loss = ConsistenceBCE(seg_output, output['reg'])
            else:
                consistence_loss = torch.zeros_like(segmentation_loss)
            consistence_loss *= self.pair_loss_weight

            l = segmentation_loss + reg_loss + consistence_loss
            

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        # self.print_to_log_file("Comutation done for current batch")
        

        result_dict=({'reg_loss': reg_loss.detach().cpu().numpy(),
                        'seg_loss': segmentation_loss.detach().cpu().numpy(),
                        'consistence_loss': consistence_loss.detach().cpu().numpy(),
                        'loss': l.detach().cpu().numpy()})
        return result_dict


    def on_train_epoch_end(self, train_outputs: List[dict]):
        outputs = collate_outputs(train_outputs)

        result_dict = dict()

        if self.is_ddp:
            for k,_ in outputs.items():
                temp = [None for _ in range(dist.get_world_size())]
                dist.all_gather_object(temp, outputs[k])
                result_dict[k] = np.vstack(temp).mean()
            # losses_tr = [None for _ in range(dist.get_world_size())]
            # dist.all_gather_object(losses_tr, outputs['loss'])
            # loss_here = np.vstack(losses_tr).mean()

            # if self.get_mse:
            #     mse_loss = [None for _ in range(dist.get_world_size())]
            #     vb_loss = [None for _ in range(dist.get_world_size())]
            #     dist.all_gather_object(mse_loss, outputs['mse_loss'])
            #     dist.all_gather_object(vb_loss, outputs['vb_loss'])
            #     mse_loss = np.vstack(mse_loss).mean()
            #     vb_loss = np.vstack(vb_loss).mean()
            #     if self.seg_loss_weight > 0:
            #         seg_loss = [None for _ in range(dist.get_world_size())]
            #         diff_loss = [None for _ in range(dist.get_world_size())]
            #         dist.all_gather_object(seg_loss, outputs['seg_loss'])
            #         dist.all_gather_object(diff_loss, outputs['diff_loss'])
            #         seg_loss = np.vstack(seg_loss).mean()
            #         diff_loss = np.vstack(diff_loss).mean()
            
        else:
            for k,v in outputs.items():
                result_dict[k] = np.mean(v)
            # loss_here = np.mean(outputs['loss'])
            # if self.get_mse:
            #     mse_loss = np.mean(outputs['mse_loss'])
            #     vb_loss = np.mean(outputs['vb_loss'])
            #     if self.seg_loss_weight > 0:
            #         seg_loss = np.mean(outputs['seg_loss'])
            #         diff_loss = np.mean(outputs['diff_loss'])

    
        # self.logger.log('train_losses', loss_here, self.current_epoch)
        # if self.get_mse:
        #     self.logger.log('train_mse_losses', mse_loss, self.current_epoch)
        #     self.logger.log('train_vb_losses', vb_loss, self.current_epoch)
        #     if self.seg_loss_weight > 0:
        #         self.logger.log('train_seg_losses', seg_loss, self.current_epoch)
        #         self.logger.log('train_diff_losses', diff_loss, self.current_epoch)      

        for k,v in result_dict.items():
            self.logger.log(k, v, self.current_epoch)         

    def on_validation_epoch_start(self):
        self.network.eval()

    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']
        aux = batch['aux']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)
        if isinstance(aux, list):
            aux = [i.to(self.device, non_blocking=True) for i in aux]
        elif isinstance(aux, dict):
            for i,(k,v) in enumerate(aux.items()):
                aux[k] = v.to(self.device, non_blocking=True)
                if self.aux_type[i]=='seg':
                    aux[k] = aux[k].long()
        else:
            aux = [aux.to(self.device, non_blocking=True)]

        context, control = OrderedDict(), OrderedDict()
        for i in self.context_indicator:
            if i in aux:
                context[i] = aux[i]
            elif i=='image':
                context[i] = data
            else:
                NotImplementedError(f"Not support context indicator: {i}")

        for i in self.control_indicator:
            if i in aux:
                control[i] = aux[i]
            elif i=='image':
                control[i] = data
            else:
                NotImplementedError(f"Not support control indicator: {i}")

        # Autocast is a little bitch.
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented,
        # even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.


        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            label_ = target[0].squeeze(dim=1).long()
            label_ = F.one_hot(label_, num_classes=self.label_manager.num_segmentation_heads).permute(0, 4, 1, 2, 3).float()
            if self.is_ddp:
                output, time, noise = self.network.module.forward_train_val(image=data, label=label_, coord=aux[self.aux_indicator[0]])
            else:
                output, time, noise = self.network.forward_train_val(image=data, label=label_, coord=aux[self.aux_indicator[0]])
            
            del data

            # segmentation loss;
            if isinstance(output, dict):
                seg_output = output['seg']
            else:
                seg_output = output
            segmentation_loss = self.loss([seg_output], [target[0]])
            segmentation_loss *= self.seg_loss_weight
            
            # coordinate regression loss;
            if self.reg_loss_weight > 0:
                reg_loss = F.mse_loss(output['reg'], aux[self.aux_indicator[0]])
            else:
                reg_loss = torch.zeros_like(segmentation_loss)
            reg_loss *= self.reg_loss_weight

            # seg and coord consistence loss;
            if self.pair_loss_weight > 0:
                assert seg_output.shape[0] > 1, f"Pair-sample loss at least need to use two samples but only one sample detected."
                # TODO: not sure if this loss is ok; 
                consistence_loss = ConsistenceBCE(seg_output, output['reg'])
            else:
                consistence_loss = torch.zeros_like(segmentation_loss)
            consistence_loss *= self.pair_loss_weight

            l = segmentation_loss + reg_loss + consistence_loss


        # we only need the output with the highest output resolution
        target = target[0]

        output = seg_output
        # the following is needed for online evaluation. Fake dice (green line)
        axes = [0] + list(range(2, len(output.shape)))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            # no need for softmax
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                # CAREFUL that you don't rely on target after this line!
                target[target == self.label_manager.ignore_label] = 0
            else:
                mask = 1 - target[:, -1:]
                # CAREFUL that you don't rely on target after this line!
                target = target[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            # if we train with regions all segmentation heads predict some kind of foreground. In conventional
            # (softmax training) there needs tobe one output for the background. We are not interested in the
            # background Dice
            # [1:] in order to remove background
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        result_dict=({'reg_loss': reg_loss.detach().cpu().numpy(),
            'seg_loss': segmentation_loss.detach().cpu().numpy(),
            'consistence_loss': consistence_loss.detach().cpu().numpy(),
            'loss': l.detach().cpu().numpy(),
            'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard})

        return result_dict

    def on_validation_epoch_end(self, val_outputs: List[dict]):
        outputs_collated = collate_outputs(val_outputs)
        tp = np.sum(outputs_collated['tp_hard'], 0)
        fp = np.sum(outputs_collated['fp_hard'], 0)
        fn = np.sum(outputs_collated['fn_hard'], 0)

        if self.is_ddp:
            world_size = dist.get_world_size()

            tps = [None for _ in range(world_size)]
            dist.all_gather_object(tps, tp)
            tp = np.vstack([i[None] for i in tps]).sum(0)

            fps = [None for _ in range(world_size)]
            dist.all_gather_object(fps, fp)
            fp = np.vstack([i[None] for i in fps]).sum(0)

            fns = [None for _ in range(world_size)]
            dist.all_gather_object(fns, fn)
            fn = np.vstack([i[None] for i in fns]).sum(0)

            result_dict = dict()
            for k,v in outputs_collated.items():
                if 'loss' in k:
                    temp = [None for _ in range(world_size)]
                    dist.all_gather_object(temp, v)
                    if 'val' not in k: k = f"{k}_val"
                    result_dict[k] = np.vstack(temp).mean()

            # losses_val = [None for _ in range(world_size)]
            # dist.all_gather_object(losses_val, outputs_collated['loss'])
            # loss_here = np.vstack(losses_val).mean()
        else:
            # loss_here = np.mean(outputs_collated['loss'])

            result_dict = dict()
            for k,v in outputs_collated.items():
                if 'loss' in k:
                    if 'val' not in k: k = f"{k}_val"
                    result_dict[k] = np.mean(v)

        global_dc_per_class = [i for i in [2 * i / (2 * i + j + k) for i, j, k in
                                           zip(tp, fp, fn)]]
        mean_fg_dice = np.nanmean(global_dc_per_class)
        self.logger.log('mean_fg_dice', mean_fg_dice, self.current_epoch)
        self.logger.log('dice_per_class_or_region', global_dc_per_class, self.current_epoch)
        # self.logger.log('val_losses', loss_here, self.current_epoch)
        for k,v in result_dict.items():
            self.logger.log(k, v, self.current_epoch)

    def on_epoch_start(self):
        self.logger.log('epoch_start_timestamps', time(), self.current_epoch)

    def on_epoch_end(self):
        self.logger.log('epoch_end_timestamps', time(), self.current_epoch)

        # todo find a solution for this stupid shit
        # self.print_to_log_file('train_loss', np.round(self.logger.my_fantastic_logging['train_losses'][-1], decimals=4))
        # if self.get_mse:
        #     self.print_to_log_file('train_mse_loss', np.round(self.logger.my_fantastic_logging['train_mse_losses'][-1], decimals=4))
        #     self.print_to_log_file('train_vb_loss', np.round(self.logger.my_fantastic_logging['train_vb_losses'][-1], decimals=4))
        #     if self.seg_loss_weight > 0:
        #         self.print_to_log_file('train_seg_loss', np.round(self.logger.my_fantastic_logging['train_seg_losses'][-1], decimals=4))
        #         self.print_to_log_file('train_diff_loss', np.round(self.logger.my_fantastic_logging['train_diff_losses'][-1], decimals=4))
        # self.print_to_log_file('val_loss', np.round(self.logger.my_fantastic_logging['val_losses'][-1], decimals=4))
        self.print_to_log_file('*' *50)
        for k in self.logger.aux_metrics:
            self.print_to_log_file(k, np.round(self.logger.my_fantastic_logging[k][-1], decimals=4))
        self.print_to_log_file('*' *50)
        self.print_to_log_file('Pseudo dice', [np.round(i, decimals=4) for i in
                                               self.logger.my_fantastic_logging['dice_per_class_or_region'][-1]])
        self.print_to_log_file(
            f"Epoch time: {np.round(self.logger.my_fantastic_logging['epoch_end_timestamps'][-1] - self.logger.my_fantastic_logging['epoch_start_timestamps'][-1], decimals=2)} s")

        # handling periodic checkpointing
        current_epoch = self.current_epoch
        if (current_epoch + 1) % self.save_every == 0 and current_epoch != (self.num_epochs - 1):
            self.save_checkpoint(join(self.output_folder, 'checkpoint_latest.pth'))

        # handle 'best' checkpointing. ema_fg_dice is computed by the logger and can be accessed like this
        if self._best_ema is None or self.logger.my_fantastic_logging['ema_fg_dice'][-1] > self._best_ema:
            self._best_ema = self.logger.my_fantastic_logging['ema_fg_dice'][-1]
            self.print_to_log_file(f"Yayy! New best EMA pseudo Dice: {np.round(self._best_ema, decimals=4)}")
            self.save_checkpoint(join(self.output_folder, 'checkpoint_best.pth'))

        if self.local_rank == 0:
            self.logger.plot_progress_png(self.output_folder)

        self.current_epoch += 1

    def save_checkpoint(self, filename: str) -> None:
        if self.local_rank == 0:
            if not self.disable_checkpointing:
                if self.is_ddp:
                    mod = self.network.module
                else:
                    mod = self.network
                if isinstance(mod, OptimizedModule):
                    mod = mod._orig_mod

                checkpoint = {
                    'network_weights': mod.state_dict(),
                    'optimizer_state': self.optimizer.state_dict(),
                    'grad_scaler_state': self.grad_scaler.state_dict() if self.grad_scaler is not None else None,
                    'logging': self.logger.get_checkpoint(),
                    '_best_ema': self._best_ema,
                    'current_epoch': self.current_epoch + 1,
                    'init_args': self.my_init_kwargs,
                    'trainer_name': self.__class__.__name__,
                    'inference_allowed_mirroring_axes': self.inference_allowed_mirroring_axes,
                }
                torch.save(checkpoint, filename)
            else:
                self.print_to_log_file('No checkpoint written, checkpointing is disabled')

    def load_checkpoint(self, filename_or_checkpoint: Union[dict, str]) -> None:
        if not self.was_initialized:
            self.initialize()

        if isinstance(filename_or_checkpoint, str):
            checkpoint = torch.load(filename_or_checkpoint, map_location=self.device)
        # if state dict comes from nn.DataParallel but we use non-parallel model here then the state dict keys do not
        # match. Use heuristic to make it match
        new_state_dict = {}
        for k, value in checkpoint['network_weights'].items():
            key = k
            if key not in self.network.state_dict().keys() and key.startswith('module.'):
                key = key[7:]
            new_state_dict[key] = value

        self.my_init_kwargs = checkpoint['init_args']
        self.current_epoch = checkpoint['current_epoch']
        self.logger.load_checkpoint(checkpoint['logging'])
        self._best_ema = checkpoint['_best_ema']
        self.inference_allowed_mirroring_axes = checkpoint[
            'inference_allowed_mirroring_axes'] if 'inference_allowed_mirroring_axes' in checkpoint.keys() else self.inference_allowed_mirroring_axes

        # messing with state dict naming schemes. Facepalm.
        if self.is_ddp:
            if isinstance(self.network.module, OptimizedModule):
                self.network.module._orig_mod.load_state_dict(new_state_dict)
            else:
                self.network.module.load_state_dict(new_state_dict)
        else:
            if isinstance(self.network, OptimizedModule):
                self.network._orig_mod.load_state_dict(new_state_dict)
            else:
                self.network.load_state_dict(new_state_dict)
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        if self.grad_scaler is not None:
            if checkpoint['grad_scaler_state'] is not None:
                self.grad_scaler.load_state_dict(checkpoint['grad_scaler_state'])


    def simple_load_checkpoint(self, filename_or_checkpoint: Union[dict, str]) -> None:
        if isinstance(filename_or_checkpoint, str):
            checkpoint = torch.load(filename_or_checkpoint, map_location=self.device)
        # if state dict comes from nn.DataParallel but we use non-parallel model here then the state dict keys do not
        # match. Use heuristic to make it match
        new_state_dict = {}
        for k, value in checkpoint['network_weights'].items():
            key = k
            if key not in self.network.state_dict().keys() and key.startswith('module.'):
                key = key[7:]
            new_state_dict[key] = value

        # messing with state dict naming schemes. Facepalm.
        if self.is_ddp:
            if isinstance(self.network.module, OptimizedModule):
                self.network.module._orig_mod.load_state_dict(new_state_dict)
            else:
                self.network.module.load_state_dict(new_state_dict, strict=False)
        else:
            if isinstance(self.network, OptimizedModule):
                self.network._orig_mod.load_state_dict(new_state_dict)
            else:
                self.network.load_state_dict(new_state_dict, strict=False)



    def perform_actual_validation(self, save_probabilities: bool = False):
        self.set_deep_supervision_enabled(False)
        self.network.eval()

        num_seg_heads = self.label_manager.num_segmentation_heads

        with multiprocessing.get_context("spawn").Pool(default_num_processes) as segmentation_export_pool:
            if self.val_mannual_name is not None:
                validation_output_folder = join(self.output_folder, f'validation_{self.step}step_{self.val_mannual_name}')
            else:
                validation_output_folder = join(self.output_folder, 'validation')
            maybe_mkdir_p(validation_output_folder)

            # we cannot use self.get_tr_and_val_datasets() here because we might be DDP and then we have to distribute
            # the validation keys across the workers.
            _, val_keys = self.do_split()
            if self.is_ddp:
                val_keys = val_keys[self.local_rank:: dist.get_world_size()]

            dataset_val = nnUNetDataset(self.preprocessed_dataset_folder, val_keys,
                                        folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
                                        num_images_properties_loading_threshold=0,
                                        aux_indicator=self.aux_indicator,
                                        aux_type=self.aux_type)

            next_stages = self.configuration_manager.next_stage_names

            if next_stages is not None:
                _ = [maybe_mkdir_p(join(self.output_folder_base, 'predicted_next_stage', n)) for n in next_stages]

            results = []
            for k in dataset_val.keys():
                proceed = not check_workers_busy(segmentation_export_pool, results,
                                                 allowed_num_queued=2 * len(segmentation_export_pool._pool))
                while not proceed:
                    sleep(0.1)
                    proceed = not check_workers_busy(segmentation_export_pool, results,
                                                     allowed_num_queued=2 * len(segmentation_export_pool._pool))

                self.print_to_log_file(f"predicting {k}")
                data, seg, aux, properties = dataset_val.load_case(k)

                model_kwargs = {'noisy_coord':aux[self.aux_indicator[0]], 
                                'coord_time':torch.zeros((data.shape[0],), dtype=torch.int, device=self.device)}

                if self.is_cascaded:
                    data = np.vstack((data, convert_labelmap_to_one_hot(seg[-1], self.label_manager.foreground_labels,
                                                                        output_dtype=data.dtype)))

                output_filename_truncated = join(validation_output_folder, k)

                try:                    
                    prediction = predict_sliding_window_return_logits(self.network, data, num_seg_heads,
                                                                      tile_size=self.configuration_manager.patch_size,
                                                                      mirror_axes=self.inference_allowed_mirroring_axes if self.infer_mirror else None,
                                                                      tile_step_size=0.5,
                                                                      use_gaussian=True,
                                                                      precomputed_gaussian=None,
                                                                      perform_everything_on_gpu=True,
                                                                      verbose=False,
                                                                      device=self.device,
                                                                      model_kwargs=model_kwargs
                                                                      ).cpu().numpy()
                except RuntimeError:
                    prediction = predict_sliding_window_return_logits(self.network, data, num_seg_heads,
                                                                      tile_size=self.configuration_manager.patch_size,
                                                                      mirror_axes=self.inference_allowed_mirroring_axes if self.infer_mirror else None,
                                                                      tile_step_size=0.5,
                                                                      use_gaussian=True,
                                                                      precomputed_gaussian=None,
                                                                      perform_everything_on_gpu=False,
                                                                      verbose=False,
                                                                      device=self.device,
                                                                      context=context,
                                                                      context_type=self.context_type,
                                                                      context_default_value=context_default_value,
                                                                      control=control,
                                                                      control_type=self.control_type,
                                                                      control_default_value=control_default_value
                                                                      ).cpu().numpy()

                # this needs to go into background processes
                results.append(
                    segmentation_export_pool.starmap_async(
                        export_prediction_from_logits, (
                            (prediction, properties, self.configuration_manager, self.plans_manager,
                             self.dataset_json, output_filename_truncated, save_probabilities),
                        )
                    )
                )
                # for debug purposes
                # export_prediction(prediction_for_export, properties, self.configuration, self.plans, self.dataset_json,
                #              output_filename_truncated, save_probabilities)

                # if needed, export the softmax prediction for the next stage
                if next_stages is not None:
                    for n in next_stages:
                        next_stage_config_manager = self.plans_manager.get_configuration(n)
                        expected_preprocessed_folder = join(nnUNet_preprocessed, self.plans_manager.dataset_name,
                                                            next_stage_config_manager.data_identifier)

                        try:
                            # we do this so that we can use load_case and do not have to hard code how loading training cases is implemented
                            tmp = nnUNetDataset(expected_preprocessed_folder, [k],
                                                num_images_properties_loading_threshold=0)
                            d, s, p = tmp.load_case(k)
                        except FileNotFoundError:
                            self.print_to_log_file(
                                f"Predicting next stage {n} failed for case {k} because the preprocessed file is missing! "
                                f"Run the preprocessing for this configuration first!")
                            continue

                        target_shape = d.shape[1:]
                        output_folder = join(self.output_folder_base, 'predicted_next_stage', n)
                        output_file = join(output_folder, k + '.npz')

                        # resample_and_save(prediction, target_shape, output_file, self.plans, self.configuration, properties,
                        #                   self.dataset_json)
                        results.append(segmentation_export_pool.starmap_async(
                            resample_and_save, (
                                (prediction, target_shape, output_file, self.plans_manager,
                                 self.configuration_manager,
                                 properties,
                                 self.dataset_json),
                            )
                        ))

            _ = [r.get() for r in results]

        if self.is_ddp:
            dist.barrier()

        if self.eval_bg:
            eval_label_list = [0] + \
            (self.label_manager.foreground_regions if self.label_manager.has_regions else self.label_manager.foreground_labels)
        else:
            eval_label_list = self.label_manager.foreground_regions if self.label_manager.has_regions else self.label_manager.foreground_labels




        if self.local_rank == 0:
            metrics = compute_metrics_on_folder(join(self.preprocessed_dataset_folder_base, 'gt_segmentations'),
                                                validation_output_folder,
                                                join(validation_output_folder, 'summary.json'),
                                                self.plans_manager.image_reader_writer_class(),
                                                self.dataset_json["file_ending"],
                                                eval_label_list,
                                                self.label_manager.ignore_label, chill=True)
            self.print_to_log_file("Validation complete", also_print_to_console=True)
            self.print_to_log_file("Mean Validation Dice: ", (metrics['foreground_mean']["Dice"]), also_print_to_console=True)
            if self.eval_bg:
                self.print_to_log_file("Mean Validation Dice With Background: ", (metrics['all_class_mean']["Dice"]), also_print_to_console=True)

        self.set_deep_supervision_enabled(True)
        compute_gaussian.cache_clear()

    def run_training(self):
        self.on_train_start()

        for epoch in range(self.current_epoch, self.num_epochs):

            if not self.training_stage == 'stage2':
                train_outputs = []
                self.on_epoch_start()
                self.on_train_epoch_start()
                for batch_id in range(self.num_iterations_per_epoch):
                    train_outputs.append(self.train_step(next(self.dataloader_train)))


            else:
                train_outputs = []
                self.on_epoch_start()
                self.on_train_epoch_start()
                self.print_to_log_file(f"Stage 2 training.")
                for batch_id in range(self.num_iterations_per_epoch):
                    train_outputs.append(self.train_step_stage2(next(self.dataloader_train)))


            # for name, param in self.network.named_parameters():
            #     if param.grad is None:
            #         print(name)


            self.on_train_epoch_end(train_outputs)

            with torch.no_grad():
                if self.is_ddp:
                    ds_state = self.network.module.train_ds
                else:
                    ds_state = self.network.train_ds

                self.network.train_ds = False
                self.on_validation_epoch_start()
                val_outputs = []
                for batch_id in range(self.num_val_iterations_per_epoch):
                    val_outputs.append(self.validation_step(next(self.dataloader_val)))
                self.on_validation_epoch_end(val_outputs)

                if self.is_ddp:
                    self.network.module.train_ds = ds_state
                else:
                    self.network.train_ds = ds_state

            self.on_epoch_end()

        self.on_train_end()



        

def ConsistenceBCE(batch_seg,batch_reg):
    b,c,h,w,d = batch_seg.shape
    batch_seg_norm = F.normalize(batch_seg, p=2, dim=1)
    total_loss = 0
    count = 0

    for i in range(b):
        for j in range(b):
            if i!=j:
                cosine_sim = F.cosine_similarity(batch_seg_norm[i], batch_seg_norm[j],dim=0)
                cosine_sim = torch.mean(F.relu(cosine_sim))

                loc_distance = torch.sqrt(torch.sum(torch.square(batch_reg[i] - batch_reg[j]),dim=0))
                loc_distance = torch.mean(loc_distance)

                consistence_loss = bce_loss_with_logits(1-loc_distance, cosine_sim)
                total_loss += consistence_loss
                count += 1
    total_loss /= count
    return total_loss


def bce_loss_with_logits(logits, targets):

    loss = targets * torch.log(logits) + (1 - targets) * torch.log(1-logits)
    return -loss.mean()

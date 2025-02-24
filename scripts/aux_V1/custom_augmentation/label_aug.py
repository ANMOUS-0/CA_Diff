from batchgenerators.transforms.abstract_transforms import AbstractTransform
import numpy as np
import scipy.ndimage as ndi
import random
import itertools


class LabelDisrupt(AbstractTransform):
    '''
    To disrupt label to simulate non-ground-truth label.

    I think the augmentation may not simulate the wrong label perfectly. Hope the distruction can push the diffusion 
    model to be able to distringuish wrong label and learn to rectify them.
    '''

    def __init__(self,
                 label_key="seg",
                 output_key="seg_disrupt",
                 dim:int=3,
                 patch_size=None,
                #  label_num:int=None,
                 p_morph_per_sample:float=1,
                 p_morph_per_class:float=1,
                 p_label_noise_per_sample:float=1,
                 p_label_noise_per_class:float=1,
                 disrupt_bg:bool=False,
                 noise_label_rate:float=0.2,
                 ):
        
        
        self.label_key = label_key
        self.output_key = output_key
        self.dim = dim
        self.patch_size = patch_size
        # self.label_num = label_num
        self.p_morph_per_sample = p_morph_per_sample
        self.p_morph_per_class = p_morph_per_class
        self.p_label_noise_per_sample = p_label_noise_per_sample
        self.p_label_noise_per_class = p_label_noise_per_class
        self.disrupt_bg = disrupt_bg
        self.noise_label_rate = noise_label_rate

    

    def morph_disrupt(self, label_map):

        # TODO: maybe add more kernel choice;
        if np.random.rand() < 0.5:
            kernel_size = (3,3,3)
        else:
            kernel_size = (5,5,5)

        augmented_label_map = np.zeros_like(label_map)
        unique_labels = np.unique(label_map)
        # cropped patch may lead to a patch may not contain all classes;
        for i in unique_labels:
            if i > 0:
                class_mask = (label_map == i)
                if np.random.rand() < self.p_morph_per_class:
                    # random select dilation or erosion;
                    if np.random.rand() > 0.5:
                        morph_mask = ndi.grey_dilation(class_mask, size=kernel_size)
                    else:
                        morph_mask = ndi.grey_erosion(class_mask, size=kernel_size)
                else:
                    # otherwise just use the raw label mask;
                    morph_mask = class_mask
                # get overlap area;
                overlap_mask = (augmented_label_map > 0) & (morph_mask > 0)
                # random keep existed overlap class; 
                replace_mask = np.random.rand(*label_map.shape) > 0.5
                # replace_mask is current mask area but overlap with former classes;
                replace_mask &= overlap_mask
                # 3 conditions: overelap and replace area will use new class, 
                # overlap but not replace will keep former existed class,
                # the rest non-overlap area will use new class;
                augmented_label_map[replace_mask] = i
                augmented_label_map[morph_mask & ~overlap_mask] = i

        return augmented_label_map
    


    def add_label_noise(self, label_map):
        # set disrupt_bg = True to disrupt background area;
        if self.disrupt_bg:
            total_voxels = label_map.size
        else:
            foreground_indices = np.argwhere(label_map > 0)
            total_voxels = len(foreground_indices)

        # amount of noise;
        num_noise = int(self.noise_label_rate * total_voxels)
        if self.disrupt_bg:
            noise_indices = np.random.choice(total_voxels, num_noise, replace=False)
        else:
            noise_voxel_indices = np.random.choice(total_voxels, num_noise, replace=False)
            noise_indices = foreground_indices[noise_voxel_indices]

        # random label; TODO: does background need to be set one of random class choice?
        unique_labels = np.unique(label_map)
        noise_values = np.random.choice(unique_labels, num_noise)

        noisy_label_map = label_map.copy()
        if self.disrupt_bg:
            flat_noisy_label_map = noisy_label_map.ravel()
            flat_noisy_label_map[noise_indices] = noise_values
            return noisy_label_map.reshape(label_map.shape)
        else:
            noisy_label_map[noise_indices[:,0], noise_indices[:,1], noise_indices[:,2]] = noise_values
            return noisy_label_map
            


    def __call__(self, **data_dict):

        aug_flag = False
        seg = data_dict.get(self.label_key)


        if self.dim == 2:
            seg_result = np.zeros((seg.shape[0], seg.shape[1], self.patch_size[0], self.patch_size[1]), dtype=np.float32)
        else:
            seg_result = np.zeros((seg.shape[0], seg.shape[1], self.patch_size[0], self.patch_size[1], self.patch_size[2]),
                                  dtype=np.float32)
            

        for sample_id in range(seg.shape[0]):
            for channel_id in range(seg.shape[1]):
                if np.random.rand() < self.p_morph_per_sample:
                    aug_flag = True
                    seg_result[sample_id, channel_id] = self.morph_disrupt(seg[sample_id, channel_id])
                if np.random.rand() < self.p_label_noise_per_sample:
                    aug_flag = True
                    seg_result[sample_id, channel_id] = self.add_label_noise(seg[sample_id, channel_id])

        if aug_flag:
            data_dict.update({f"{self.output_key}":seg_result})
        else:
            data_dict.update({f'{self.output_key}':seg})

        return data_dict

                    




class LabelDisrupt_V1(AbstractTransform):
    '''
    To disrupt label to simulate non-ground-truth label.

    The morph operation is enhanced by support more kernel sizes using random combination
    and independent kernel size for each class.
    '''

    def __init__(self,
                 label_key="seg",
                 output_key="seg_disrupt",
                 dim:int=3,
                 patch_size=None,
                #  label_num:int=None,
                 p_morph_per_sample:float=1,
                 p_morph_per_class:float=1,
                 p_label_noise_per_sample:float=1,
                 p_label_noise_per_class:float=1,
                 disrupt_bg:bool=False,
                 noise_label_rate:float=0.2,
                 kernel_choices:list=None,
                 ):
        
        
        self.label_key = label_key
        self.output_key = output_key
        self.dim = dim
        self.patch_size = patch_size
        # self.label_num = label_num
        self.p_morph_per_sample = p_morph_per_sample
        self.p_morph_per_class = p_morph_per_class
        self.p_label_noise_per_sample = p_label_noise_per_sample
        self.p_label_noise_per_class = p_label_noise_per_class
        self.disrupt_bg = disrupt_bg
        self.noise_label_rate = noise_label_rate

        self.kernel_choices = list(itertools.product(kernel_choices, repeat=3)) \
            or [(3,3,3), (5,5,5), (7,7,7)]

    

    def morph_disrupt(self, label_map):

        augmented_label_map = np.zeros_like(label_map)
        unique_labels = np.unique(label_map)
        # cropped patch may lead to a patch may not contain all classes;
        for i in unique_labels:
            if i > 0:
                class_mask = (label_map == i)
                if np.random.rand() < self.p_morph_per_class:
                    # make each class can have independent kernel size;
                    kernel_size = random.choice(self.kernel_choices)
                    # random select dilation or erosion;
                    if np.random.rand() > 0.5:
                        morph_mask = ndi.grey_dilation(class_mask, size=kernel_size)
                    else:
                        morph_mask = ndi.grey_erosion(class_mask, size=kernel_size)
                else:
                    # otherwise just use the raw label mask;
                    morph_mask = class_mask
                # get overlap area;
                overlap_mask = (augmented_label_map > 0) & (morph_mask > 0)
                # random keep existed overlap class; 
                replace_mask = np.random.rand(*label_map.shape) > 0.5
                # replace_mask is current mask area but overlap with former classes;
                replace_mask &= overlap_mask
                # 3 conditions: overelap and replace area will use new class, 
                # overlap but not replace will keep former existed class,
                # the rest non-overlap area will use new class;
                augmented_label_map[replace_mask] = i
                augmented_label_map[morph_mask & ~overlap_mask] = i

        return augmented_label_map
    


    def add_label_noise(self, label_map):
        # set disrupt_bg = True to disrupt background area;
        if self.disrupt_bg:
            total_voxels = label_map.size
        else:
            foreground_indices = np.argwhere(label_map > 0)
            total_voxels = len(foreground_indices)

        # amount of noise;
        num_noise = int(self.noise_label_rate * total_voxels)
        if self.disrupt_bg:
            noise_indices = np.random.choice(total_voxels, num_noise, replace=False)
        else:
            noise_voxel_indices = np.random.choice(total_voxels, num_noise, replace=False)
            noise_indices = foreground_indices[noise_voxel_indices]

        # random label; TODO: does background need to be set one of random class choice?
        unique_labels = np.unique(label_map)
        noise_values = np.random.choice(unique_labels, num_noise)

        noisy_label_map = label_map.copy()
        if self.disrupt_bg:
            flat_noisy_label_map = noisy_label_map.ravel()
            flat_noisy_label_map[noise_indices] = noise_values
            return noisy_label_map.reshape(label_map.shape)
        else:
            noisy_label_map[noise_indices[:,0], noise_indices[:,1], noise_indices[:,2]] = noise_values
            return noisy_label_map
            


    def __call__(self, **data_dict):

        aug_flag = False
        seg = data_dict.get(self.label_key)


        if self.dim == 2:
            seg_result = np.zeros((seg.shape[0], seg.shape[1], self.patch_size[0], self.patch_size[1]), dtype=np.float32)
        else:
            seg_result = np.zeros((seg.shape[0], seg.shape[1], self.patch_size[0], self.patch_size[1], self.patch_size[2]),
                                  dtype=np.float32)
            

        for sample_id in range(seg.shape[0]):
            for channel_id in range(seg.shape[1]):
                if np.random.rand() < self.p_morph_per_sample:
                    aug_flag = True
                    seg_result[sample_id, channel_id] = self.morph_disrupt(seg[sample_id, channel_id])
                if np.random.rand() < self.p_label_noise_per_sample:
                    aug_flag = True
                    seg_result[sample_id, channel_id] = self.add_label_noise(seg[sample_id, channel_id])

        if aug_flag:
            data_dict.update({f"{self.output_key}":seg_result})
        else:
            data_dict.update({f'{self.output_key}':seg})

        return data_dict

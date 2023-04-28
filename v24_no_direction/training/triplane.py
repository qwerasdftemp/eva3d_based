# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import torch
from torch_utils import persistence
from training.networks_stylegan2 import Generator as StyleGAN2Backbone,MappingNetwork
from training.volumetric_rendering.renderer import ImportanceRenderer
from training.volumetric_rendering.ray_sampler import RaySampler
import dnnlib
from training import gqz_texture
import flame.FLAME
import torch.nn.functional as F
import my_smpl
import time 
import torch.nn as nn
import numpy as np

@persistence.persistent_class
class TriPlaneGenerator(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality.
        c_dim,                      # Conditioning label (C) dimensionality.
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output resolution.
        img_channels,               # Number of output color channels.
        sr_num_fp16_res     = 0,
        mapping_kwargs      = {},   # Arguments for MappingNetwork.
        rendering_kwargs    = {},
        sr_kwargs = {},
        **synthesis_kwargs,         # Arguments for SynthesisNetwork.
    ):
        super().__init__()
        self.z_dim=z_dim
        self.c_dim=c_dim
        self.w_dim=w_dim
        self.img_resolution=img_resolution
        self.img_channels=img_channels
        self.renderer = ImportanceRenderer()
        self.ray_sampler = RaySampler()
        self.backbone = StyleGAN2Backbone(z_dim, c_dim, w_dim, img_resolution=256, img_channels=32*3, mapping_kwargs=mapping_kwargs, **synthesis_kwargs)
        self.superresolution = dnnlib.util.construct_class_by_name(class_name=rendering_kwargs['superresolution_module'], channels=32, img_resolution=img_resolution, sr_num_fp16_res=sr_num_fp16_res, sr_antialias=rendering_kwargs['sr_antialias'], **sr_kwargs)
        self.decoder = OSGDecoder(32, {'decoder_lr_mul': rendering_kwargs.get('decoder_lr_mul', 1), 'decoder_output_dim': 3})
        self.neural_rendering_resolution = 512
        self.rendering_kwargs = rendering_kwargs
        self.rendering_kwargs['feature_dim'] = 3

        self._last_planes = None
        self.mapping_net = MappingNetwork(z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, num_ws=24, **mapping_kwargs)

        self.tex_feature =  gqz_texture.Texture_Feature(z_dim=z_dim, c_dim=c_dim, w_dim=w_dim,mapping_kwargs=mapping_kwargs,synthesis_kwargs=synthesis_kwargs)
        self.flame_model =  my_smpl.SMPL()


    def mapping(self, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False):
        if self.rendering_kwargs['c_gen_conditioning_zero']:
                c = torch.zeros_like(c)
        return self.mapping_net(z, c * self.rendering_kwargs.get('c_scale', 0), truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)

    def synthesis(self, ws, c,additional_dict = {}, neural_rendering_resolution=None, update_emas=False, cache_backbone=False, use_cached_backbone=False, **synthesis_kwargs):
        cam2world_matrix = c[:, :16].view(-1, 4, 4)
        intrinsics = c[:, 16:25].view(-1, 3, 3)
        
        if neural_rendering_resolution is None:
            neural_rendering_resolution = self.neural_rendering_resolution
        else:
            self.neural_rendering_resolution = neural_rendering_resolution
        # import pdb; pdb.set_trace()
        # Create a batch of rays for volume rendering
        ray_origins, ray_directions = self.ray_sampler(cam2world_matrix, intrinsics, neural_rendering_resolution)
        # import pdb; pdb.set_trace()
        # Create triplanes by running StyleGAN backbone
        N, M, _ = ray_origins.shape
        if use_cached_backbone and self._last_planes is not None:
            planes = self._last_planes
        else:
            planes = self.backbone.synthesis(ws[:,0:14], update_emas=update_emas, **synthesis_kwargs)
        if cache_backbone:
            self._last_planes = planes
        # texture_map = self.backbone.synthesis(ws[:,0:14], update_emas=update_emas, **synthesis_kwargs)
        # Reshape output into three 32-channel planes
        planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])
        # planes = None
        if 'distance_range' not in self.rendering_kwargs:
            self.rendering_kwargs['distance_range']=0.2
            #print('auto set self.rendering_kwarg distance_range=0.2')
        additional_dict['tex_feature'] = self.tex_feature
        additional_dict['flame_model'] = self.flame_model
        additional_dict['neural_rendering_resolution'] = neural_rendering_resolution
        additional_dict['extrinsic'] = cam2world_matrix
        additional_dict['intrinsic'] = intrinsics
        additional_dict['ws'] = ws
        # additional_dict['texture_map'] = texture_map
        # Perform volume rendering
        # start_time = time.time()
        # import pdb; pdb.set_trace()
        # feature_samples, depth_samples, weights_samples = self.renderer(planes, self.decoder, ray_origins, ray_directions, self.rendering_kwargs,additional_dict=additional_dict) # channels last
        out_dict  = self.renderer(planes, self.decoder, ray_origins, ray_directions, self.rendering_kwargs,additional_dict=additional_dict) # channels last
        
        # #print(time.time()-start_time,'total render time')
        feature_samples = out_dict['rgb_final']
        depth_samples = out_dict['depth_final']
        # weights_samples = out_dict['feature_samples']
   
        # Reshape into 'raw' neural-rendered image
        # H = W = self.neural_rendering_resolution
        H = self.neural_rendering_resolution
        W = self.neural_rendering_resolution//2
        
        
        feature_image = feature_samples.permute(0, 2, 1).reshape(N, feature_samples.shape[-1], H, W).contiguous()
        depth_image = depth_samples.permute(0, 2, 1).reshape(N, 1, H, W)
        image_resolution = self.rendering_kwargs['image_resolution']
        # Run superresolution to get final image
        rgb_image = feature_image[:, :3]
        #print('rgb_image sum',rgb_image.sum())
        #print('feature_image shape',feature_image.shape)
        # sr_image = self.superresolution(rgb_image, feature_image, ws, noise_mode=self.rendering_kwargs['superresolution_noise_mode'], **{k:synthesis_kwargs[k] for k in synthesis_kwargs.keys() if k != 'noise_mode'})
        sr_image = torch.nn.functional.interpolate(rgb_image, size=(image_resolution, image_resolution//2), mode='nearest', antialias=False)

        # sr_image = self.superresolution(rgb_image, feature_image, ws, noise_mode=self.rendering_kwargs['superresolution_noise_mode'], **{k:synthesis_kwargs[k] for k in synthesis_kwargs.keys() if k != 'noise_mode'})
        result_dict =  {'image': sr_image, 'image_raw': rgb_image, 'image_depth': depth_image,'mesh_normal':out_dict['mesh_normal'],'pred_sdf':out_dict['all_pred_sdf']}
        #print('sr sum',sr_image.sum())
        if 'render_normal' in additional_dict:
            # import pdb; pdb.set_trace()
            ray_origins = ray_origins.permute(0, 2, 1).reshape(N, 3, H, W)
            ray_directions = ray_directions.permute(0, 2, 1).reshape(N, 3, H, W)

            img = ray_origins + ray_directions*depth_image
            

            shift_l, shift_r = img[:,:,2:,:], img[:,:,:-2,:]
            shift_u, shift_d = img[:,:,:,2:], img[:,:,:,:-2]
            diff_hor = F.normalize(shift_r - shift_l, dim=1)[:, :, :, 1:-1]
            diff_ver = F.normalize(shift_u - shift_d, dim=1)[:, :, 1:-1, :]
            normal = -torch.cross(diff_hor,diff_ver)
            img = F.normalize(normal,dim=1)
            result_dict['normal'] = img

        return result_dict
    
    def sample(self, coordinates, directions, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        # Compute RGB features, density for arbitrary 3D coordinates. Mostly used for extracting shapes. 
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        planes = self.backbone.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])
        return self.renderer.run_model(planes, self.decoder, coordinates, directions, self.rendering_kwargs)

    def sample_mixed(self, coordinates, directions, ws, truncation_psi=1,all_flame_para=None,truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        # Same as sample, but expects latent vectors 'ws' instead of Gaussian noise 'z'
        planes = self.backbone.synthesis(ws[:,0:14], update_emas = update_emas, **synthesis_kwargs)
        planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])
        # planes = None

        b = coordinates.shape[0]
        # shape = torch.zeros((b,100)).to(coordinates.device)
        # exp = torch.zeros((b,50)).to(coordinates.device)
        # pose = torch.zeros((b,15)).to(coordinates.device)
        shape = all_flame_para[:,0:10]
        pose = all_flame_para[:,10:].view(b,24,3,3)
        # pose = all_flame_para[:,150:165]
        # with torch.
        vertices,canonical_vertices=  self.flame_model(pose,shape)

        tex_feature,template_sdf = self.tex_feature(coordinates,ws,vertices,canonical_vertices)

        return self.renderer.run_model(planes, self.decoder, tex_feature,template_sdf, coordinates, directions, self.rendering_kwargs)

    def forward(self, z, c, truncation_psi=1,additional_dict={},  truncation_cutoff=None, neural_rendering_resolution=None, update_emas=False, cache_backbone=False, use_cached_backbone=False, **synthesis_kwargs):
        # Render a batch of generated images.
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        return self.synthesis(ws, c, additional_dict=additional_dict,update_emas=update_emas, neural_rendering_resolution=neural_rendering_resolution, cache_backbone=cache_backbone, use_cached_backbone=use_cached_backbone, **synthesis_kwargs)


from training.networks_stylegan2 import FullyConnectedLayer
class LinearLayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, bias_init=0, std_init=1, freq_init=False, is_first=False):
        super().__init__()
        if is_first:
            self.weight = nn.Parameter(torch.empty(out_dim, in_dim).uniform_(-1 / in_dim, 1 / in_dim))
        elif freq_init:
            self.weight = nn.Parameter(torch.empty(out_dim, in_dim).uniform_(-np.sqrt(6 / in_dim) / 25, np.sqrt(6 / in_dim) / 25))
        else:
            self.weight = nn.Parameter(0.25 * nn.init.kaiming_normal_(torch.randn(out_dim, in_dim), a=0.2, mode='fan_in', nonlinearity='leaky_relu'))

        self.bias = nn.Parameter(nn.init.uniform_(torch.empty(out_dim), a=-np.sqrt(1/in_dim), b=np.sqrt(1/in_dim)))

        self.bias_init = bias_init
        self.std_init = std_init

    def forward(self, input):
        out = self.std_init * F.linear(input, self.weight, bias=self.bias) + self.bias_init

        return out
class OSGDecoder(torch.nn.Module):
    def __init__(self, n_features, options):
        super().__init__()
        self.hidden_dim = 64
        # self.merge_feature = torch.nn.Sequential(
        #     FullyConnectedLayer(n_features, self.hidden_dim, lr_multiplier=options['decoder_lr_mul']),
        #     torch.nn.Softplus(),
        #     FullyConnectedLayer(self.hidden_dim, 1 + options['decoder_output_dim'], lr_multiplier=options['decoder_lr_mul'])
        # )

        self.net = torch.nn.Sequential(
            FullyConnectedLayer(n_features, self.hidden_dim, lr_multiplier=options['decoder_lr_mul']),
            torch.nn.Softplus(),
            FullyConnectedLayer(self.hidden_dim,self.hidden_dim, lr_multiplier=options['decoder_lr_mul'])
        )
        self.act = torch.nn.Softplus()
        self.sdf_layer =  LinearLayer(self.hidden_dim, 1, freq_init=True)
        self.feature_layer = LinearLayer(self.hidden_dim, options['decoder_output_dim'], freq_init=True)

    def forward(self, sampled_features, ray_directions):
        # Aggregate features
        # sampled_features = sampled_features.mean(1)
        x = sampled_features

        N, M, C = x.shape
        x = x.view(N*M, C)

        x = self.net(x)
        x = self.act(x)
        
        sdf  = self.sdf_layer(x)
        rgb = self.feature_layer(x)

        rgb = torch.sigmoid(rgb)*(1 + 2*0.001) - 0.001
        rgb = rgb.view(N, M, -1)
        sdf = sdf.view(N, M, -1)


        return {'rgb': rgb, 'sigma': sdf}
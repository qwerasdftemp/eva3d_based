# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
The renderer is a module that takes in rays, decides where to sample along each
ray, and computes pixel colors using the volume rendering equation.
"""

import math
import torch
import torch.nn as nn

from training.volumetric_rendering.ray_marcher import MipRayMarcher2
from training.volumetric_rendering import math_utils
from training.render_mesh import Render_Mesh
import os
import my_smpl
import sys
sys.path.append('..')
import clib
import time

import torch.nn.functional as F
def generate_planes():
    """
    Defines planes by the three vectors that form the "axes" of the
    plane. Should work with arbitrary number of planes and planes of
    arbitrary orientation.
    """
    # return torch.tensor([[[1, 0, 0],
    #                         [0, 1, 0],
    #                         [0, 0, 1]],
    #                         [[1, 0, 0],
    #                         [0, 0, 1],
    #                         [0, 1, 0]],
    #                         [[0, 0, 1],
    #                         [1, 0, 0],
    #                         [0, 1, 0]]], dtype=torch.float32)
    return torch.tensor([[[1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1]],
                            [[1, 0, 0],
                            [0, 0, 1],
                            [0, 1, 0]],
                            [[0, 0, 1],
                            [0, 1, 0],
                            [1, 0, 0]]], dtype=torch.float32)

def project_onto_planes(planes, coordinates):
    """
    Does a projection of a 3D point onto a batch of 2D planes,
    returning 2D plane coordinates.

    Takes plane axes of shape n_planes, 3, 3
    # Takes coordinates of shape N, M, 3
    # returns projections of shape N*n_planes, M, 2
    """
    N, M, C = coordinates.shape
    n_planes, _, _ = planes.shape
    coordinates = coordinates.unsqueeze(1).expand(-1, n_planes, -1, -1).reshape(N*n_planes, M, 3)
    inv_planes = torch.linalg.inv(planes).unsqueeze(0).expand(N, -1, -1, -1).reshape(N*n_planes, 3, 3)
    projections = torch.bmm(coordinates, inv_planes)
    return projections[..., :2]

def sample_from_planes(plane_axes, plane_features, coordinates, mode='bilinear', padding_mode='zeros', box_warp=None):
    assert padding_mode == 'zeros'
    N, n_planes, C, H, W = plane_features.shape
    _, M, _ = coordinates.shape
    plane_features = plane_features.view(N*n_planes, C, H, W)
    
    # import pdb; pdb.set_trace()

    coordinates = (2/box_warp) * coordinates # TODO: add specific box bounds
    
    projected_coordinates = project_onto_planes(plane_axes, coordinates).unsqueeze(1)
    output_features = torch.nn.functional.grid_sample(plane_features, projected_coordinates.float(), mode=mode, padding_mode=padding_mode, align_corners=False).permute(0, 3, 2, 1).reshape(N, n_planes, M, C)
    return output_features

def sample_from_3dgrid(grid, coordinates):
    """
    Expects coordinates in shape (batch_size, num_points_per_batch, 3)
    Expects grid in shape (1, channels, H, W, D)
    (Also works if grid has batch size)
    Returns sampled features of shape (batch_size, num_points_per_batch, feature_channels)
    """
    batch_size, n_coords, n_dims = coordinates.shape
    sampled_features = torch.nn.functional.grid_sample(grid.expand(batch_size, -1, -1, -1, -1),
                                                       coordinates.reshape(batch_size, 1, 1, -1, n_dims),
                                                       mode='bilinear', padding_mode='zeros', align_corners=False)
    N, C, H, W, D = sampled_features.shape
    sampled_features = sampled_features.permute(0, 4, 3, 2, 1).reshape(N, H*W*D, C)
    return sampled_features

class ImportanceRenderer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ray_marcher = MipRayMarcher2()
        self.plane_axes = generate_planes()
        self.render_mesh = Render_Mesh()
        self.sigmoid_beta = nn.Parameter(0.1 * torch.ones(1))
    def forward(self, planes, decoder, ray_origins, ray_directions, rendering_options,additional_dict={}):
        self.plane_axes = self.plane_axes.to(ray_origins.device)
        batch_size = ray_origins.shape[0]
        # if rendering_options['ray_start'] == rendering_options['ray_end'] == 'auto':
        #     ray_start, ray_end = math_utils.get_ray_limits_box(ray_origins, ray_directions, box_side_length=rendering_options['box_warp'])
        #     is_ray_valid = ray_end > ray_start
        #     if torch.any(is_ray_valid).item():
        #         ray_start[~is_ray_valid] = ray_start[is_ray_valid].min()
        #         ray_end[~is_ray_valid] = ray_start[is_ray_valid].max()
        #     depths_coarse = self.sample_stratified(ray_origins, ray_start, ray_end, rendering_options['depth_resolution'], rendering_options['disparity_space_sampling'])
        # else:
        #     # Create stratified depth samples
        #     depths_coarse = self.sample_stratified(ray_origins, rendering_options['ray_start'], rendering_options['ray_end'], rendering_options['depth_resolution'], rendering_options['disparity_space_sampling'])

        # batch_size, num_rays, samples_per_ray, _ = depths_coarse.shape

        # # Coarse Pass
        # sample_coordinates = (ray_origins.unsqueeze(-2) + depths_coarse * ray_directions.unsqueeze(-2)).reshape(batch_size, -1, 3)
        # sample_directions = ray_directions.unsqueeze(-2).expand(-1, -1, samples_per_ray, -1).reshape(batch_size, -1, 3)
        # import pdb; pdb.set_trace()
        # start = time.time()
        if 'flame_para' in additional_dict:
            b = ray_origins.shape[0]
            # print('use transform coarse')
            # flame_model = additional_dict['flame_model'].to(ray_origins.device)
            all_flame_para =  additional_dict['flame_para']
            shape = all_flame_para[:,0:10]
            pose = all_flame_para[:,10:].view(b,24,3,3)
            # pose = all_flame_para[:,150:165]
            # with torch.
            vertices,canonical_vertices=  additional_dict['flame_model'](pose,shape)
            
        else:
            b = ray_origins.shape[0]
            shape = torch.zeros((b,10)).to(ray_origins.device)
            # pose = torch.zeros((b,24,3,3)).to(ray_origins.device)
            pose = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(b,24,1,1).to(ray_origins.device)
            # pose = torch.zeros((b,15)).to(ray_origins.device)
            vertices,canonical_vertices=   additional_dict['flame_model'](pose,shape)
            # vertices = canonical_vertices
        # print('SMPL time',time.time()-start)
        sample_directions = None
        
        hit_min_depth = []
        hit_max_depth = []
        hit_mask = []
        # start = time.time()
        for b_index in range(batch_size):
            hit_mask_this, hit_min_depth_this, hit_max_depth_this = clib.cloud_ray_intersect(rendering_options['distance_range'], vertices[b_index:b_index+1], ray_origins[b_index:b_index+1], ray_directions[b_index:b_index+1].contiguous())
            hit_min_depth.append(hit_min_depth_this)
            hit_max_depth.append(hit_max_depth_this)
            hit_mask.append(hit_mask_this)
        # print('hint  time',time.time()-start)

        hit_mask = torch.cat(hit_mask,dim=0).unsqueeze(-1)
        hit_min_depth = torch.cat(hit_min_depth,dim=0).unsqueeze(-1)
        hit_max_depth = torch.cat(hit_max_depth,dim=0).unsqueeze(-1)
        # import pdb; pdb.set_trace()

        hit_min_depth[hit_mask==0] = hit_min_depth.min()
        hit_max_depth[hit_mask==0] = hit_max_depth.max()
        depth_resolution = rendering_options['depth_resolution']
        N,M,_ = ray_origins.shape
        depths_coarse = torch.linspace(0, 1, depth_resolution, device=ray_origins.device).reshape(1, 1, depth_resolution, 1).repeat(N, M, 1, 1)
        # import pdb; pdb.set_trace()
        depths_coarse += torch.rand_like(depths_coarse)*(hit_max_depth-hit_min_depth)/depth_resolution
        depths_coarse = depths_coarse*(hit_max_depth-hit_min_depth)+hit_min_depth
        
        # depth_delta = (hit_max_depth-hit_min_depth)
        
        sample_coordinates = (ray_origins.unsqueeze(-2) + depths_coarse * ray_directions.unsqueeze(-2))
        batch_size, num_rays, samples_per_ray, _ = depths_coarse.shape

        # start = time.time()
        batch_size,point_numb,sample_depth,_ = sample_coordinates.shape

        our_need_rgb = torch.zeros((batch_size,point_numb,sample_depth,rendering_options['feature_dim'])).to(sample_coordinates.device)
        our_need_sigma = torch.zeros((batch_size,point_numb,sample_depth,1)).to(sample_coordinates.device)
        # sample_coordinates = sample_coordinates]
        hit_mask = hit_mask.view(batch_size,-1,1).expand(batch_size,point_numb,sample_depth)

        all_pred_sdf = []
        for b_index in range(batch_size):
            
            this_sample_coordinates = sample_coordinates[b_index:b_index+1]

            this_hit_mask = hit_mask[b_index:b_index+1]

            need_point = this_sample_coordinates[this_hit_mask==1,:].unsqueeze(0)
            
            tex_feature,template_sdf = additional_dict['tex_feature'](need_point,additional_dict['ws'][b_index:b_index+1],vertices[b_index:b_index+1],canonical_vertices[b_index:b_index+1])

            temp_out = self.run_model(planes[b_index:b_index+1], decoder,tex_feature,template_sdf, sample_coordinates, sample_directions, rendering_options) 
            all_pred_sdf.append(temp_out['pred_sdf'])
            our_need_rgb[b_index][this_hit_mask[0]==1,:] = temp_out['rgb']
            our_need_sigma[b_index][this_hit_mask[0]==1,:] = temp_out['sigma']

        
        # print('texture time',time.time()-start)
        # all_pred_sdf = torch.cat(all_pred_sdf,dim=1)
        # import pdb; pdb.set_trace()
        mesh_normal = self.render_mesh(vertices,additional_dict['extrinsic'] ,additional_dict['intrinsic'],rendering_options['image_resolution'] )

        colors_coarse = our_need_rgb
        densities_coarse = our_need_sigma
        colors_coarse = colors_coarse.reshape(batch_size, num_rays, samples_per_ray, colors_coarse.shape[-1])
        densities_coarse = densities_coarse.reshape(batch_size, num_rays, samples_per_ray, 1)

        N_importance = rendering_options['depth_resolution_importance']
        if N_importance > 0:
            _, _, weights = self.ray_marcher(colors_coarse, densities_coarse, depths_coarse, rendering_options)

            depths_fine = self.sample_importance(depths_coarse, weights, N_importance)

            sample_directions = ray_directions.unsqueeze(-2).expand(-1, -1, N_importance, -1).reshape(batch_size, -1, 3)
            sample_coordinates = (ray_origins.unsqueeze(-2) + depths_fine * ray_directions.unsqueeze(-2))


            batch_size, num_rays, samples_per_ray, _ = depths_coarse.shape

            batch_size,point_numb,sample_depth,_ = sample_coordinates.shape

            our_need_rgb = torch.zeros((batch_size,point_numb,sample_depth,rendering_options['feature_dim'])).to(sample_coordinates.device)
            our_need_sigma = torch.zeros((batch_size,point_numb,sample_depth,1)).to(sample_coordinates.device)
            # sample_coordinates = sample_coordinates]
            # hit_mask = hit_mask.view(batch_size,-1,1).expand(batch_size,point_numb,sample_depth)


            for b_index in range(batch_size):
                # this_sample_
                
                this_sample_coordinates = sample_coordinates[b_index:b_index+1]
                this_hit_mask = hit_mask[b_index:b_index+1]
                need_point = this_sample_coordinates[this_hit_mask==1,:].unsqueeze(0)
                # import pdb; pdb.set_trace()
                tex_feature,template_sdf = additional_dict['tex_feature'](need_point,additional_dict['ws'][b_index:b_index+1],vertices[b_index:b_index+1],canonical_vertices[b_index:b_index+1])

                temp_out = self.run_model(planes[b_index:b_index+1], decoder,tex_feature, template_sdf, sample_coordinates, sample_directions, rendering_options) 
                all_pred_sdf.append(temp_out['pred_sdf'])
                our_need_rgb[b_index][this_hit_mask[0]==1,:] = temp_out['rgb']
                our_need_sigma[b_index][this_hit_mask[0]==1,:] = temp_out['sigma']

            colors_fine = our_need_rgb
            densities_fine = our_need_sigma
            colors_fine = colors_fine.reshape(batch_size, num_rays, N_importance, colors_fine.shape[-1])
            densities_fine = densities_fine.reshape(batch_size, num_rays, N_importance, 1)

            all_depths, all_colors, all_densities = self.unify_samples(depths_coarse, colors_coarse, densities_coarse,
                                                                  depths_fine, colors_fine, densities_fine)

            # Aggregate
            rgb_final, depth_final, weights = self.ray_marcher(all_colors, all_densities, all_depths, rendering_options)
        else:
        
            rgb_final, depth_final, weights = self.ray_marcher(colors_coarse, densities_coarse, depths_coarse, rendering_options)
        # print('ray marching time',time.time()-start)
        # import pdb; pdb.set_trace()
        all_pred_sdf = torch.cat(all_pred_sdf,dim=1)
        out_dict = {}
        out_dict['rgb_final'] = rgb_final
        out_dict['depth_final'] = depth_final
        out_dict['weights'] = weights
        out_dict['mesh_normal'] = mesh_normal
        out_dict['all_pred_sdf'] = all_pred_sdf

        return out_dict
    def sdf_activation(self, input):
        self.sigmoid_beta.data.copy_(max(torch.zeros_like(self.sigmoid_beta.data) + 2e-3, self.sigmoid_beta.data))
        sigma = torch.sigmoid(input / self.sigmoid_beta) / self.sigmoid_beta

        return sigma
    
    def run_model(self, planes, decoder, uvd, template_sdf, sample_coordinates, sample_directions, options):
        

        sampled_features = sample_from_planes(self.plane_axes, planes, uvd, padding_mode='zeros', box_warp=options['box_warp'])
        sampled_features = sampled_features.mean(1)

        # merge_feature = torch.cat([sampled_features,tex_feature],dim=-1)
        
        out = decoder(sampled_features, sample_directions)
        # pred_sdf =
        out['pred_sdf'] =  out['sigma']
        if options.get('density_noise', 0) > 0:
            out['sigma'] += torch.randn_like(out['sigma']) * options['density_noise'] 

        # import pdb; pdb.set_trace()

        out['sigma'] +=  template_sdf
        # import pdb; pdb.set_trace()
        out['sigma'] = self.sdf_activation(-out['sigma'])

        return out

    def sort_samples(self, all_depths, all_colors, all_densities):
        _, indices = torch.sort(all_depths, dim=-2)
        all_depths = torch.gather(all_depths, -2, indices)
        all_colors = torch.gather(all_colors, -2, indices.expand(-1, -1, -1, all_colors.shape[-1]))
        all_densities = torch.gather(all_densities, -2, indices.expand(-1, -1, -1, 1))
        return all_depths, all_colors, all_densities

    def unify_samples(self, depths1, colors1, densities1, depths2, colors2, densities2):
        all_depths = torch.cat([depths1, depths2], dim = -2)
        all_colors = torch.cat([colors1, colors2], dim = -2)
        all_densities = torch.cat([densities1, densities2], dim = -2)

        _, indices = torch.sort(all_depths, dim=-2)
        all_depths = torch.gather(all_depths, -2, indices)
        all_colors = torch.gather(all_colors, -2, indices.expand(-1, -1, -1, all_colors.shape[-1]))
        all_densities = torch.gather(all_densities, -2, indices.expand(-1, -1, -1, 1))

        return all_depths, all_colors, all_densities

    def sample_stratified(self, ray_origins, ray_start, ray_end, depth_resolution, disparity_space_sampling=False):
        """
        Return depths of approximately uniformly spaced samples along rays.
        """
        N, M, _ = ray_origins.shape
        if disparity_space_sampling:
            depths_coarse = torch.linspace(0,
                                    1,
                                    depth_resolution,
                                    device=ray_origins.device).reshape(1, 1, depth_resolution, 1).repeat(N, M, 1, 1)
            depth_delta = 1/(depth_resolution - 1)
            depths_coarse += torch.rand_like(depths_coarse) * depth_delta
            depths_coarse = 1./(1./ray_start * (1. - depths_coarse) + 1./ray_end * depths_coarse)
        else:
            if type(ray_start) == torch.Tensor:
                depths_coarse = math_utils.linspace(ray_start, ray_end, depth_resolution).permute(1,2,0,3)
                depth_delta = (ray_end - ray_start) / (depth_resolution - 1)
                depths_coarse += torch.rand_like(depths_coarse) * depth_delta[..., None]
            else:
                depths_coarse = torch.linspace(ray_start, ray_end, depth_resolution, device=ray_origins.device).reshape(1, 1, depth_resolution, 1).repeat(N, M, 1, 1)
                depth_delta = (ray_end - ray_start)/(depth_resolution - 1)
                depths_coarse += torch.rand_like(depths_coarse) * depth_delta

        return depths_coarse

    def sample_importance(self, z_vals, weights, N_importance):
        """
        Return depths of importance sampled points along rays. See NeRF importance sampling for more.
        """
        with torch.no_grad():
            batch_size, num_rays, samples_per_ray, _ = z_vals.shape

            z_vals = z_vals.reshape(batch_size * num_rays, samples_per_ray)
            weights = weights.reshape(batch_size * num_rays, -1) # -1 to account for loss of 1 sample in MipRayMarcher

            # smooth weights
            weights = torch.nn.functional.max_pool1d(weights.unsqueeze(1).float(), 2, 1, padding=1)
            weights = torch.nn.functional.avg_pool1d(weights, 2, 1).squeeze()
            weights = weights + 0.01

            z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:])
            importance_z_vals = self.sample_pdf(z_vals_mid, weights[:, 1:-1],
                                             N_importance).detach().reshape(batch_size, num_rays, N_importance, 1)
        return importance_z_vals

    def sample_pdf(self, bins, weights, N_importance, det=False, eps=1e-5):
        """
        Sample @N_importance samples from @bins with distribution defined by @weights.
        Inputs:
            bins: (N_rays, N_samples_+1) where N_samples_ is "the number of coarse samples per ray - 2"
            weights: (N_rays, N_samples_)
            N_importance: the number of samples to draw from the distribution
            det: deterministic or not
            eps: a small number to prevent division by zero
        Outputs:
            samples: the sampled samples
        """
        N_rays, N_samples_ = weights.shape
        weights = weights + eps # prevent division by zero (don't do inplace op!)
        pdf = weights / torch.sum(weights, -1, keepdim=True) # (N_rays, N_samples_)
        cdf = torch.cumsum(pdf, -1) # (N_rays, N_samples), cumulative distribution function
        cdf = torch.cat([torch.zeros_like(cdf[: ,:1]), cdf], -1)  # (N_rays, N_samples_+1)
                                                                   # padded to 0~1 inclusive

        if det:
            u = torch.linspace(0, 1, N_importance, device=bins.device)
            u = u.expand(N_rays, N_importance)
        else:
            u = torch.rand(N_rays, N_importance, device=bins.device)
        u = u.contiguous()

        inds = torch.searchsorted(cdf, u, right=True)
        below = torch.clamp_min(inds-1, 0)
        above = torch.clamp_max(inds, N_samples_)

        inds_sampled = torch.stack([below, above], -1).view(N_rays, 2*N_importance)
        cdf_g = torch.gather(cdf, 1, inds_sampled).view(N_rays, N_importance, 2)
        bins_g = torch.gather(bins, 1, inds_sampled).view(N_rays, N_importance, 2)

        denom = cdf_g[...,1]-cdf_g[...,0]
        denom[denom<eps] = 1 # denom equals 0 means a bin has weight 0, in which case it will not be sampled
                             # anyway, therefore any value for it is fine (set to 1 here)

        samples = bins_g[...,0] + (u-cdf_g[...,0])/denom * (bins_g[...,1]-bins_g[...,0])
        return samples
""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.renderer import (
    BlendParams, blending, look_at_view_transform, FoVOrthographicCameras,
    PointLights, RasterizationSettings, PointsRasterizationSettings,
    PointsRenderer, AlphaCompositor, PointsRasterizer, MeshRenderer,
    MeshRasterizer, SoftPhongShader, SoftSilhouetteShader, TexturesVertex,TexturesAtlas)
from training.networks import *

from training.networks_stylegan2 import Generator as StyleGAN2Backbone,MappingNetwork
import sys
sys.path.append('..')
import clib
import cv2
from training.networks_stylegan2 import SynthesisLayer,FullyConnectedLayer,SynthesisBlock

from training.superresolution import SynthesisBlockNoUp
import numpy as np
# from training.networks_stylegan2 import FullyConnectedLayer
# from  pytorch3d.loss import point_mesh_face_distance
import time
from knn_cuda import KNN

from pytorch3d.ops.knn import knn_gather, knn_points

class UNet(nn.Module):
    def __init__(self, n_channels,**block_kwargs):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        use_fp16 = False
        # self.inc = DoubleConv(n_channels, 64)
        self.down = nn.AvgPool2d(2)
        self.inc = SynthesisBlockNoUp(n_channels, 32, w_dim=512, resolution=256,
                img_channels=0, is_last=False, use_fp16=use_fp16, conv_clamp=(256 if use_fp16 else None),architecture='resnet',**block_kwargs)
        
        self.down1 = SynthesisBlockNoUp(32, 64, w_dim=512, resolution=128,
                img_channels=0, is_last=False, use_fp16=use_fp16, conv_clamp=(256 if use_fp16 else None),architecture='resnet',**block_kwargs)
        
        self.down2 = SynthesisBlockNoUp(64, 128, w_dim=512, resolution=64,
                img_channels=0, is_last=False, use_fp16=use_fp16, conv_clamp=(256 if use_fp16 else None),architecture='resnet',**block_kwargs)
        
        self.down3 = SynthesisBlockNoUp(128, 256, w_dim=512, resolution=32,
                img_channels=0, is_last=False, use_fp16=use_fp16, conv_clamp=(256 if use_fp16 else None),architecture='resnet',**block_kwargs)

        self.up1 =  SynthesisBlock(256, 128, w_dim=512, resolution=64,
                img_channels=0, is_last=False, use_fp16=use_fp16, conv_clamp=(256 if use_fp16 else None),architecture='resnet',**block_kwargs)
        self.up2 =  SynthesisBlock(128+128, 128, w_dim=512, resolution=128,
                img_channels=0, is_last=False, use_fp16=use_fp16, conv_clamp=(256 if use_fp16 else None),architecture='resnet',**block_kwargs)
        self.up3 =  SynthesisBlock(128+64, 64, w_dim=512, resolution=256,
                img_channels=0, is_last=False, use_fp16=use_fp16, conv_clamp=(256 if use_fp16 else None),architecture='resnet',**block_kwargs)

        self.out = SynthesisBlockNoUp(64+32, 64, w_dim=512, resolution=256,
                img_channels=0, is_last=False, use_fp16=use_fp16, conv_clamp=(256 if use_fp16 else None),architecture='resnet',**block_kwargs)
        


    def forward(self, input_texture_map,ws,**block_kwargs):
        # ws = 
        # ws = ws.to(x.device)
        # w_iter = iter(ws.unbind(dim=1))
        # import pdb; pdb.set_trace()
        if self.training:
            noise_mode = 'random'
        else:
            noise_mode = 'const'
            # print(noise_mode)
        #print('noise_model',noise_mode)
        x1,_ = self.inc(input_texture_map,img=None,ws = ws[:,0:2],noise_mode=noise_mode)

        x2 = self.down(x1)
        x2,_  = self.down1(x2,img=None,ws = ws[:,2:4],noise_mode=noise_mode)
        # 128 128 128

        x3 = self.down(x2)
        x3,_  = self.down2(x3,img=None,ws = ws[:,4:6],noise_mode=noise_mode)
        # 256 64 64
        x4 = self.down(x3)
        x4,_  = self.down3(x4,img=None,ws = ws[:,6:8],noise_mode=noise_mode)
        # 512 32 32
        # x5 = self.down(x4)
        # x5,_  = self.down4(x5,img=None,ws = ws[:,6:8])

        
        x,_  = self.up1(x4,img=None,ws = ws[:,8:10],noise_mode=noise_mode)
        # 256,64,64
        x = torch.cat([x,x3],dim=1)
        x,_  = self.up2(x,img=None,ws = ws[:,10:12],noise_mode=noise_mode)
        x = torch.cat([x,x2],dim=1)

        x,_ = self.up3(x,img=None,ws = ws[:,12:14],noise_mode=noise_mode)
        # import pdb
        # pdb.set_trace()
        x = torch.cat([x,x1],dim=1)

        x,_ = self.out(x,img=None,ws = ws[:,14:16],noise_mode=noise_mode)


        return x

def positional_encoding(p, size=7, use_pos=True):
    
    p_transformed = torch.cat([torch.cat(
            [torch.sin((2 ** i) * np.pi * p),
            torch.cos((2 ** i) * np.pi * p)],
            dim=-1) for i in range(size)], dim=-1)
    if use_pos:
        p_transformed = torch.cat([p_transformed, p], -1)
    
    return p_transformed


def mesh_to_normal_map(verts,faces):
    pass
    #print()

class style_Linear(torch.nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_channels, out_channels,w_dim,
        activation='lrelu', 
        resample_filter=[1,3,3,1],
        magnitude_ema_beta = -1, ):

        super(style_Linear, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.w_dim = w_dim
        self.padding = 0
        self.activation = activation
        memory_format = torch.contiguous_format

        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        self.weight = torch.nn.Parameter(
               torch.randn([out_channels, in_channels, 1, 1]).to(memory_format=memory_format))
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))
        self.act_gain = bias_act.activation_funcs[activation].def_gain
        # self.reset_parameters()
        self.magnitude_ema_beta = magnitude_ema_beta
        if magnitude_ema_beta > 0:
            self.register_buffer('w_avg', torch.ones([]))
        self.conv_clamp = None

    def forward(self, x,w,gain=1,up=1,fused_modconv=None):
        assert x.ndim==3
        styles = self.affine(w) 
        act = self.activation
        flip_weight = True #
        # input feature shape b,p,3
        b,p,c = x.shape
        
        x = x.permute(0,2,1)
        x =x.view(b,c,p,1)
        

        if fused_modconv is None:
            with misc.suppress_tracer_warnings(): # this value will be treated as a constant
                fused_modconv = not self.training
        if x.size(0) > styles.size(0):
            styles = repeat(styles, 'b c -> (b s) c', s=x.size(0) // styles.size(0))

        x = modulated_conv2d(x=x, weight=self.weight, styles=styles, noise=None, up=up,
                padding=self.padding, resample_filter=self.resample_filter, 
                flip_weight=flip_weight, fused_modconv=fused_modconv)
        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None

        x = bias_act.bias_act(x, self.bias.to(x.dtype), act=act, gain=act_gain, clamp=act_clamp)

        x = x.permute(0,2,1,3)

        x = x.view(b,p,self.out_channels)
        
        return x


    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_channels) + ' -> ' \
               + str(self.out_channels) + ')'



class style_residual(torch.nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.hidden_dim=64
        # self.merge_feature = torch.nn.Sequential(
        #     FullyConnectedLayer(n_features, self.hidden_dim, lr_multiplier=options['decoder_lr_mul']),
        #     torch.nn.Softplus(),
        #     FullyConnectedLayer(self.hidden_dim, 1 + options['decoder_output_dim'], lr_multiplier=options['decoder_lr_mul'])
        # )

        
        self.net1 =    style_Linear(n_features, self.hidden_dim,512)
      
        self.net2 =    style_Linear(self.hidden_dim, self.hidden_dim,512)
        
        self.net3 =    style_Linear(self.hidden_dim, self.hidden_dim,512)

        self.act = torch.nn.Tanh()

        self.final = nn.Linear(self.hidden_dim,3)
        self.final.weight.data = nn.parameter.Parameter(torch.zeros_like(self.final.weight.data))
        self.final.bias.data = nn.parameter.Parameter(torch.zeros_like(self.final.bias.data))
    def forward(self, x,w):
        # import pdb; pdb.set_trace()
        x = self.net1(x,w)
        x = self.act(x)
        x = self.net2(x,w)
        x = self.act(x)
        x = self.net3(x,w)
        x = self.act(x)
        x = self.final(x)
        # print(x)
        return x


class Texture_Feature(nn.Module):
    def __init__(self,input_features=3, hidden_feature=64, bilinear=False,template_path = '../data_for_run/cut_alongsmplseams.obj',position_f = 5,mapping_kwargs=None,synthesis_kwargs=None,z_dim=None,c_dim=None,w_dim=None):
        super(Texture_Feature, self).__init__()
        # self.Unet = UNet(input_features)
        
        # self.Unet = StyleGAN2Backbone(z_dim, c_dim, w_dim, img_resolution=256, img_channels=64, mapping_kwargs=mapping_kwargs, **synthesis_kwargs)

        # obj_filename = 'flame/head_template_mesh.obj'
        # self.Residual = Tex_Dis_Residual(hidden_feature+position_f*2+1)
        verts, faces, aux = load_obj(template_path)
        uvcoords = aux.verts_uvs[None,...]*2-1 # (N, V, 2)
        # uvcoords = torch.cat([uvcoords, torch.zeros((uvcoords.shape[0],uvcoords.shape[1],1)).to(uvcoords.device)], dim=2)
        uvcoords = torch.cat([uvcoords, torch.zeros((uvcoords.shape[0],uvcoords.shape[1],1)).to(uvcoords.device)], dim=2)

        self.register_buffer('uvcoords', uvcoords)

        uvfaces = faces.textures_idx[None,...]  # (N, F, 3)
        self.register_buffer('uvfaces', uvfaces)

        faces = faces.verts_idx[None,...]
        self.register_buffer('faces', faces)

        self.uvcoords.requires_grad=False
        self.uvfaces.requires_grad=False
        self.faces.requires_grad=False
        # self.decoder_merge = Tex_Dis_to_Feature(64+5*2+1,32)
        self.set_render_para()
        
        # self.merge_feature = Merge_Decoder(64+(7*2+1)*1)
        self.resiudal =  style_residual((5*2+1)*6)

        vertics_uv_position = torch.zeros_like(verts)

        vertics_uv_position[faces[0]] = uvcoords[0,uvfaces[0]]
        vertics_uv_position = vertics_uv_position[:,0:2]
        # import pdb; pdb.set_trace()
        self.register_buffer('vertics_uv_position', vertics_uv_position.unsqueeze(0))
        # normal_vertices = normal_vertices.cpu().detach().numpy()

        # mag, ang = cv2.cartToPolar(normal_vertices[...,0], normal_vertices[...,1])

        # hsv = np.zeros_like(normal_vertices)
        # self.merge_
        # self.merge_mesh=self.net = torch.nn.Sequential(
        #     FullyConnectedLayer(64+5*2+1, self.hidden_dim),
        #     torch.nn.Softplus(),
        #     FullyConnectedLayer(self.hidden_dim, self.hidden_dim),

        
        # self.flame_model = flame.FLAME.FLAME(n_shape=100,n_exp = 50)
        self.cos = nn.CosineSimilarity(dim=2, eps=1e-6)
    def set_render_para(self):

        R, T = look_at_view_transform(20, 0, 0)

        cameras = FoVOrthographicCameras(
        R=R, T=T, 
        znear=1.0,
        zfar=-1.0,
        max_y=1.0,
        min_y=-1.0,
        max_x=1.0,
        min_x=-1.0,)

        raster_settings_mesh = RasterizationSettings(
                    image_size=256,
                    blur_radius=0,
                    faces_per_pixel=10,
                )

        meshRas = MeshRasterizer(cameras=cameras, raster_settings=raster_settings_mesh)

        blendparam = BlendParams(1e-4, 1e-4, (-1.0, -1.0, -1.0))

        lights = PointLights( ambient_color=((0.8, 0.8, 0.8), ),
                                diffuse_color=((0.2, 0.2, 0.2), ),
                                specular_color=((0.0, 0.0, 0.0), ),
                                location=[[0.0, 0.0, 1.0]])
        self.renderer = MeshRenderer(rasterizer=meshRas,
                                        shader=SoftPhongShader(
                                            
                                            cameras=cameras,
                                            lights=lights,
                                            blend_params=blendparam))
        
    # def render_mask(self):
    #     pass
    
    
    def forward(self,query_points,w,vertices,canonical_vertices):
        batch_size = w.shape[0]
        b,pb,_ = query_points.shape
        # return  torch.zeros((b,pb,32)).to(query_points.device)
       
        assert w.shape[0]==vertices.shape[0]
        with torch.no_grad():
            mesh_base = Meshes(verts=canonical_vertices,faces=self.faces.expand(batch_size,-1,-1))

            mesh = Meshes(verts=self.uvcoords.expand(batch_size,-1,-1),faces=self.uvfaces.expand(batch_size,-1,-1))

            vertices_norm = torch.stack(mesh_base.verts_normals_list())

            vertices_new_normal = torch.zeros(self.uvcoords.shape).repeat(batch_size,1,1).to(vertices_norm.device)

            del mesh
            del mesh_base
            # import pdb; pdb.set_trace()
        
        mesh_this = Meshes(verts=vertices,faces=self.faces.expand(batch_size,-1,-1))
        vertices_this_norm = torch.stack(mesh_this.verts_normals_list())
        # ew = w[:,:,None,None].expand(-1,-1,texture_norm.shape[2],texture_norm.shape[3])
        vertice_deform = canonical_vertices-vertices
        # now p + deform = canonical deform
        #print(texture_norm.mean(),'texture_norm.mean()')
        #print(w.mean(),'w.mean')
        if self.training:
            noise_mode='random'
        else:
            noise_mode='const'

        # feature_map = self.Unet.synthesis(w[:,0:14],noise_mode=noise_mode)
        # del texture_norm
        # import pdb; pdb.set_trace()
        # feature_dim = feature_map.shape[1]
        # vec_feature_dim = feature_vector_map.shape[1]

        all_real_dis = []
        all_uv_position = []
        all_deform = []
        # start = time.time()
        
        # import pdb; pdb.set_trace()
        k_numb = 3
        nn = knn_points(query_points, vertices, K=k_numb)
      
        all_distance = (nn.dists)**0.5
        #print(all_distance.mean(),'all distance mean')
        all_idx = nn.idx
        # tempdis = torch.cat(all_distance,dim=1)
        # all_idx = torch.cat(all_idx,dim=1)
        
        all_uv_position_knn = []
        all_nearest_norm = []
        all_nearest_point = []
        all_nearest_deform = []
        for batch in range(batch_size):
            # 
            vertics_uv_position = self.vertics_uv_position[0]
            this_uv = vertics_uv_position[all_idx[batch]]
            this_normal = vertices_this_norm[batch][all_idx[batch]]

            all_uv_position_knn.append(this_uv)
            all_nearest_norm.append(this_normal)
            all_nearest_point.append(vertices[batch][all_idx[batch]])
            all_nearest_deform.append(vertice_deform[batch][all_idx[batch]])
            # 
        all_uv_position_knn = torch.stack(all_uv_position_knn,dim=0)
        all_nearest_norm = torch.stack(all_nearest_norm,dim=0)
        all_nearest_point = torch.stack(all_nearest_point,dim=0)
        all_nearest_deform = torch.stack(all_nearest_deform,dim=0)
        

        k_weight = F.normalize(1./(all_distance+1e-10),p=1,dim = 2)
        # import pdb; pdb.set_trace()
        # 

        all_nearest_norm = (k_weight.unsqueeze(-1)*all_nearest_norm).sum(dim=2)
        all_nearest_point = (k_weight.unsqueeze(-1)*all_nearest_point).sum(dim=2)
        all_nearest_deform = (k_weight.unsqueeze(-1)*all_nearest_deform).sum(dim=2)
        all_distance = (k_weight*all_distance).sum(dim=2).unsqueeze(-1)
        all_uv_position_knn = (k_weight.unsqueeze(-1)*all_uv_position_knn).sum(dim=2)
        
        merge_point = torch.cat([query_points,all_nearest_point],dim=2)
        position_merge = positional_encoding(merge_point,size=5)
        # residual = self.resiudal(position_merge,w[:,0])

        all_direction = query_points-all_nearest_point
        # import pdb; pdb.set_trace()
        canonical_query_point = query_points+all_nearest_deform
        cos_sim_dirction = self.cos(all_direction,all_nearest_norm).unsqueeze(-1)
        # import pdb; pdb.set_trace()
        # all_uv_position = all_uv_position_knn.view(batch_size,pb*k_numb,1,2)
        
        cos_sim_dirction[cos_sim_dirction>0]=1
        cos_sim_dirction[cos_sim_dirction<0]=-1

        template_sdf = all_distance*cos_sim_dirction
        template_sdf = torch.clamp(template_sdf, min=-0.24, max=0.24)

        # embed_distance = template_sdf* 3.5

        
        uvd = torch.cat([all_uv_position_knn,template_sdf],dim=2)
        
        
        # uvd = uvd + residual
        
        # import pdb; pdb.set_trace()

        return uvd, template_sdf


        #print(all_uv_position.mean(),'all_uv_position.mean()')
        #print(feature_map.mean(),'feature mean')
        # texture_feature = F.grid_sample(feature_map, all_uv_position, mode='bilinear', align_corners=False) 
        # # import pdb; pdb.set_trace()
        # #print(texture_feature.mean(),'texture mean before')
        # texture_feature = texture_feature.view(batch_size,feature_dim,pb,k_numb)
        # texture_feature =  texture_feature.permute(0, 2, 3, 1)
        # texture_feature =  (k_weight.unsqueeze(-1)*texture_feature).sum(dim=2)
        
         
        # # all_distance = torch.cat(all_real_dis,dim=0)
        # all_distance_ebd = positional_encoding(all_distance)
        # xyz_embedding = positional_encoding(canonical_query_point)
        # texture_feature = torch.cat([texture_feature,all_distance_ebd],dim=2)
        # # import pdb; pdb.set_trace()
        # texture_feature = self.merge_feature(texture_feature)
        # #print(texture_feature)
        # print(texture_feature.mean(),'texture mean')
        # texture_feature[(all_distance).expand(batch_size,pb,feature_dim)>0.15] = 0
        
        # import trimesh
        # import pdb; pdb.set_trace()
        
        # all_uv_position_debug = []
        # all_real_dis = []
        # for b in range(batch_size):
        #     this_points = query_points[b].contiguous()
        #     this_mesh_p = vertices[b]
        #     this_deform = vertice_deform[b]

        #     triangles = F.embedding(self.faces[0], this_mesh_p)
        #     l_idx = torch.tensor([0,]).long().to(triangles.device)
        #     min_dis, min_face_idx, w0, w1, w2 = clib._ext.point_face_dist_forward(
        #     this_points, l_idx, triangles, l_idx, this_points.size(0))
        #     bary_coords = torch.stack([w0, w1, w2], 1)
            
        #     uv_postion = F.embedding(self.uvfaces[0], self.uvcoords[0])
        #     base_deform = F.embedding(self.faces[0], this_deform)

        #     sampled_uvs = (uv_postion[min_face_idx] * bary_coords.unsqueeze(-1)).sum(1)[:,0:2]
        #     sampled_deform = (base_deform[min_face_idx] * bary_coords.unsqueeze(-1)).sum(1)

        #     sampled_uvs = sampled_uvs[None, :, None, :].to(query_points.device)

        #     all_uv_position_debug.append(sampled_uvs)
            
        #     min_dis = min_dis.view(1,-1,1)
        #     # all_real_dis = positional_encoding(min_dis)
        #     all_real_dis.append(min_dis)
            
        # all_real_dis = torch.cat(all_real_dis,dim=0)
        # all_uv_position_debug = torch.cat(all_uv_position_debug,dim=0)
        # # 
        # all_real_dis[all_real_dis>0.1]=0
        # all_distance[all_distance>0.1]=0
        # diff = all_real_dis-all_distance

        
        # import trimesh
        # points = query_points
        # points = points.detach().view(-1,3).cpu().numpy()
    
        # points=trimesh.PointCloud(points)
        # points.export('debug_all.ply')
        
        # points = vertices.detach().view(-1,3).cpu().numpy()
        # points=trimesh.PointCloud(points)
        # points.export('smpl.ply')
        
        


if __name__=='__main__':
    pass
    import json
    # Texture_Feature
    # import flame.FLAME
    Texture_Generator = Texture_Feature(device='cuda')
    # vertices = torch.rand(6890,)
    the_model = flame.FLAME.FLAME(n_shape=100,n_exp = 50).cuda()
   
    the_data = json.load(open('/mnt/data/gqzdata/dataset/FFHQ_512_eg3d_flame/00000/img00000033.json'))
    shape1 = torch.tensor(the_data['shape']).view(-1,100).cuda()
    exp1 = torch.tensor(the_data['exp']).view(-1,50).cuda()
    pose1 = torch.tensor(the_data['pose']).view(-1,15).cuda()

    the_data = json.load(open('/mnt/data/gqzdata/dataset/FFHQ_512_eg3d_flame/00000/img00000985.json'))
    shape2 = torch.tensor(the_data['shape']).view(-1,100).cuda()
    exp2 = torch.tensor(the_data['exp']).view(-1,50).cuda()
    pose2 = torch.tensor(the_data['pose']).view(-1,15).cuda()
    
    shape = torch.cat([shape1,shape2])
    exp = torch.cat([exp1,exp2])
    pose = torch.cat([pose1,pose2])

    vertices, canonical_vertices =  the_model(exp,shape,pose)
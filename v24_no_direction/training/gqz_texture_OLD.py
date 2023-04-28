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



import sys
sys.path.append('..')
import clib

from training.networks_stylegan2 import SynthesisLayer,FullyConnectedLayer,SynthesisBlock

from training.superresolution import SynthesisBlockNoUp
import numpy as np
# from training.networks_stylegan2 import FullyConnectedLayer



class UNet(nn.Module):
    def __init__(self, n_channels,**block_kwargs):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        use_fp16 = False
        # self.inc = DoubleConv(n_channels, 64)
        self.down = nn.AvgPool2d(2)
        self.inc = SynthesisBlockNoUp(n_channels, 64, w_dim=512, resolution=256,
                img_channels=0, is_last=False, use_fp16=use_fp16, conv_clamp=(256 if use_fp16 else None),architecture='resnet',**block_kwargs)
        
        self.down1 = SynthesisBlockNoUp(64, 128, w_dim=512, resolution=128,
                img_channels=0, is_last=False, use_fp16=use_fp16, conv_clamp=(256 if use_fp16 else None),architecture='resnet',**block_kwargs)
        
        self.down2 = SynthesisBlockNoUp(128, 256, w_dim=512, resolution=64,
                img_channels=0, is_last=False, use_fp16=use_fp16, conv_clamp=(256 if use_fp16 else None),architecture='resnet',**block_kwargs)
        
        self.down3 = SynthesisBlockNoUp(256, 512, w_dim=512, resolution=32,
                img_channels=0, is_last=False, use_fp16=use_fp16, conv_clamp=(256 if use_fp16 else None),architecture='resnet',**block_kwargs)
        # self.inc2 = DoubleConv(512, 64)
        # self.down4 = SynthesisBlockNoUp(256, 512, w_dim=512, resolution=32,
        #         img_channels=0, is_last=False, use_fp16=use_fp16, conv_clamp=(256 if use_fp16 else None),architecture='resnet',**block_kwargs)
        # # self.inc3 = DoubleConv(128, 64)
        self.up1 =  SynthesisBlock(512, 256, w_dim=512, resolution=64,
                img_channels=0, is_last=False, use_fp16=use_fp16, conv_clamp=(256 if use_fp16 else None),architecture='resnet',**block_kwargs)
        self.up2 =  SynthesisBlock(256+256, 256, w_dim=512, resolution=128,
                img_channels=0, is_last=False, use_fp16=use_fp16, conv_clamp=(256 if use_fp16 else None),architecture='resnet',**block_kwargs)
        self.up3 =  SynthesisBlock(256+128, 128, w_dim=512, resolution=256,
                img_channels=0, is_last=False, use_fp16=use_fp16, conv_clamp=(256 if use_fp16 else None),architecture='resnet',**block_kwargs)

        self.out = SynthesisBlockNoUp(128+64, 64, w_dim=512, resolution=256,
                img_channels=0, is_last=False, use_fp16=use_fp16, conv_clamp=(256 if use_fp16 else None),architecture='resnet',**block_kwargs)
        


    def forward(self, input_texture_map,ws,**block_kwargs):
        # ws = 
        # ws = ws.to(x.device)
        # w_iter = iter(ws.unbind(dim=1))
        # import pdb; pdb.set_trace()
        x1,_ = self.inc(input_texture_map,img=None,ws = ws[:,0:2])

        x2 = self.down(x1)
        x2,_  = self.down1(x2,img=None,ws = ws[:,2:4])
        # 128 128 128

        x3 = self.down(x2)
        x3,_  = self.down2(x3,img=None,ws = ws[:,4:6])
        # 256 64 64
        x4 = self.down(x3)
        x4,_  = self.down3(x4,img=None,ws = ws[:,6:8])
        # 512 32 32
        # x5 = self.down(x4)
        # x5,_  = self.down4(x5,img=None,ws = ws[:,6:8])

        
        x,_  = self.up1(x4,img=None,ws = ws[:,8:10])
        # 256,64,64
        x = torch.cat([x,x3],dim=1)
        x,_  = self.up2(x,img=None,ws = ws[:,10:12])
        x = torch.cat([x,x2],dim=1)

        x,_ = self.up3(x,img=None,ws = ws[:,12:14])
        # import pdb
        # pdb.set_trace()
        x = torch.cat([x,x1],dim=1)

        x,_ = self.out(x,img=None,ws = ws[:,14:16])


        return x

def positional_encoding(p, size=5, use_pos=True):
    
    p_transformed = torch.cat([torch.cat(
            [torch.sin((2 ** i) * np.pi * p),
            torch.cos((2 ** i) * np.pi * p)],
            dim=-1) for i in range(size)], dim=-1)
    if use_pos:
        p_transformed = torch.cat([p_transformed, p], -1)
    
    return p_transformed


def mesh_to_normal_map(verts,faces):
    pass
    print()


class Merge_Decoder(torch.nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.hidden_dim = 64
        # self.merge_feature = torch.nn.Sequential(
        #     FullyConnectedLayer(n_features, self.hidden_dim, lr_multiplier=options['decoder_lr_mul']),
        #     torch.nn.Softplus(),
        #     FullyConnectedLayer(self.hidden_dim, 1 + options['decoder_output_dim'], lr_multiplier=options['decoder_lr_mul'])
        # )

        self.net = torch.nn.Sequential(
            FullyConnectedLayer(n_features, self.hidden_dim),
            torch.nn.Softplus(),
            FullyConnectedLayer(self.hidden_dim, self.hidden_dim),
            torch.nn.Softplus(),
            FullyConnectedLayer(self.hidden_dim, 64),
            torch.nn.Softplus(),
        )
        # self.net[-1].weight.data = nn.parameter.Parameter(torch.zeros_like(self.net[-1].weight.data))
        # self.net[-1].bias.data = nn.parameter.Parameter(torch.zeros_like(self.net[-1].bias.data ))
    def forward(self, texture_feature):
        # Aggregate features
        # sampled_features = sampled_features.mean(1)
        x = texture_feature.contiguous()

        N, M, C = x.shape
        x = x.view(N*M, C)

        x = self.net(x)
        x = x.view(N, M, -1)

        return x


class Texture_Feature(nn.Module):
    def __init__(self,input_features=3, hidden_feature=64, bilinear=False,template_path = '../data_for_run/cut_alongsmplseams.obj',position_f = 5):
        super(Texture_Feature, self).__init__()
        self.Unet = UNet(input_features)
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
        
        self.merge_feature = Merge_Decoder(64+5*2+1)
        # self.merge_
        # self.merge_mesh=self.net = torch.nn.Sequential(
        #     FullyConnectedLayer(64+5*2+1, self.hidden_dim),
        #     torch.nn.Softplus(),
        #     FullyConnectedLayer(self.hidden_dim, self.hidden_dim),
        # )
        
        # self.flame_model = flame.FLAME.FLAME(n_shape=100,n_exp = 50)

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

            for i in range(batch_size):
                this_face_norm = self.faces[0].view(-1)
                this_face_vet = self.uvfaces[0].view(-1)
                this_v_norm = vertices_norm[i]
                this_v_vet = vertices_new_normal[i]
                this_v_vet[this_face_vet] = this_v_norm[this_face_norm]
            # vertices_new_normal = vertices_new_normal/2+0.5
            mesh.textures = TexturesVertex(verts_features=vertices_new_normal)
            self.renderer.to(vertices.device)

            texture_norm  = self.renderer(mesh)[:,:,:,0:3]

            texture_norm = texture_norm.permute(0,3,1,2)
            # import PIL.Image
            # img = (texture_norm * 255).permute(0,2,3,1).clamp(0, 255).to(torch.uint8).cpu().detach().numpy()[0]
            # PIL.Image.fromarray(img, 'RGB').save('checkpoints/debug.png')
            del mesh
            del mesh_base
            # import pdb; pdb.set_trace()

        # ew = w[:,:,None,None].expand(-1,-1,texture_norm.shape[2],texture_norm.shape[3])
        vertice_deform = canonical_vertices-vertices
        # now p + deform = canonical deform

        feature_map = self.Unet(texture_norm,w)
        del texture_norm

        feature_dim = feature_map.shape[1]
        # vec_feature_dim = feature_vector_map.shape[1]

        all_real_dis = []
        all_uv_position = []
        all_deform = []
        for b in range(batch_size):
            this_points = query_points[b].contiguous()
            this_mesh_p = vertices[b]
            this_deform = vertice_deform[b]

            triangles = F.embedding(self.faces[0], this_mesh_p)
            l_idx = torch.tensor([0,]).long().to(triangles.device)
            min_dis, min_face_idx, w0, w1, w2 = clib._ext.point_face_dist_forward(
            this_points, l_idx, triangles, l_idx, this_points.size(0))
            bary_coords = torch.stack([w0, w1, w2], 1)
            
            uv_postion = F.embedding(self.uvfaces[0], self.uvcoords[0])
            base_deform = F.embedding(self.faces[0], this_deform)

            sampled_uvs = (uv_postion[min_face_idx] * bary_coords.unsqueeze(-1)).sum(1)[:,0:2]
            sampled_deform = (base_deform[min_face_idx] * bary_coords.unsqueeze(-1)).sum(1)

            sampled_uvs = sampled_uvs[None, :, None, :].to(query_points.device)

            all_uv_position.append(sampled_uvs)
            
            min_dis = min_dis.view(1,-1,1)
            # all_real_dis = positional_encoding(min_dis)
            all_real_dis.append(min_dis)
            all_deform.append(sampled_deform)


        all_uv_position = torch.cat(all_uv_position,dim=0)
        # all_deform = torch.stack(all_deform,dim=0)

        texture_feature = F.grid_sample(feature_map, all_uv_position, mode='bilinear', align_corners=False) 
        texture_feature = texture_feature.permute(0, 2, 3, 1).reshape(batch_size,-1, feature_dim)
         
        all_distance = torch.cat(all_real_dis,dim=0)
        all_distance_ebd = positional_encoding(all_distance)
        texture_feature = torch.cat([texture_feature,all_distance_ebd],dim=2)
        texture_feature = self.merge_feature(texture_feature)
        texture_feature[(all_distance).expand(batch_size,pb,feature_dim)>0.2] = 0
        # import pdb; pdb.set_trace()
        # import trimesh
        # points = query_points.detach().cpu().view(-1,3)
        # points=trimesh.PointCloud(points)
        # points.export('debug_all.ply')


        return texture_feature


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
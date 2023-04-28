""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.io import load_obj,save_obj
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.renderer import (
    BlendParams, PerspectiveCameras, look_at_view_transform, FoVOrthographicCameras,
    PointLights, RasterizationSettings, PointsRasterizationSettings,
    PointsRenderer, AlphaCompositor, PointsRasterizer, MeshRenderer,
    MeshRasterizer, SoftPhongShader, SoftSilhouetteShader, TexturesVertex,TexturesAtlas)

import numpy as np
import cv2


class Render_Mesh(nn.Module):
    def __init__(self,template_path = '../data_for_run/cut_alongsmplseams.obj'):
        super(Render_Mesh, self).__init__()
         
        verts, faces, aux = load_obj(template_path)
        # _,just_face_faces,_ = load_obj(just_face_path)
        uvfaces = faces.textures_idx[None,...] 
        faces = faces.verts_idx[None,...]
        # just_face_faces = just_face_faces.verts_idx[None,...]
        # import test_video_flame
        # self.flame_model = my_smpl.SMPLModel()
        self.register_buffer('faces', faces)
        # self.register_buffer('just_face_faces', just_face_faces)
        # se    

         # (N, F, 3)
        self.register_buffer('uvfaces', uvfaces)

        uvcoords = aux.verts_uvs[None,...]*2-1 # (N, V, 2)
        # print(uvcoords.max())
        # print(uvcoords.min())
        uvcoords = torch.cat([uvcoords, torch.zeros((uvcoords.shape[0],uvcoords.shape[1],1)).to(uvcoords.device)], dim=2)

        normal_vertices = torch.zeros_like(verts)

        normal_vertices[faces[0]] = uvcoords[0,uvfaces[0]]

        normal_vertices = normal_vertices.cpu().detach().numpy()

        mag, ang = cv2.cartToPolar(normal_vertices[...,0], normal_vertices[...,1])

        hsv = np.zeros_like(normal_vertices)

        hsv[...,1] = 255
        hsv[...,0] = (ang*180/np.pi/2)[...,0]
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)[...,0]

        hsv = hsv[np.newaxis,:,:].astype(np.uint8)
        bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)
        
        self.vertices_color = (torch.from_numpy(bgr)/255)*2-1


    
    def forward(self,vertices,cam2world_matrix,intrinsics,resolution_base):
        resolution = (resolution_base,resolution_base//2)
        # print('this is run start ',vertices.device.index)
        # print(cam2world_matrix)
        # print(intrinsics)
        # print(vertices.max())
        # print(vertices.min())
        # print('this run over')
        cam2world_matrix = cam2world_matrix.clone()
        intrinsics = intrinsics.clone()
        # vertices = vertices.clone()

        batch_size = vertices.shape[0]

        K = torch.tensor( [
                [0,    0,   0,   0],
                [0,    0,   0,   0],
                [0,    0,    0,   1],
                [0,    0,    1.0,   0]
            ]   
            ).to(vertices.device).view(1,4,4)
        K = K.repeat(batch_size,1,1)
        K[:,0:3,0:3] = intrinsics
        K[:,0:1] *=resolution[1]
        K[:,1:2] *=resolution[0]
        K[:,2,2] = 0

        K[:,0,0] *=-1 
        K[:,1,1] *=-1 


        
        image_size =torch.zeros_like(K[:,0:2,0])
        image_size[:,0]=resolution[0]
        image_size[:,1]=resolution[1]


        # print(image_size)
        # import pdb; pdb.set_trace()
        

        try:
            world_position = torch.inverse(cam2world_matrix)
            R =  world_position[:,0:3,0:3].inverse()
            T = world_position[:,0:3,3]
        except:
            print('camera wrong')
            world_position = torch.eye(4).unsqueeze(0).to(cam2world_matrix.device)
            R=  world_position[:,0:3,0:3]
            T = world_position[:,0:3,3]

        cameras = PerspectiveCameras(R=R,T=T,K=K,in_ndc  = False,image_size=image_size,device=vertices.device)

        # import pdb; pdb.set_trace()
        raster_settings_mesh = RasterizationSettings(
                    image_size=resolution,
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
        
        

        mesh_base = Meshes(verts=vertices,faces=self.faces.expand(batch_size,-1,-1).to(vertices.device))


        vertices_norm = self.vertices_color.expand(batch_size,-1,-1).to(vertices.device)
        mesh_base.textures = TexturesVertex(verts_features=vertices_norm)
    


        self.renderer.to(vertices.device)
        
        mask_image = self.renderer(mesh_base)[:,:,:,0:3]

        # import pdb; pdb.set_trace()
        mask_image = mask_image.permute(0,3,1,2)


        del vertices
        del cam2world_matrix
        del intrinsics
        # del 
        return mask_image

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
import pyrender
import numpy as np
from tqdm import tqdm
from glob import glob
from pyrender.constants import RenderFlags

class Renderer:
    def __init__(self, resolution=(1080,1080)):
        self.resolution = resolution

        self.renderer = pyrender.OffscreenRenderer(
            viewport_height=self.resolution[0],
            viewport_width=self.resolution[1],
            point_size=1.0
        )

        # set the scene
        self.scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=(0.3, 0.3, 0.3))

        light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=1)

        light_pose = np.eye(4)
        light_pose[:3, 3] = [0, -1, -5]
        self.scene.add(light, pose=light_pose)

        light_pose[:3, 3] = [0, 1, -5]
        self.scene.add(light, pose=light_pose)

        light_pose[:3, 3] = [1, 1, -4]
        self.scene.add(light, pose=light_pose)

    def render(self, mesh, camera, camera_pose, bkgd=None, color=None, color_map=None, wireframe=False):
        if color is None:
            color = colorsys.hsv_to_rgb(np.random.rand(), 0.5, 1.0)
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            alphaMode='OPAQUE',
            baseColorFactor=(color[0], color[1], color[2], 1.0)
        )

        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
        if color_map is not None:
            mesh.primitives[0].color_0 = color_map

        mesh_node = self.scene.add(mesh, 'mesh')
        cam_node = self.scene.add(camera, pose=camera_pose)

        if wireframe:
            render_flags = RenderFlags.RGBA | RenderFlags.ALL_WIREFRAME
        else:
            render_flags = RenderFlags.RGBA

        rgb, _ = self.renderer.render(self.scene, flags=render_flags)
        valid_mask = (rgb[:, :, -1] > 0)[:, :, np.newaxis]
        if rgb.shape[2] == 4:
            if bkgd is None:
                output_img = rgb[:, :, :-1] * valid_mask
            else:
                output_img = rgb[:, :, :-1] * valid_mask + (1 - valid_mask) * bkgd
        else:
            if bkgd is None:
                output_img = rgb * valid_mask
            else:
                output_img = rgb * valid_mask + (1 - valid_mask) * bkgd[:, :, :-1]
        render_img = output_img.astype(np.uint8)

        self.scene.remove_node(mesh_node)
        self.scene.remove_node(cam_node)

        return render_img, valid_mask

renderer = Renderer(resolution=size)
camera = pyrender.camera.IntrinsicsCamera(
        fx=focal,
        fy=focal,
        cx=W/2,
        cy=H/2,
    )

for i in tqdm(range(len(frame_ids))):
    frame_id = frame_ids[i]
    cam = cams[frame_id]
    cur_pose = poses[frame_id]
    cur_betas = betas[frame_id]
    cur_transl = np.array([cam[2], cam[3], 2*focal/(cam[0]*W)])
    camera_pose = np.diag(np.array([1, -1, -1, 1], dtype=np.float32))
    moco_dict['frames'] += [{
        'file_path': f'{frame_id:04d}.png',
        'camera_pose': camera_pose,
        'pose': cur_pose,
        'betas': cur_betas,
        'transl': cur_transl,
    }]

    if vis:
        verts = smpl.forward(torch.from_numpy(cur_pose).unsqueeze(dim=0).float(), \
                                        torch.from_numpy(cur_betas).unsqueeze(dim=0).float())[0]
        mesh = trimesh.Trimesh(verts + cur_transl, smpl.faces)
        rendered_img, _mask = renderer.render(mesh, camera, camera_pose, \
            bkgd=imageio.imread(f'{save_folder}/images/{frame_id:04d}.png'), color=(0.5, 0.8, 1.0))
        vis_imgs += [rendered_img]
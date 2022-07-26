import torch
from torch import nn

from .utils import face_vertices


class CudaRenderer(nn.Module):
    """ A cuda-based renderer, adapted from the DECA implementation by YadiraF
    (https://github.com/YadiraF/DECA/blob/master/decalib/utils/renderer.py). """
    def __init__(self, height, width=None, device='cuda'):
        """ use fixed raster_settings for rendering faces. """
        super().__init__()
        
        self.h = height
        self.w = width if width is not None else height
        self.device = device

    def forward(self, v, f, attrs=None):
        
        h, w = self.height, self.width
        device = self.device
        bz = v.shape[0]

        depth_buffer = torch.zeros([bz, h, w]).float().to(device) + 1e6
        triangle_buffer = torch.zeros([bz, h, w]).int().to(device) - 1
        baryw_buffer = torch.zeros([bz, h, w, 3]).float().to(device)

        v = v.clone().float()
        
        v[..., :2] = -v[..., :2]
        v[..., 0] = v[..., 0] * w/2 + w/2
        v[..., 1] = v[..., 1] * h/2 + h/2
        v[..., 0] = w - 1 - v[..., 0]
        v[..., 1] = h - 1 - v[..., 1]
        v[..., 0] = -1 + (2 * v[..., 0] + 1) / w
        v[..., 1] = -1 + (2 * v[..., 1] + 1) / h

        v = v.clone().float()
        v[..., 0] = v[..., 0] * w/2 + w/2 
        v[..., 1] = v[..., 1] * h/2 + h/2 
        v[..., 2] = v[..., 2] * w/2
        f_vs = face_vertices(v, f)

        # Import here instead of toplevel, so the rasterize_cuda does not necessarily
        # have to be installed
        from rasterize_cuda import rasterize
        rasterize(f_vs, depth_buffer, triangle_buffer, baryw_buffer, h, w)

        pix_to_face = triangle_buffer[:, :, :, None].long()
        bary_coords = baryw_buffer[:, :, :, None, :]
        vismask = (pix_to_face > -1).float()
        D = attrs.shape[-1]
        attrs = attrs.clone()
        attrs = attrs.view(attrs.shape[0] * attrs.shape[1], 3, attrs.shape[-1])
        N, H, W, K, _ = bary_coords.shape
        mask = pix_to_face == -1
        pix_to_face = pix_to_face.clone()
        pix_to_face[mask] = 0
        idx = pix_to_face.view(N * H * W * K, 1, 1).expand(N * H * W * K, 3, D)
        pixel_face_vals = attrs.gather(0, idx).view(N, H, W, K, 3, D)
        pixel_vals = (bary_coords[..., None] * pixel_face_vals).sum(dim=-2)
        pixel_vals[mask] = 0  # Replace masked values in output.
        pixel_vals = pixel_vals[:, :, :, 0].permute(0, 3, 1, 2)
        pixel_vals = torch.cat([pixel_vals, vismask[:, :, :, 0][:, None, :, :]], dim=1)
        
        return pixel_vals
    

class PytorchRenderer(nn.Module):

    from pytorch3d.io import load_obj
    from pytorch3d.structures import Meshes
    from pytorch3d.renderer import look_at_view_transform, FoVPerspectiveCameras
    from pytorch3d.renderer import RasterizationSettings, MeshRenderer, MeshRasterizer
    from pytorch3d.renderer import DirectionalLights, SoftPhongShader, TexturesVertex

    def __init__(self, obj_filename):
        super().__init__()

        verts, faces, aux = load_obj(obj_filename)
        faces = faces.verts_idx[None, ...].cuda()
        self.register_buffer('faces', faces)

        R, T = look_at_view_transform(2.7, 10.0, 10.0)
        self.cameras = FoVPerspectiveCameras(device='cuda:0', R=R, T=T, fov=6)
        raster_settings = RasterizationSettings(
            image_size=512,
            blur_radius=0.0,
            faces_per_pixel=1,
            perspective_correct=True
        )

        lights = DirectionalLights(
            device='cuda:0',
            direction=((0, 0, 1),),
            ambient_color=((0.4, 0.4, 0.4),),
            diffuse_color=((0.35, 0.35, 0.35),),
            specular_color=((0.05, 0.05, 0.05),))

        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=self.cameras, raster_settings=raster_settings),
            shader=SoftPhongShader(device='cuda:0', cameras=self.cameras, lights=lights)
        )

    def render_mesh(self, vertices, faces=None, verts_rgb=None):
        B, N, V = vertices.shape
        if faces is None:
            faces = self.faces.repeat(B, 1, 1)
        else:
            faces = faces.repeat(B, 1, 1)

        if verts_rgb is None:
            verts_rgb = torch.ones_like(vertices)
        textures = TexturesVertex(verts_features=verts_rgb.cuda())
        meshes = Meshes(verts=vertices, faces=faces, textures=textures)

        rendering = self.renderer(meshes).permute(0, 3, 1, 2)
        color = rendering[:, 0:3, ...]

        return color
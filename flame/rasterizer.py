import torch
from torch.nn import Module
from rasterize_cuda import rasterize

from .recon.utils import face_vertices


class CudaRasterizer(Module):
    """ Alg: https://www.scratchapixel.com/lessons/3d-basic-rendering/rasterization-practical-implementation"""
    def __init__(self, height, width=None, device='cuda'):
        """
        use fixed raster_settings for rendering faces
        """
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
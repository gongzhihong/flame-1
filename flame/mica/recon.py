import torch
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict

from ..core import FlameReconModel
from ..decoders import FLAME
from .encoders import MappingNetwork, Arcface


class MicaReconModel(FlameReconModel):

    # May have some speed benefits
    torch.backends.cudnn.benchmark = True

    def __init__(self, device='cuda'):
        self.device = device
        self._load_cfg()  # method inherited from parent
        self._create_submodels()
        self._load_submodels()

    def _create_submodels(self):
        """ Loads the submodels associated with MICA. To summarizes:
        - `E_arcface`: predicts a 512-D embedding for a (cropped, 112x112) image
        - `E_flame`: predicts (coarse) FLAME parameters given a 512-D embedding
        - `D_flame`: outputs a ("coarse") mesh given shape FLAME parameters
        """
        self.E_arcface = Arcface().to(self.device)
        self.E_arcface.eval()
        self.E_flame = MappingNetwork(512, 300, 300).to(self.device)
        self.E_flame.eval()
        self.D_flame = FLAME(self.cfg['flame_path'], n_shape=300, n_exp=0).to(self.device)
        self.D_flame.eval()
        torch.set_grad_enabled(False)  # apparently speeds up forward pass, too

    def _load_submodels(self):
        """ Loads the weights for the Arcface submodel as well as the MappingNetwork
        that predicts FLAME shape parameters from the Arcface output. """
        checkpoint = torch.load(self.cfg['mica_path'])
        self.E_arcface.load_state_dict(checkpoint['arcface'])
        
        # The original weights also included the data for the FLAME model (template
        # vertices, faces, etc), which we don't need here, because we use a common
        # FLAME decoder model (in decoders.py)
        new_checkpoint = OrderedDict()
        for key, value in checkpoint['flameModel'].items():
            # The actual mapping-network weights are stored in keys starting with
            # regressor.
            if 'regressor.' in key:
                new_checkpoint[key.replace('regressor.', '')] = value
        
        self.E_flame.load_state_dict(new_checkpoint)

    def _encode(self, image):
        image = self._check_input(image, expected_wh=(112, 112))        
        out_af = self.E_arcface(image)  # output of arcface
        out_af = F.normalize(out_af)
        return self.E_flame(out_af)

    def _decode(self, code):

        v, _ = self.D_flame(code)
        v = v.squeeze().detach().cpu()
        out = {'v': v, 'mat': np.eye(4)}

        return out

    def __call__(self, image):
        image = self._check_input(image, expected_wh=(112, 112))
        enc_dict = self._encode(image)
        dec_dict = self._decode(enc_dict)
        return dec_dict
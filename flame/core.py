import yaml
import torch
from pathlib import Path
from abc import ABCMeta, abstractmethod


class FlameReconModel(metaclass=ABCMeta):

    def _load_cfg(self):
        """ Loads a (default) config file. """
        data_dir = Path(__file__).parent / 'data'
        cfg = data_dir / 'config.yaml'

        if not cfg.is_file():
            raise ValueError(f"Could not find {str(cfg)}! "
                              "Did you run the validate_external_data.py script?")

        with open(cfg, "r") as f_in:
            self.cfg = yaml.safe_load(f_in)

    def _check_input(self, image, expected_wh=(224, 224), dtype=torch.float32):
        """ Assumes that self.device attribute exists. """
        if not torch.is_tensor(image):
            # Expects a 1 x 224 x 224 x 3 tensor
            image = torch.from_numpy(image).to(self.device)

        # Check data type       
        image = image.to(dtype=dtype)

        if image.shape[0] != 1:
            # Add singleton batch dimension
            image = image.unsqueeze(dim=0)

        if image.shape[1] != 3 and image.shape[3] == 3:
            # Expects channels (RGB) first, not last
            image = image.permute((0, 3, 1, 2))

        if image.shape[2:] != expected_wh:
            w, h = expected_wh
            raise ValueError(f"Image should have dimensions {w} (w) x {h} (h)!")

        return image

    def get_faces(self):
        
        if hasattr(self, 'dense'):
            if self.dense:
                return self.dense_template['f']
        else:   
            # Cast to cpu and to numpy
            faces = self.faces.cpu().detach().numpy().squeeze()
            return faces

    def close(self):
        
        if hasattr(self, 'tform'):
            self.tform = None
            
        pass

import yaml
import torch
from pathlib import Path
from .data import get_template_flame


class FlameReconModel:

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

        if image.dim == 3:
            if image.size(0) == 3:
                image = image.unsqueeze(dim=0)
        
        if image.shape[1] != 3 and image.shape[3] == 3:
            # Expects channels (RGB) first, not last
            image = image.permute((0, 3, 1, 2))
        
        if image.shape[2:] != expected_wh:
            w, h = expected_wh
            raise ValueError(f"Image should have dimensions {w} (w) x {h} (h)!")

        return image

    def get_faces(self):
        """ Retrieves the faces (polygons) associated with the predicted vertex mesh. """
        if hasattr(self, 'dense'):
            dense = self.dense
        else:
            # Assume that we're using the coarse version (e.g., for MICA)
            dense = False

        template = get_template_flame(dense=dense)
        return template['f']

    def close(self):
        
        if hasattr(self, 'tform'):
            self.tform = None
            
        pass

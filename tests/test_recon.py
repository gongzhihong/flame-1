import torch
import pytest
import numpy as np

from PIL import Image
from pathlib import Path
from flame import FlameReconModel


@pytest.mark.parametrize("name", ['deca-coarse', 'deca-dense', 'emoca-coarse', 'emoca-dense'])
def test_recon(name, device='cuda'):
    
    model = FlameReconModel(name, img_size=(224, 224), device=device)
    model.tform = np.eye(3)  # no crop

    img = np.array(Image.open(Path(__file__).parent / 'obama.jpeg'))
    img = img.transpose(2, 0, 1) / 255.
    img = torch.tensor(img).float().unsqueeze(0).to(device)
    out = model(img)
    
    assert(out['mat'].shape == (4, 4))
    
    if 'dense' in name:
        assert(out['v'].shape == (59315, 3))
    else:
        assert(out['v'].shape == (5023, 3))

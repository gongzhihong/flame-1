import os
import torch
import pytest
import numpy as np

from PIL import Image
from pathlib import Path
from flame import DecaReconModel


@pytest.fixture()
def example_img(device):
    """ Prepares an example image to be reused for the different model tests. """
    img = np.array(Image.open(Path(__file__).parent / 'obama_cropped.jpeg'))
    img = img.transpose(2, 0, 1) / 255.
    img = torch.tensor(img).float().unsqueeze(0).to(device)
    return img    


@pytest.mark.parametrize("name", ['deca-coarse', 'deca-dense', 'emoca-coarse', 'emoca-dense'])
@pytest.mark.parametrize("device", ['cuda', 'cpu'])
def test_recon(name, device, example_img):

    if 'GITHUB_ACTIONS' in os.environ and device == 'cuda':
        return
    
    model = DecaReconModel(name, img_size=(224, 224), device=device)
    out = model(example_img)
    
    assert(out['mat'].shape == (4, 4))
    
    if 'dense' in name:
        assert(out['v'].shape == (59315, 3))
    else:
        assert(out['v'].shape == (5023, 3))

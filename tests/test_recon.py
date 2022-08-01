import os
import torch
import pytest
import numpy as np

from PIL import Image
from pathlib import Path
from flame import DecaReconModel, MicaReconModel


@pytest.fixture()
def example_img(name, device):
    """ Prepares an example image to be reused for the different model tests. """
    
    if name == 'mica':
        f_in = Path(__file__).parent / 'obama_cropped_112.png'
    else:
        f_in = Path(__file__).parent / 'obama_cropped_224.jpeg'
    
    img = np.array(Image.open(f_in))
    img = img.transpose(2, 0, 1)
    img = torch.tensor(img).float().unsqueeze(0).to(device)
    
    if name == 'mica':
        img = (img - 127.5) * (1 / 127.5)
    else:
        img = img / 255.

    return img    


@pytest.mark.parametrize("name", ['mica', 'deca-coarse', 'deca-dense', 'emoca-coarse', 'emoca-dense'])
@pytest.mark.parametrize("device", ['cuda', 'cpu'])
def test_deca_recon(name, device, example_img):

    if 'GITHUB_ACTIONS' in os.environ and device == 'cuda':
        return
    
    if name == 'mica':
        model = MicaReconModel(device=device)
    else:
        model = DecaReconModel(name, img_size=(224, 224), device=device)

    out = model(example_img)
    
    assert(out['mat'].shape == (4, 4))
    
    if 'dense' in name:
        assert(out['v'].shape == (59315, 3))
    else:
        assert(out['v'].shape == (5023, 3))


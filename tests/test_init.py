import os
import pytest
from flame import FlameReconModel


@pytest.mark.parametrize("name", ['deca-coarse', 'deca-dense', 'emoca-coarse', 'emoca-dense'])
def test_init_model(name, device='cuda'):

    if 'GITHUB_ACTIONS' in os.environ:
        device = 'cpu'
    
    model = FlameReconModel(name, img_size=(224, 224), device=device)


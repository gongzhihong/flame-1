import os
import pytest
from pathlib import Path
from flame.crop import FanCropModel, InsightFaceCropModel


@pytest.mark.parametrize("Model", [FanCropModel, InsightFaceCropModel])
@pytest.mark.parametrize("device", ['cuda', 'cpu'])
def test_recon(Model, device):

    if 'GITHUB_ACTIONS' in os.environ and device == 'cuda':
        return
    
    img = Path(__file__).parent / 'obama2.jpeg'
    crop_model = Model(device=device)
    out = crop_model(img)
    print(out)

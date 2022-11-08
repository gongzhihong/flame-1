import os
import pytest
from pathlib import Path
from flame.crop import FanCropModel, InsightFaceCropModel


@pytest.mark.parametrize("Model", [FanCropModel, InsightFaceCropModel])
@pytest.mark.parametrize("device", ['cuda', 'cpu'])
def test_recon(Model, device):

    if 'GITHUB_ACTIONS' in os.environ and device == 'cuda':
        return
    
    img = Path(__file__).parent / 'obama.jpeg'
    crop_model = Model(device=device)
    out = crop_model(img)
    
    if Model == FanCropModel:
        img_crop = out[0]
        img_crop = crop_model.to_numpy(img_crop, scale=127.5, mean=127.5, to_rgb=True)
        assert(img_crop.shape[1:] == (224, 224, 3))
    else:
        img = crop_model.to_numpy(out, scale=255., mean=0, to_rgb=False)
        assert(img.shape[1:] == (112, 112, 3))

# flame: 3D face reconstruction models based on the FLAME topology

## Installation

To install this package, first download the package from github (using `git`), then
download the necessary data, and finally install the downloaded package (`pip install .`).

For details on which data to download and how to do so, check [this page](https://medusa.lukas-snoek.com/medusa/getting_started/installation.html).

## Usage

```python
from flame.data import get_example_img
from flame.crop import CropModel
from flame import FlameReconModel

# img = path to a standard RGB jpg image
img = get_example_img()

# Most Flame-based models (such as EMOCA and DECA) expect a cropped (224 x 224) image
# as input (or, actually, a 1 x 3 x 224 x 224 torch Tensor), so we'll use the
# CropModel class from the package
crop_model = CropModel(device='cpu')
cropped_img = crop_model(img)

# Initialize reconstrution model (other options include 'emoca-dense', 'deca-coarse' and
# 'deca-dense')
recon_model = FlameReconModel(name='emoca-coarse', device='cpu')

# If we want our world matrix to be relative to the original (rather than cropped)
# image, we need to pass the cropping matrix parameters to the model
recon_model.tform = crop_model.tform.params

# Perform the actual reconstruction, which returns a dictionary with two keys
out = recon_model(cropped_img)

# v = vertices
print(out['v'].shape)  # (5023, 3)

# mat = world matrix
print(out['mat'].shape)  # (4, 4) 
```

# flame: 3D face reconstruction models based on the FLAME topology

## Installation

See [here](https://lukas-snoek.com/medusa/getting_started/installation.html).

## Usage

```python
from medusa.data import get_example_frame
from medusa.recon import FAN
from flame import FlameReconModel

img = get_example_frame()
model = FlameRecon_model(name='emoca-coarse', img_size=img.shape[:2], device='cpu')
fan = FAN(lm_type='2D')
cropped_img = fan.prepare_for_emoca(img)
model.tform = fan.tform.params
out = model(cropped_img)  # dict with keys 'v' and 'mat'
```
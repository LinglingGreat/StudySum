## 保存网格图
```
from PIL import Image
def image_grid(imgs, rows, cols):
	assert len(imgs) == rows*cols
	w, h = imgs[0].size
	grid = Image.new('RGB', size=(cols*w, rows*h))
	grid_w, grid_h = grid.size
	for i, img in enumerate(imgs):
		grid.paste(img, box=(i%cols*w, i//cols*h))
	return grid

image_grid(images, 1, num_samples)
grid.save("clipguide.png")
```

## 展示图片

```python
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from PIL import Image


def show_images(x):

	"""Given a batch of images x, make a grid and convert to PIL"""
	x = x * 0.5 + 0.5 # Map from (-1, 1) back to (0, 1)
	grid = torchvision.utils.make_grid(x)
	grid_im = grid.detach().cpu().permute(1, 2, 0).clip(0, 1) * 255
	grid_im = Image.fromarray(np.array(grid_im).astype(np.uint8))
	return grid_im

  
  

def make_grid(images, size=64):

	"""Given a list of PIL images, stack them together into a line for easy viewing"""
	
	output_im = Image.new("RGB", (size * len(images), size))
	for i, im in enumerate(images):
		output_im.paste(im.resize((size, size)), (i * size, 0))
	return output_im
```
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



from PIL import Image
import numpy as np

# 图像的手绘效果的几个特征
# 黑白灰色
# 边界线条较重
# 相同或相近色彩趋于白色
# 略有光源效果
a = np.asarray(Image.open('./beijing.jpg').convert('L')).astype('float')

# 手绘风格是在对图像灰度化的基础上，由立体效果和明暗效果叠加而成
# 灰度值代表了图像的明暗变化，梯度值是灰度的变化率，可以通过调整像素的梯度值间接改变图像的明暗程度，立体效果通过添加虚拟深度值实现
# 利用像素之间的梯度值和虚拟深度值对图像进行重构，根据灰度变化来模拟人类视觉的远近程度
depth = 10. 						# 预设深度值为10，取值范围(0-100)
grad = np.gradient(a)				# 取图像灰度的梯度值，是一个包含x方向和y方向的数据对
grad_x, grad_y = grad 				# 分别取横纵图像梯度值
grad_x = grad_x*depth/100.          # 根据深度调整x和y方向的梯度值
grad_y = grad_y*depth/100.

# 光源效果，根据灰度变化来模拟人类视觉的远近程度
vec_el = np.pi/2.2 					# 光源的俯视角度，弧度值
vec_az = np.pi/4. 					# 光源的方位角度，弧度值
# np.cos(vec_el)为单位光线在地平面上的投影长度，dx,dy,dz是光源对x/y/z三方向的影响程度
dx = np.cos(vec_el)*np.cos(vec_az) 	# 光源对x 轴的影响
dy = np.cos(vec_el)*np.sin(vec_az) 	# 光源对y 轴的影响
dz = np.sin(vec_el) 				# 光源对z 轴的影响

A = np.sqrt(grad_x**2 + grad_y**2 + 1.)     # 构造x和y轴梯度的三维归一化单位坐标系
uni_x = grad_x/A
uni_y = grad_y/A
uni_z = 1./A
b = 255*(dx*uni_x + dy*uni_y + dz*uni_z) 	# 光源归一化，梯度与光源相互作用，将梯度转化为灰度
# 图像生成
b = b.clip(0,255)   # 为避免数据越界，将生成的灰度值裁剪至0-255区间

im = Image.fromarray(b.astype('uint8')) 	# 重构图像
im.save('./beijingHD.jpg')


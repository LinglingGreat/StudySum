
from PIL import Image
import numpy as np
# 图像变换的基本流程
# 读入图像，对图像的RGB值进行运算，保存为另一个图像

# 图像是一个由像素组成的二维矩阵，每个元素是一个RGB值
im = np.array(Image.open('./beijing.jpg'))
print(im.shape, im.dtype)
# 图像读取到数组中，是一个三维数组，维度分别是高度、宽度和像素RGB值

# 图像的变换
# 读入图像后，获得像素RGB值，修改后保存为新的文件
a = np.array(Image.open('./fcity.jpg'))
print(a.shape, a.dtype)
b = [255, 255, 255] - a    # 对图像的每个像素，计算RGB三个通道的补值
im = Image.fromarray(b.astype('uint8'))   # 将数组b生成新的图像
im.save('./fcity2.jpg')

# 图像变换成灰度，类似底片
a = np.array(Image.open('./fcity.jpg').convert('L'))    #  将一个彩色的图片变成灰度值的图片
# 数组a是一个二维数据，每一个元素对应灰度值，而不是RGB值
b = 255 - a    # 对灰度值取反
im = Image.fromarray(b.astype('uint8'))
im.save('./fcity3.jpg')

# 图像变换成颜色较淡的灰度图片
a = np.array(Image.open('./fcity.jpg').convert('L'))
c = (100/255)*a + 150   # 区间变换，将当前图片的灰度值进行区间压缩，再扩充一个区间范围
im = Image.fromarray(c.astype('uint8'))
im.save('./fcity4.jpg')

# 图像变换成灰度非常重，类似黑色
a = np.array(image.open('./fcity.jpg').convert('L'))
d = 255 * (a/255)**2     # 像素平方
im = Image.fromarray(d.astype('uint8'))
im.save('./fcity.jpg')






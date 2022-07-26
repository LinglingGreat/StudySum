import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# 物理学中，引力波是因为时空弯曲对外以辐射形式传播的能量

# 从配置文档中读取时间相关数据
rate_h, hstrain= wavfile.read(r"H1_Strain.wav","rb")
rate_l, lstrain= wavfile.read(r"L1_Strain.wav","rb")
# reftime, ref_H1 = np.genfromtxt('GW150914_4_NR_waveform_template.txt').transpose()
reftime, ref_H1 = np.genfromtxt('wf_template.txt').transpose()   # 使用python123.io下载文件
# 提供的引力波理论模型，reftime是时间序列，ref_H1是洗好的数据
# genfromtxt主要执行两个循环，将文件的每一行转化成字符串序列；将每个字符串序列转换成相应的数据类型
# 读取出来的是一个两行的矩阵

# 读取应变数据
htime_interval = 1/rate_h    # 得到波形的时间间隔
ltime_interval = 1/rate_l

# 丢失信号起始点
htime_len = hstrain.shape[0]/rate_h      # 矩阵第一维度的长度即数据点的个数/rate，得到函数在坐标轴上的总长度
htime = np.arange(-htime_len/2, htime_len/2 , htime_interval)
# 画出以时间为X轴，应变数据为Y轴的图像，并设置标题和坐标轴的标签

# 绘制H1 Strain, 使用来自H1探测器的数据作图
fig = plt.figure(figsize=(12, 6))    # 创建一个大小为12*6的绘图空间
plth = fig.add_subplot(221)
plth.plot(htime, hstrain, 'y')
plth.set_xlabel('Time (seconds)')
plth.set_ylabel('H1 Strain')
plth.set_title('H1 Strain')

ltime_len = lstrain.shape[0]/rate_l
ltime = np.arange(-ltime_len/2, ltime_len/2 , ltime_interval)
# 放在绘图区域的第一列右边
pltl = fig.add_subplot(222)
pltl.plot(ltime, lstrain, 'g')
pltl.set_xlabel('Time (seconds)')
pltl.set_ylabel('L1 Strain')
pltl.set_title('L1 Strain')

# 放在绘图区域的第二列
pltref = fig.add_subplot(212)
pltref.plot(reftime, ref_H1)
pltref.set_xlabel('Time (seconds)')
pltref.set_ylabel('Template Strain')
pltref.set_title('Template')
fig.tight_layout()     # 自动调整图像外部边缘

plt.savefig("Gravitational_Waves_Original.png")
plt.show()
plt.close(fig)

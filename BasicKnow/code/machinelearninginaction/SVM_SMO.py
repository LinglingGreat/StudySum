# -*- coding: utf-8 -*-
# @Author: LiLing
# @Date:   2018-09-11 08:45:56
# @Last Modified by:   Liling
# @Last Modified time: 2018-09-11 09:03:06
"""
最大化数据集中所有点到分隔面的最小间隔的2倍
label*(w^T*x+b)称为点到分隔面的函数间隔，labe*(w^T*x+b)/||w||称为点到分隔面的几何间隔
目标函数是argmax{min(label*(w^T*x+b))*1/||w||}
w和b可以通过放缩使得min(label*(w^T*x+b))=1，原问题转变为有约束的优化问题
进而用拉格朗日乘子法
"""

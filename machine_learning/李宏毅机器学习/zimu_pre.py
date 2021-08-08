# -*- encoding: utf-8 -*-
'''
@文件        :zimu_pre.py
@说明        :
@时间        :2021/06/06 21:32:13
@作者        :codingling
'''
import re

input_file = "DNN.txt"
output_file = 'DNN_op.txt'
with open(input_file, encoding='utf-8') as f:
    lines = f.readlines()

lines = [i for i in lines if not re.match("^[0-9]{2}:[0-9]{2}$", i.strip())]

with open(output_file, 'w', encoding='utf-8') as f:
    f.writelines(lines)

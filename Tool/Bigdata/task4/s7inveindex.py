# -*- coding: utf-8 -*-
# @Author: LiLing
# @Date:   2019-08-03 01:01:59
# @Last Modified by:   LL
# @Last Modified time: 2019-08-03 20:06:22
#!/usr/bin/python
# -*- coding: utf-8 -*-
from mrjob.job import MRJob
import re


class MRratingAvg(MRJob):
    '''
    计算每个用户的平均评分
    '''
    def mapper(self, _, line):
        line = line.strip().split(',')
        docum = line[0]
        words = line[1].split()
        for word in words:
            yield word, docum

    def reducer(self, word, docum):
        #shuff and sort 之后
        temp = []
        for d in docum:
            temp.append(d)
        yield word, temp


if __name__ == '__main__':
    MRratingAvg.run()
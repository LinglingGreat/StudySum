# -*- coding: utf-8 -*-
# @Author: LiLing
# @Date:   2019-08-03 01:01:25
# @Last Modified by:   LL
# @Last Modified time: 2019-08-03 11:20:57
#!/usr/bin/env python
#! coding=utf-8
from mrjob.job import MRJob
import re


class MRdedup(MRJob):
    '''
    实现去重
    '''
    def mapper(self, _, line):
        line = line.strip().split(',')
        yield line[0], (line[1], line[2])

    def reducer(self, id, values):
        exist_list = []
        for value in values:
            if value not in exist_list:
                exist_list.append(value)
                print id, (value[0], value[1])


if __name__ == '__main__':
    MRdedup.run() #run()方法，开始执行MapReduce任务。


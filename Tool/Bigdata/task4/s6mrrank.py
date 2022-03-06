# -*- coding: utf-8 -*-
# @Author: LiLing
# @Date:   2019-08-03 01:01:06
# @Last Modified by:   LL
# @Last Modified time: 2019-08-03 20:06:09
#!/usr/bin/python
# -*- coding: utf-8 -*-
from mrjob.job import MRJob
import re


class MRrank(MRJob):
    '''
    计算每个用户的平均评分
    '''
    def mapper(self, _, line):
        user_id, score = line.split(',')
        yield 'rank', (float(score), user_id)

    def reducer(self, user_id, values):
        # shuff and sort 之后
        l = list(values)
        l.sort()
        for key in l:
            print key[1], key[0]


if __name__ == '__main__':
    MRrank.run()
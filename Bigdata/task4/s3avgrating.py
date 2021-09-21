# -*- coding: utf-8 -*-
# @Author: LiLing
# @Date:   2019-08-03 00:28:54
# @Last Modified by:   LL
# @Last Modified time: 2019-08-03 00:47:45
#!/usr/bin/python
# -*- coding: utf-8 -*-
from mrjob.job import MRJob


class MRratingAvg(MRJob):
    '''
    计算每个用户的平均评分
    '''
    def mapper(self, _, line):
        user_id, item_id, rating, timestamp = line.strip().split(',')
        if not user_id.isdigit():
            return
        yield user_id, float(rating)

    def reducer(self, user_id, values):
        #shuff and sort 之后
        '''
        (user_id,[rating1,rating2,rating3])
        '''
        l = list(values)
        yield (user_id, sum(l)/len(l))


if __name__ == '__main__':
    MRratingAvg.run()

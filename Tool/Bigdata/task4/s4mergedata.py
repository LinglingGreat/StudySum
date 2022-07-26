# -*- coding: utf-8 -*-
# @Author: LiLing
# @Date:   2019-08-03 00:28:54
# @Last Modified by:   LL
# @Last Modified time: 2019-08-03 00:55:39
#!/usr/bin/python
# -*- coding: utf-8 -*-
from mrjob.job import MRJob
import sys
reload(sys)
sys.setdefaultencoding('utf-8')


class MRdataMerge(MRJob):
    '''
    根据user_id merge data和item
    '''
    def mapper(self, _, line):
        line = line.strip().split('|')
        print(line)
        # if len(line) > 1:
        #     movie_id, movie_title = line[0], line[1]
        #     yield movie_id, movie_title
        # else:
        line = line[0].strip().split('\t')
        user_id, item_id, rating, _ = line
        yield item_id, (user_id, rating)

    def reducer(self, item_id, values):
        movie_title = ''
        print values
        # for items in values:
        #     if len(items) == 1:
        #         movie_title = items
        yield item_id, values
        # for item1, item2, item3 in values:
        #     user_id, rating = item1, item2
        #     yield user_id, (item_id, movie_title, rating)


if __name__ == '__main__':
    MRdataMerge.run()

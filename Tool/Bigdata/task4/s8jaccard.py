# -*- coding: utf-8 -*-
# @Author: LiLing
# @Date:   2019-08-03 01:00:30
# @Last Modified by:   LL
# @Last Modified time: 2019-08-03 13:14:47
#!/usr/bin/python
# -*- coding: utf-8 -*-
from mrjob.job import MRJob
 
class MRjaccard(MRJob):
    def mapper(self, _, line):
        line = line.strip().split()
        user1, user2 = line[0], line[1]
        item1, item2 = line[2], line[3]
        yield 'simi', (user1, user2, item1, item2)

    def reducer(self, key, lines):
        #shuff and sort 之后
        for line in lines:
            users = line[:2]
            items = line[2:]
            unions = len(set(items[0]).union(set(items[1])))
            intersections = len(set(items[0]).intersection(set(items[1])))
            yield (users[0], users[1]), (intersections, unions, float(intersections)/unions)
 
 
if __name__ == '__main__':
    MRjaccard.run()

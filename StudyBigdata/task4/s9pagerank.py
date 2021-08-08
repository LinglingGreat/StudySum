# -*- coding: utf-8 -*-
# @Author: LiLing
# @Date:   2019-08-03 01:00:55
# @Last Modified by:   LL
# @Last Modified time: 2019-08-03 13:20:21
#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Iterative implementation of the PageRank algorithm:
http://en.wikipedia.org/wiki/PageRank
"""
from mrjob.job import MRJob
from mrjob.protocol import JSONProtocol
from mrjob.step import MRStep


class MRPageRank(MRJob):

    def mapper(self, _, line):
        data = line.strip().split(',')
        p = float(data[1])
        target = data[2:]
        n = len(target)
        for i in target:
            yield i, p/n

    def reducer(self, id, p):
        ''' reducer of pagerank algorithm'''
        alpha = 0.8
        N = 4  # Size of the web pages
        value = 0.0
        for v in p:
            value += float(v)
        values = alpha * value + (1 - alpha) / N
        yield id, values


if __name__ == '__main__':
    MRPageRank.run()
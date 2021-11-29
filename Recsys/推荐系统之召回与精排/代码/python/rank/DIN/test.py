#coding=utf-8
"""
Author:tongqing
data:2020/5/7 15:54
desc
"""
import tensorflow as tf
import numpy as np

s=tf.convert_to_tensor(np.random.random([3,5]))
s1=tf.convert_to_tensor(np.random.random([3,5]))
# k=tf.tile(s,[1,7])
# re=tf.reshape(k,[3,7,5])
#
# k1=tf.expand_dims(s,1)
# re1=tf.tile(k1,[1,7,1])
t=tf.concat([s,s1],axis=-1)
print(t)

sess=tf.Session()
print(sess.run(t))

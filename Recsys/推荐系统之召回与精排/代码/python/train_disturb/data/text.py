#
import tensorflow as tf
import numpy as np
import pandas as pd
import tensorflow as tf






#
#
#
#
#
# x = tf.placeholder("float",shape=[None,2])
# y = tf.placeholder("float",shape=[None,1])
# with tf.variable_scope("weight",regularizer=tf.contrib.layers.l2_regularizer(0.1)) as scope:
#     W = tf.get_variable("weight", [2, 1], initializer=tf.truncated_normal_initializer(0.1))
# b = tf.Variable(tf.zeros([1]), name='bais')
#
# glob_step = tf.train.get_or_create_global_step()  # 获得迭代次数
# z = tf.matmul(x, W) + b
# tf.summary.histogram('z', z)
# cost1 = tf.losses.mean_squared_error(y, z)
# cost=tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,cost1)
# cost=cost1+12
# tf.summary.scalar('loss_function', cost)
# lr = 0.01
# opt = tf.train.AdagradOptimizer(lr).minimize(cost, global_step=glob_step)
# saver = tf.train.Saver(max_to_keep=1)
# merged_summary_op = tf.summary.merge_all()
# init = tf.global_variables_initializer()
#
#
#
#
#
# dt=pd.read_csv(r"C:\Users\tongqing\PycharmProjects\untitled\tesnsorflow_test\train_disturb\data\train.csv")
# train_x=np.array(dt[["x1","x2"]]).reshape((-1,2))
# train_y=np.array(dt["y"]).reshape((-1,1))
#
#
# with tf.Session() as sess:
#     sess.run(init)
#     print("sess ok")
#     print(glob_step.eval(session=sess))
#     for _x,_y in zip(train_x,train_y):
#         print(np.array(_x).reshape((-1,2)),np.array(_y).reshape((-1,1)))
#         _x=np.array(_x).reshape((-1,2))
#         _y=np.array(_y).reshape((-1,1))
#         _,epoch=sess.run([opt,glob_step],feed_dict={x:_x,y:_y})
#         #生成summary文件
#         # summary_str=sess.run(merged_summary_op,feed_dict={x:_x,y:_y})
#         # sv.summary_computed(sess,summary_str,global_step=epoch)
#         if epoch%2==0:
#             loss1,loss=sess.run([cost1,cost],feed_dict={x:train_x,y:train_y})
#             print("Epoch:",epoch+1,"cost1=",loss1,"loss=",loss,"W=",sess.run(W),"b=",sess.run(b))
#
#
#
#
#

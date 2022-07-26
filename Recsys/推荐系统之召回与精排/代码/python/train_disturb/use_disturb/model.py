import tensorflow as tf
import conf



class model():
    def __init__(self,feature_size=2,label_size=1,decay=0.1):

        with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d"%conf.task_index,cluster= conf.cluster_spec)):
            x=tf.placeholder("float",[None,feature_size])
            y=tf.placeholder("float",[None,label_size])
            with tf.variable_scope("weight",regularizer=tf.contrib.layers.l2_regularizer(decay)) as scope:
                W = tf.get_variable("weight", [feature_size,label_size], initializer=tf.truncated_normal_initializer(decay))
                b = tf.Variable(tf.zeros([1]), name='bais')

            glob_step=tf.train.get_or_create_global_step()#获得迭代次数
            z=tf.matmul(x,W)+b
            tf.summary.histogram('z',z)
            cost=tf.losses.mean_squared_error(y,z)
            l2=tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            cost=cost+l2
            tf.summary.scalar('loss_function',cost)
            lr=0.01
            opt=tf.train.AdagradOptimizer(lr).minimize(cost,global_step=glob_step)
            saver=tf.train.Saver(max_to_keep=1)
            merged_summary_op=tf.summary.merge_all()
            init=tf.global_variables_initializer()
def train():

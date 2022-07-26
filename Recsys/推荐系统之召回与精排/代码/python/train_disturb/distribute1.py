import tensorflow as tf
import numpy as np

#################################################################################1.为每个角色添加ip端口，创建server
#定义ip和端口




strps_hosts="localhost:1681"
strwork_hosts="localhost:1682,localhost:1683"

strps_jobname='ps'
task_index=0
#将字符串转为数组

ps_hosts=strps_hosts.split(",")
worker_hosts=strwork_hosts.split(",")

cluster_spec=tf.train.ClusterSpec({'ps':ps_hosts,"worker":worker_hosts})
server=tf.train.Server({'ps':ps_hosts,"worker":worker_hosts},
                       job_name=strps_jobname,
                       task_index=task_index
                       )
#################################################################################2.为ps角色添加等待函数
if strps_jobname=='ps':
    print("wait")
    server.join()
#################################################################################3.创建网络结构
with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d"%task_index,cluster= cluster_spec)):
    x=tf.placeholder("float",[None,2])
    y=tf.placeholder("float",[None,1])
    W = tf.get_variable("weight", [2,1], initializer=tf.truncated_normal_initializer(0.1))
    b = tf.Variable(tf.zeros([1]), name='bais')

    glob_step=tf.train.get_or_create_global_step()#获得迭代次数
    z=tf.matmul(x,W)+b
    tf.summary.histogram('z',z)
    cost=tf.losses.mean_squared_error(y,z)
    tf.summary.scalar('loss_function',cost)
    lr=0.01
    opt=tf.train.AdagradOptimizer(lr).minimize(cost,global_step=glob_step)
    saver=tf.train.Saver(max_to_keep=1)
    merged_summary_op=tf.summary.merge_all()
    init=tf.global_variables_initializer()

#################################################################################4.创建Supervisor，管理session
train_epoch=2200
display_step=2

sv=tf.train.Supervisor(is_chief=(task_index==0),
                       logdir=r'E:\class\badou\day04\train_disturb\log\super',
                       init_op=init,
                       summary_op=None,
                       saver=saver,
                       global_step=glob_step,
                       save_model_secs=5
                       )
######迭代训练
#读取数据
import pandas as pd
import numpy as np
dt=pd.read_csv(r"E:\class\badou\day04\train_disturb\data\train.csv")

train_x=np.array(dt[["x1","x2"]]).reshape((-1,2))
train_y=np.array(dt["y"]).reshape((-1,1))


with sv.managed_session(server.target) as sess:
    print("sess ok")
    print(glob_step.eval(session=sess))
    for _x,_y in zip(train_x,train_y):
        _x = np.array(_x).reshape((-1, 2))
        _y = np.array(_y).reshape((-1, 1))
        _,epoch=sess.run([opt,glob_step],feed_dict={x:_x,y:_y})
        #生成summary文件
        summary_str=sess.run(merged_summary_op,feed_dict={x:_x,y:_y})
        sv.summary_computed(sess,summary_str,global_step=epoch)
        if epoch%display_step==0:
            loss=sess.run(cost,feed_dict={x:train_x,y:train_y})
            print("Epoch:",epoch+1,"cost=",loss,"W=",sess.run(W),"b=",sess.run(b))







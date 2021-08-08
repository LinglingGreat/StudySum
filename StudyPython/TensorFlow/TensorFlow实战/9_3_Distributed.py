# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Distributed MNIST training and validation, with model replicas.

A simple softmax model with one hidden layer is defined. The parameters
(weights and biases) are located on two parameter servers (ps), while the
ops are defined on a worker node. The TF sessions also run on the worker
node.
Multiple invocations of this script can be done in parallel, with different
values for --task_index. There should be exactly one invocation with
--task_index, which will create a master session that carries out variable
initialization. The other, non-master, sessions will wait for the master
session to finish the initialization before proceeding to the training stage.

The coordination between the multiple worker invocations occurs due to
the definition of the parameters on the same ps devices. The parameter updates
from one worker is visible to all other workers. As such, the workers can
perform forward computation and gradient calculation in parallel, which
should lead to increased training speed for the simple model.
"""

# 需要在3台不同的机器上分别执行，启动3个task，每次执行时需要传入job_name和task_index指定worker的身份
# 在三台机器上分别启动了一个parameter server及两个worker
# python distributed.py --job_name=ps --task_index=0
# python distributed.py --job_name=worker --task_index=0
# python distributed.py --job_name=worker --task_index=1

#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function

import math
#import sys
import tempfile
import time

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


flags = tf.app.flags
flags.DEFINE_string("data_dir", "/tmp/mnist-data",
                    "Directory for storing mnist data")
#flags.DEFINE_boolean("download_only", False,
#                     "Only perform downloading of data; Do not proceed to "
#                     "session preparation, model definition or training")
flags.DEFINE_integer("task_index", None,
                     "Worker task index, should be >= 0. task_index=0 is "
                     "the master worker task the performs the variable "
                     "initialization ")
#flags.DEFINE_integer("num_gpus", 2,
#                     "Total number of gpus for each machine."
#                     "If you don't use GPU, please set it to '0'")
# 这个参数代表进行同步并行时，一共积攒多少个batch的梯度才进行一次参数更新
# 设为None则使用worker的数量，即所有worker都完成一个batch的训练后再更新模型
flags.DEFINE_integer("replicas_to_aggregate", None,
                     "Number of replicas to aggregate before parameter update"
                     "is applied (For sync_replicas mode only; default: "
                     "num_workers)")
flags.DEFINE_integer("hidden_units", 100,
                     "Number of units in the hidden layer of the NN")
flags.DEFINE_integer("train_steps", 1000000,
                     "Number of (global) training steps to perform")
flags.DEFINE_integer("batch_size", 100, "Training batch size")
flags.DEFINE_float("learning_rate", 0.01, "Learning rate")
# 是否使用同步并行
flags.DEFINE_boolean("sync_replicas", False,
                     "Use the sync_replicas (synchronized replicas) mode, "
                     "wherein the parameter updates from workers are aggregated "
                     "before applied to avoid stale gradients")
#flags.DEFINE_boolean(
#    "existing_servers", False, "Whether servers already exists. If True, "
#    "will use the worker hosts via their GRPC URLs (one client process "
#    "per worker host). Otherwise, will create an in-process TensorFlow "
#    "server.")
# 定义parameter server，worker的地址
flags.DEFINE_string("ps_hosts","192.168.233.201:2222",
                    "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("worker_hosts", "192.168.233.202:2223,192.168.233.203:2224",
                    "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("job_name", None,"job name: worker or ps")

FLAGS = flags.FLAGS


IMAGE_PIXELS = 28


def main(unused_argv):
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
#  if FLAGS.download_only:
#    sys.exit(0)

  if FLAGS.job_name is None or FLAGS.job_name == "":
    raise ValueError("Must specify an explicit `job_name`")
  if FLAGS.task_index is None or FLAGS.task_index =="":
    raise ValueError("Must specify an explicit `task_index`")

  print("job name = %s" % FLAGS.job_name)
  print("task index = %d" % FLAGS.task_index)

  #Construct the cluster and start the server
  ps_spec = FLAGS.ps_hosts.split(",")
  worker_spec = FLAGS.worker_hosts.split(",")

  # Get the number of workers.
  num_workers = len(worker_spec)

  # 生成一个TensorFlow Cluster的对象，传入的参数是ps的地址信息和worker的地址信息
  cluster = tf.train.ClusterSpec({
      "ps": ps_spec,
      "worker": worker_spec})

  #if not FLAGS.existing_servers:
    # Not using existing servers. Create an in-process server.
  # 创建当前机器的server,用以连接到Cluster
  server = tf.train.Server(
      cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
  # 如果当前节点是parameter server,则不再进行后续的操作，而是使用server.join等待worker工作
  if FLAGS.job_name == "ps":
    server.join()

  # 判断当前机器是否是主节点
  is_chief = (FLAGS.task_index == 0)
  
#  if FLAGS.num_gpus > 0:
#    if FLAGS.num_gpus < num_workers:
#      raise ValueError("number of gpus is less than number of workers")
#    # Avoid gpu allocation conflict: now allocate task_num -> #gpu 
#    # for each worker in the corresponding machine
#    gpu = (FLAGS.task_index % FLAGS.num_gpus)
#    worker_device = "/job:worker/task:%d/gpu:%d" % (FLAGS.task_index, gpu)
#  elif FLAGS.num_gpus == 0:
#    # Just allocate the CPU to worker server
#    cpu = 0
#    worker_device = "/job:worker/task:%d/cpu:%d" % (FLAGS.task_index, cpu)
#  # The device setter will automatically place Variables ops on separate
#  # parameter servers (ps). The non-Variable ops will be placed on the workers.
#  # The ps use CPU and workers use corresponding GPU
  
  worker_device = "/job:worker/task:%d/gpu:0" % FLAGS.task_index
  # 设置worker的资源，worker_device为计算资源,ps_device为存储模型参数的资源
  # 通过replica_device_setter将模型参数部署在独立的ps服务器"/job:ps/cpu:0"，并将训练操作部署在"/job:worker/task:0/gpu:0"即本机的GPU
  with tf.device(
      tf.train.replica_device_setter(
          worker_device=worker_device,
          ps_device="/job:ps/cpu:0",
          cluster=cluster)):
    global_step = tf.Variable(0, name="global_step", trainable=False)

    # Variables of the hidden layer
    hid_w = tf.Variable(
        tf.truncated_normal(
            [IMAGE_PIXELS * IMAGE_PIXELS, FLAGS.hidden_units],
            stddev=1.0 / IMAGE_PIXELS),
        name="hid_w")
    hid_b = tf.Variable(tf.zeros([FLAGS.hidden_units]), name="hid_b")

    # Variables of the softmax layer
    sm_w = tf.Variable(
        tf.truncated_normal(
            [FLAGS.hidden_units, 10],
            stddev=1.0 / math.sqrt(FLAGS.hidden_units)),
        name="sm_w")
    sm_b = tf.Variable(tf.zeros([10]), name="sm_b")

    # Ops: located on the worker specified with FLAGS.task_index
    x = tf.placeholder(tf.float32, [None, IMAGE_PIXELS * IMAGE_PIXELS])
    y_ = tf.placeholder(tf.float32, [None, 10])

    hid_lin = tf.nn.xw_plus_b(x, hid_w, hid_b)
    hid = tf.nn.relu(hid_lin)

    y = tf.nn.softmax(tf.nn.xw_plus_b(hid, sm_w, sm_b))
    cross_entropy = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))

    opt = tf.train.AdamOptimizer(FLAGS.learning_rate)

    if FLAGS.sync_replicas:
      if FLAGS.replicas_to_aggregate is None:
        replicas_to_aggregate = num_workers
      else:
        replicas_to_aggregate = FLAGS.replicas_to_aggregate

      # 创建同步训练的优化器，会将原始优化器改造为同步的分布式训练版本
      opt = tf.train.SyncReplicasOptimizer(
          opt,
          replicas_to_aggregate=replicas_to_aggregate,
          total_num_replicas=num_workers,
          replica_id=FLAGS.task_index,
          name="mnist_sync_replicas")

    train_step = opt.minimize(cross_entropy, global_step=global_step)

    if FLAGS.sync_replicas and is_chief:
      # Initial token and chief queue runners required by the sync_replicas mode
      # 创建队列执行器，并创建全局参数初始化器
      chief_queue_runner = opt.get_chief_queue_runner()
      init_tokens_op = opt.get_init_tokens_op()

    init_op = tf.global_variables_initializer()
    train_dir = tempfile.mkdtemp()    # 创建临时的训练目录
    # 创建分布式训练的监督器，会管理我们的task参与到分布式训练
    sv = tf.train.Supervisor(
        is_chief=is_chief,
        logdir=train_dir,
        init_op=init_op,
        recovery_wait_secs=1,
        global_step=global_step)

    # 设置Session的参数，allow_soft_placement=True代表党某个操作在指定的device不能执行时，可以转到其他device执行
    sess_config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False,
        device_filters=["/job:ps", "/job:worker/task:%d" % FLAGS.task_index])

    # The chief worker (task_index==0) session will prepare the session,
    # while the remaining workers will wait for the preparation to complete.
    if is_chief:
      print("Worker %d: Initializing session..." % FLAGS.task_index)
    else:
      print("Worker %d: Waiting for session to be initialized..." %
            FLAGS.task_index)

#    if FLAGS.existing_servers:
#      server_grpc_url = "grpc://" + worker_spec[FLAGS.task_index]
#      print("Using existing server at: %s" % server_grpc_url)
#
#      sess = sv.prepare_or_wait_for_session(server_grpc_url, config=sess_config)
#    else:
    # 若为主节点则会创建Session，若为分支节点则会等待
    sess = sv.prepare_or_wait_for_session(server.target,
                                            config=sess_config)

    print("Worker %d: Session initialization complete." % FLAGS.task_index)

    if FLAGS.sync_replicas and is_chief:
      # Chief worker will start the chief queue runner and call the init op
      print("Starting chief queue runner and running init_tokens_op")
      sv.start_queue_runners(sess, [chief_queue_runner])
      sess.run(init_tokens_op)

    # Perform training
    time_begin = time.time()
    print("Training begins @ %f" % time_begin)

    local_step = 0
    while True:
      # Training feed
      batch_xs, batch_ys = mnist.train.next_batch(FLAGS.batch_size)
      train_feed = {x: batch_xs, y_: batch_ys}

      _, step = sess.run([train_step, global_step], feed_dict=train_feed)
      local_step += 1

      now = time.time()
      print("%f: Worker %d: training step %d done (global step: %d)" %
            (now, FLAGS.task_index, local_step, step))

      if step >= FLAGS.train_steps:
        break

    time_end = time.time()
    print("Training ends @ %f" % time_end)
    training_time = time_end - time_begin
    print("Training elapsed time: %f s" % training_time)

    # Validation feed
    val_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
    val_xent = sess.run(cross_entropy, feed_dict=val_feed)
    print("After %d training step(s), validation cross entropy = %g" %
          (FLAGS.train_steps, val_xent))


if __name__ == "__main__":
  tf.app.run()

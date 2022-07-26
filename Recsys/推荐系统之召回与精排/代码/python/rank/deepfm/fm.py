#coding=utf-8
"""
Author:tongqing
data:2020/6/10 14:17
desc
"""
import pandas as pd
import numpy as np
import tensorflow as tf
from tqdm import tqdm_notebook as tqdm
from sklearn.preprocessing import   OneHotEncoder
from data_process import  load_rating,load_train_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.metrics import roc_auc_score
import  pandas as pd


def batcher(X_, y_=None, batch_size=-1):
    n_samples = X_.shape[0]

    if batch_size == -1:
        batch_size = n_samples
    if batch_size < 1:
       raise ValueError('Parameter batch_size={} is unsupported'.format(batch_size))

    for i in range(0, n_samples, batch_size):
        upper_bound = min(i + batch_size, n_samples)
        ret_x = X_[i:upper_bound]
        ret_y = None
        if y_ is not None:
            ret_y = y_[i:i + batch_size]
            yield (ret_x, ret_y)


class fm:
    def __init__(self,feature_nums,k_size=64):
        k = k_size
        self.feature_nums=feature_nums

        self.x = tf.placeholder('float',[None,feature_nums])
        self.y = tf.placeholder('float',[None,1])
        w0 = tf.Variable(tf.zeros([1]))
        w = tf.Variable(tf.zeros([p]))
        v = tf.Variable(tf.random_normal([k,p],mean=0,stddev=0.01))

        linear_terms = tf.add(w0,tf.reduce_sum(tf.multiply(w,self.x),1,keep_dims=True)) # n * 1

        pair_interactions = 0.5 * tf.reduce_sum(
            tf.subtract(
                tf.pow(
                    tf.matmul(self.x,tf.transpose(v)),2),
                tf.matmul(tf.pow(self.x,2),tf.transpose(tf.pow(v,2)))
            ),axis = 1 , keep_dims=True)


        y_hat = tf.add(linear_terms,pair_interactions)


        self.y_hat=tf.nn.sigmoid(y_hat)

        loss = tf.losses.sigmoid_cross_entropy(self.y,self.y_hat)

        lambda_w = tf.constant(0.001,name='lambda_w')
        lambda_v = tf.constant(0.001,name='lambda_v')
        l2_norm = tf.reduce_sum(
            tf.add(
                tf.multiply(lambda_w,tf.pow(w,2)),
                tf.multiply(lambda_v,tf.pow(v,2))
            )
        )
        self.loss=loss+l2_norm

        self.train_op = tf.train.AdamOptimizer(learning_rate=0.01).minimize(self.loss)

    def predict(self,X,sess):
        pre = []
        for bX in batcher(X,batch_size=32):
            s1=sess.run(self.y_hat, feed_dict={self.x: bX.reshape(-1,self.feature_nums )})

            pre.append(s1)
        return pre
if __name__ == '__main__':
    x_feature, y = load_train_data()
    enc = OneHotEncoder()
    x_feature = enc.fit_transform(x_feature).todense()

    # split train and test
    x_train, x_test, y_train, y_test = train_test_split(x_feature, y, test_size=0.33, random_state=42)
    n, p = x_train.shape

    epochs =30
    batch_size = 100

    with tf.Graph().as_default():
        fmModel=fm(feature_nums=p)
        # Launch the graph
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            for epoch in tqdm(range(epochs), unit='epoch'):
                perm = np.random.permutation(x_train.shape[0])
                # iterate over batches
                for bX, bY in batcher(x_train[perm], y_train[perm], batch_size):
                    _,t = sess.run([fmModel.train_op,fmModel.loss], feed_dict={fmModel.x: bX.reshape(-1, p), fmModel.y: bY.reshape(-1, 1)})
                    print(t)
            y_pre=[]
            for bX, bY in batcher(x_test, y_test):
                y_pre=sess.run(fmModel.y_hat,feed_dict={fmModel.x: bX.reshape(-1, p)})

            y_pre=np.array(y_pre)[:,0]
            result=pd.DataFrame({"pre":y_pre,"y_true":y_test})
            print(result.head(20))

            print(roc_auc_score(result["y_true"],result["pre"]))
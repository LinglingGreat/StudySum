import tensorflow as tf
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder


class model_test:

    def __init__(self,classes=3,con_featureNum=4,hidden_layyer=[16,32,16],lr=0.01):

        self.x=tf.placeholder(dtype=tf.float32,shape=[None,con_featureNum],name="input")
        self.y=tf.placeholder(dtype=tf.int32,shape=[None,classes])

        #mlp
        inputs=self.x
        with tf.variable_scope("mlp"):
            for unit in hidden_layyer:
                inputs=tf.layers.dense(inputs,unit,activation=tf.nn.relu)


        #softmax
        logits=tf.layers.dense(inputs,classes,activation=tf.nn.softmax)
        self.loss=tf.losses.softmax_cross_entropy(self.y,logits)

        self.opt=tf.train.AdagradOptimizer(learning_rate=lr).minimize(self.loss)


        self.pre=tf.argmax(logits,axis=-1,name="output")

        init = tf.global_variables_initializer()


def batch_data(x,y,batch_size=5):
    m=x.shape[0]
    for i in range(1,m,batch_size):
        start=i
        end=min(i+batch_size,m)
        yield x[start:end],y[start:end]






if __name__ == '__main__':
    data = load_iris()
    x = data.data
    y = data.target
    print(x,y)
    enc=OneHotEncoder()
    label_y=enc.fit_transform(y.reshape(-1,1)).todense()

    x=np.array(x)
    labels=np.array(label_y)

    model=model_test()

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for epoch in range(100):
            for bx,by in batch_data(x,labels):
                loss,_=sess.run([model.loss,model.opt],feed_dict={model.x:bx,model.y:by})

            print(loss)

        pre=sess.run(model.pre,feed_dict={model.x:x,model.y:labels})
        print(pre)

        graph = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, ["output"])
        tf.train.write_graph(graph, '.', 'rf.pb', as_text=False)
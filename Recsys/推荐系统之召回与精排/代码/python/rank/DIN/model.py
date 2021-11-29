import tensorflow as tf

from Dice import dice


class Model(object):
    def __init__(self,user_count,item_count,cate_count,cate_list):
        """

        :param user_count: 用户 id总数
        :param item_count: 物品 id总数
        :param cate_count: 品牌 id总数
        :param cate_list: 1*N,其中表示每个物品对应的牌子id
        """
        # self.u = tf.placeholder(tf.int32,[None,],name='user')
        #item i，j，是否点击y，历史物品列表hist_i,s1物品列表长度，lr学习率
        self.i = tf.placeholder(tf.int32,[None,],name='item')
        self.j = tf.placeholder(tf.int32,[None,],name='item_j')
        self.y = tf.placeholder(tf.float32,[None,],name='label')
        self.hist_i = tf.placeholder(tf.int32,[None,None],name='history_i')
        self.sl = tf.placeholder(tf.int32, [None,] , name='sequence_length')
        self.lr = tf.placeholder(tf.float64,name='learning_rate')
        #隐藏层32
        hidden_units = 32

        # user_emb_w = tf.get_variable("user_emb_w",[user_count,hidden_units])
        # N*16 item embedding
        item_emb_w = tf.get_variable("item_emb_w",[item_count,hidden_units//2])

        #embedding nge embedding
        item_b = tf.get_variable("item_b",[item_count],initializer=tf.constant_initializer(0.0))

        #品类embediing M*16

        cate_emb_w = tf.get_variable("cate_emb_w",[cate_count,hidden_units//2])
        #cate_list  [1，N]  [1,3,4,2,3,1,4,2,4,5,1,2,3,4]
        cate_list = tf.convert_to_tensor(cate_list,dtype=tf.int64)

        # u_emb = tf.nn.embedding_lookup(user_emb_w,self.u)


        # ic是item到category的转换,为了共享embedding。

        ###物品i的embedding
        self.ic = tf.gather(cate_list,self.i)
        #物品embedding和 品类embedding， N*32

        i_emb = tf.concat(values=[
            tf.nn.embedding_lookup(item_emb_w,self.i),
            tf.nn.embedding_lookup(cate_emb_w,self.ic)
        ],axis=1)


        i_b = tf.gather(item_b,self.i)


        ###物品j的embedding
        self.jc = tf.gather(cate_list, self.j)

        j_emb = tf.concat([
            tf.nn.embedding_lookup(item_emb_w, self.j),
            tf.nn.embedding_lookup(cate_emb_w, self.jc),
        ], axis=1)
        j_b = tf.gather(item_b, self.j)

        ###物品历史、品类历史embedding  【b*m*n】
        self.hc = tf.gather(cate_list, self.hist_i)
        h_emb = tf.concat([
            tf.nn.embedding_lookup(item_emb_w, self.hist_i),
            tf.nn.embedding_lookup(cate_emb_w, self.hc),
        ], axis=2)

        ###物品i和 历史 做attention   B*1*H
        hist = attention(i_emb,h_emb,self.sl)
        ##bn
        hist = tf.layers.batch_normalization(inputs=hist)
        ##B*H
        hist = tf.reshape(hist,[-1,hidden_units])
        ##一层全连接  B*H
        hist = tf.layers.dense(hist,hidden_units)
        ## user 结合hist的embeding
        u_emb = hist


        # fcn begin，将用户历史信息attention与 预推荐的物品i做contancat
        din_i = tf.concat([u_emb, i_emb], axis=-1)
        #加一层bn
        din_i = tf.layers.batch_normalization(inputs=din_i, name='b1')
        #全连接
        d_layer_1_i = tf.layers.dense(din_i, 80, activation=None, name='f1')
        #加一个dice激活
        d_layer_1_i = dice(d_layer_1_i, name='dice_1_i')
        #全连接
        d_layer_2_i = tf.layers.dense(d_layer_1_i, 40, activation=None, name='f2')
        #dice激活
        d_layer_2_i = dice(d_layer_2_i, name='dice_2_i')
        #输出 i  sigmod
        d_layer_3_i = tf.layers.dense(d_layer_2_i, 1, activation=None, name='f3')


        # attention与  物品j做contancat，但是attention（u_emb）是与i做的
        din_j = tf.concat([u_emb, j_emb], axis=-1)
        din_j = tf.layers.batch_normalization(inputs=din_j, name='b1', reuse=True)
        d_layer_1_j = tf.layers.dense(din_j, 80, activation=None, name='f1', reuse=True)
        d_layer_1_j = dice(d_layer_1_j, name='dice_1_j')
        d_layer_2_j = tf.layers.dense(d_layer_1_j, 40, activation=None, name='f2', reuse=True)
        d_layer_2_j = dice(d_layer_2_j, name='dice_2_j')
        # 输出 j  sigmod
        d_layer_3_j = tf.layers.dense(d_layer_2_j, 1, activation=None, name='f3', reuse=True)


        d_layer_3_i = tf.reshape(d_layer_3_i, [-1])
        d_layer_3_j = tf.reshape(d_layer_3_j, [-1])

        #此处按照原理来说d_layer_3_i+ib 要比 d_layer_3_j+jb 大
        x = i_b - j_b + d_layer_3_i - d_layer_3_j  # [B]
        #加了一个物品i的bais，
        self.logits = i_b + d_layer_3_i


        # logits for all item:      B*M  B*1*M
        u_emb_all = tf.expand_dims(u_emb, 1)
        #B*N*M 这只针对 物品i对历史物品的attention
        u_emb_all = tf.tile(u_emb_all, [1, item_count, 1])


        #concat([N*M,N*M])
        all_emb = tf.concat([
            item_emb_w,
            tf.nn.embedding_lookup(cate_emb_w, cate_list)
        ], axis=1)
        #1*N*M
        all_emb = tf.expand_dims(all_emb, 0)
        #512*N*M
        all_emb = tf.tile(all_emb, [512, 1, 1])
        ##512*N*M   512*N*M
        din_all = tf.concat([u_emb_all, all_emb], axis=-1)
        din_all = tf.layers.batch_normalization(inputs=din_all, name='b1', reuse=True)
        d_layer_1_all = tf.layers.dense(din_all, 80, activation=None, name='f1', reuse=True)
        d_layer_1_all = dice(d_layer_1_all, name='dice_1_all')
        d_layer_2_all = tf.layers.dense(d_layer_1_all, 40, activation=None, name='f2', reuse=True)
        d_layer_2_all = dice(d_layer_2_all, name='dice_2_all')
        d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name='f3', reuse=True)
        #512*N
        d_layer_3_all = tf.reshape(d_layer_3_all, [-1, item_count])


        self.logits_all = tf.sigmoid(item_b + d_layer_3_all)
        # -- fcn end -------

        self.mf_auc = tf.reduce_mean(tf.to_float(x > 0))
        self.score_i = tf.sigmoid(i_b + d_layer_3_i)
        self.score_j = tf.sigmoid(j_b + d_layer_3_j)
        self.score_i = tf.reshape(self.score_i, [-1, 1])
        self.score_j = tf.reshape(self.score_j, [-1, 1])
        self.p_and_n = tf.concat([self.score_i, self.score_j], axis=-1)


        # Step variable
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.global_epoch_step = \
            tf.Variable(0, trainable=False, name='global_epoch_step')
        self.global_epoch_step_op = \
            tf.assign(self.global_epoch_step, self.global_epoch_step + 1)

        # loss and train
        self.loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.logits,
                labels=self.y)
        )

        trainable_params = tf.trainable_variables()
        self.train_op = tf.train.GradientDescentOptimizer(learning_rate=self.lr).minimize(self.loss)

    def train(self,sess,uij,l):
        loss,_ = sess.run([self.loss,self.train_op],feed_dict={
            #self.u : uij[0],
            self.i : uij[1],
            self.y : uij[2],
            self.hist_i : uij[3],
            self.sl : uij[4],
            self.lr : l
        })

        return loss

    def eval(self, sess, uij):
        u_auc, socre_p_and_n = sess.run([self.mf_auc, self.p_and_n], feed_dict={
            #self.u: uij[0],
            self.i: uij[1],
            self.j: uij[2],
            self.hist_i: uij[3],
            self.sl: uij[4],
        })
        return u_auc, socre_p_and_n

    def test(self, sess, uid, hist_i, sl):
        return sess.run(self.logits_all, feed_dict={
            self.u: uid,
            self.hist_i: hist_i,
            self.sl: sl,
        })

    def save(self, sess, path):
        saver = tf.train.Saver()
        saver.save(sess, save_path=path)

    def restore(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, save_path=path)


def extract_axis_1(data, ind):
    batch_range = tf.range(tf.shape(data)[0])
    indices = tf.stack([batch_range, ind], axis=1)
    res = tf.gather_nd(data, indices)
    return res


def attention(queries,keys,keys_length):
    '''
        queries:     [B, H]
        keys:        [B, T, H]
        keys_length: [B]
    '''

    queries_hidden_units = queries.get_shape().as_list()[-1]
    #第一个位置复制T列  [B,T*H]
    queries = tf.tile(queries,[1,tf.shape(keys)[1]])
    #reshape 一下
    queries = tf.reshape(queries,[-1,tf.shape(keys)[1],queries_hidden_units])
    #B*T*4H
    din_all=tf.concat([queries,keys,queries-keys,queries*keys],axis=-1)
    # 三层全链接
    d_layer_1_all=tf.layers.dense(din_all,80,activation=tf.nn.sigmoid,name='f1_att')
    d_layer_2_all = tf.layers.dense(d_layer_1_all, 40, activation=tf.nn.sigmoid, name='f2_att')
    #B*T*1
    d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name='f3_att') #B*T*1

    #转置
    outputs = tf.reshape(d_layer_3_all,[-1,1,tf.shape(keys)[1]]) #B*1*T

    # Mask B*T里面不满足长度L的都被设置为false
    key_masks = tf.sequence_mask(keys_length,tf.shape(keys)[1])
    #维度扩展  B*1*T
    key_masks = tf.expand_dims(key_masks,1) # B*1*T
    #padding
    paddings=tf.ones_like(outputs)*(-2*32+1) # 在补足的地方附上一个很小的值，而不是0

    outputs = tf.where(key_masks,outputs,paddings) # B * 1 * T

    # Scale outputs /sqrt(H) B*1*T
    outputs = outputs / (keys.get_shape().as_list()[-1] ** 0.5)

    # Activation,B*1*T
    outputs = tf.nn.softmax(outputs) # B * 1 * T

    # Weighted Sum B*1*T   B*T*H
    outputs = tf.matmul(outputs,keys) # B * 1 * H 三维矩阵相乘，相乘发生在后两维，即 B * (( 1 * T ) * ( T * H ))

    return outputs








# K.expand_dims

expand_dims(x, dim=-1)：在下标为`dim`的轴上增加一维

例子：

```python
from keras import backend as K
import numpy as np
 
x=np.array([[2,3],[2,3],[2,3]])
print("x shape:",x.shape)   #(3, 2)
y1=K.expand_dims(x, 0)
y2=K.expand_dims(x, 1)
y3=K.expand_dims(x, 2)
y4=K.expand_dims(x, -1)
 
print("y1 shape:",y1.shape)  #(1,3,2)
print("y2 shape:",y2.shape)  #(3,1,2)
print("y3 shape:",y3.shape)  #(3,2,1)
print("y4 shape:",y4.shape)  #(3,2,1)
```

# K.tile

tile(x, n)：将x在各个维度上重复n次，x为张量，n为与x维度数目相同的列表

例如K.tile(initial_state, [1, 64])

initial_state 的shape=（？，1）

最终 K.tile(initial_state, [1, 64]) 的shape=（？，64）

```python
from keras import backend as K
import numpy as np
 
x=np.array([[1,2],[3,4],[5,6]])
a=K.tile(x, [1, 2])
b=K.tile(x, [2, 2])
 
print(x,x.shape)  #(3,2)
print(a,a.shape)  #(3,4)
print(b,b.shape)  #(6, 4)
```

# Lambda 层

keras.layers.core.Lambda(function, output_shape=None, mask=None, arguments=None)

本函数用以对上一层的输出施以任何Theano/TensorFlow表达式。

如果你只是想对流经该层的数据做个变换，而这个变换本身没有什么需要学习的参数，那么直接用Lambda Layer是最合适的了。

Lambda函数接受两个参数，第一个是输入张量对输出张量的映射函数，第二个是输入的shape对输出的shape的映射函数。

**参数**

- function：要实现的函数，该函数仅接受一个变量，即上一层的输出
- output_shape：函数应该返回的值的shape，可以是一个tuple，也可以是一个根据输入shape计算输出shape的函数
- mask: 
- arguments：可选，字典，用来记录向函数中传递的其他关键字参数

例子

```python
# add a x -> x^2 layer
model.add(Lambda(lambda x: x ** 2))
# add a layer that returns the concatenation# of the positive part of the input and
# the opposite of the negative part
 
def antirectifier(x):
    x -= K.mean(x, axis=1, keepdims=True)
    x = K.l2_normalize(x, axis=1)
    pos = K.relu(x)
    neg = K.relu(-x)
    return K.concatenate([pos, neg], axis=1)
 
def antirectifier_output_shape(input_shape):
    shape = list(input_shape)
    assert len(shape) == 2  # only valid for 2D tensors
    shape[-1] *= 2
    return tuple(shape)
 
model.add(Lambda(antirectifier,
         output_shape=antirectifier_output_shape)
```

输入shape：任意，当使用该层作为第一层时，要指定input_shape

输出shape：由output_shape参数指定的输出shape，当使用tensorflow时可自动推断

例子2：**将矩阵的每一列提取出来,然后单独进行操作,最后在拼在一起。**

```python
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation,Reshape
from keras.layers import merge
from keras.utils.visualize_util import plot
from keras.layers import Input, Lambda
from keras.models import Model
 
def slice(x,index):
　　return x[:,:,index]
 
a = Input(shape=(4,2))
x1 = Lambda(slice,output_shape=(4,1),arguments={‘index‘:0})(a)
x2 = Lambda(slice,output_shape=(4,1),arguments={‘index‘:1})(a)
x1 = Reshape((4,1,1))(x1)
x2 = Reshape((4,1,1))(x2)
output = merge([x1,x2],mode=‘concat‘)
 
model = Model(a, output)
x_test = np.array([[[1,2],[2,3],[3,4],[4,5]]])
print model.predict(x_test)
plot(model, to_file=‘lambda.png‘,show_shapes=True)
```

# categorical_crossentropy 和 sparse_categorical_crossentropy

在 tf.keras 中，有两个交叉熵相关的损失函数 tf.keras.losses.categorical_crossentropy 和 tf.keras.losses.sparse_categorical_crossentropy 。其中 sparse 的含义是，真实的标签值 y_true 可以直接传入 int 类型的标签类别，即sparse时 y 不需要one-hot，而 categorical 需要。

```python
loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=y_pred)

loss = tf.keras.losses.categorical_crossentropy(
    y_true=tf.one_hot(y, depth=tf.shape(y_pred)[-1]),
    y_pred=y_pred
)

```

# tf.gather tf.gather_nd 和 tf.batch_gather

在计算机视觉的项目中，会遇到根据索引获取数组特定下标元素的任务。常用的函数有tf.gather , tf.gather_nd 和 tf.batch_gather。

1.`tf.gather(params,indices,validate_indices=None,name=None,axis=0)`

主要参数：

- params：被索引的张量
- indices：一维索引张量
- name：返回张量名称
- 返回值：通过indices获取params下标的张量。

例子：

```python
import tensorflow as tf
tensor_a = tf.Variable([[1,2,3],[4,5,6],[7,8,9]])
tensor_b = tf.Variable([1,2,0],dtype=tf.int32)
tensor_c = tf.Variable([0,0],dtype=tf.int32)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(tf.gather(tensor_a,tensor_b)))
    print(sess.run(tf.gather(tensor_a,tensor_c)))
```

上个例子tf.gather(tensor_a,tensor_b) 的值为[[4,5,6],[7,8,9],[1,2,3]],tf.gather(tensor_a,tensor_b) 的值为[[1,2,3],[1,2,3]]

对于tensor_a,其第1个元素为[4,5,6]，第2个元素为[7,8,9],第0个元素为[1,2,3],所以以[1,2,0]为索引的返回值是[[4,5,6],[7,8,9],[1,2,3]]，同样的，以[0,0]为索引的值为[[1,2,3],[1,2,3]]。

https://www.tensorflow.org/api_docs/python/tf/gather

 

2.`tf.gather_nd(params,indices,name=None)`

功能和参数与tf.gather类似，不同之处在于tf.gather_nd支持多维度索引。

例子：

```python
import tensorflow as tf
tensor_a = tf.Variable([[1,2,3],[4,5,6],[7,8,9]])
tensor_b = tf.Variable([[1,0],[1,1],[1,2]],dtype=tf.int32)
tensor_c = tf.Variable([[0,2],[2,0]],dtype=tf.int32)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(tf.gather_nd(tensor_a,tensor_b)))
    print(sess.run(tf.gather_nd(tensor_a,tensor_c)))
```




tf.gather_nd(tensor_a,tensor_b)值为[4,5,6],tf.gather_nd(tensor_a,tensor_c)的值为[3,7].

对于tensor_a,下标[1,0]的元素为4,下标为[1,1]的元素为5,下标为[1,2]的元素为6,索引[1,0],[1,1],[1,2]]的返回值为[4,5,6],同样的，索引[[0,2],[2,0]]的返回值为[3,7].

https://www.tensorflow.org/api_docs/python/tf/gather_nd

3.`tf.batch_gather(params,indices,name=None)`

支持对张量的批量索引，各参数意义见（1）中描述。注意因为是批处理，所以indices要有和params相同的第0个维度。

例子：

```python
import tensorflow as tf
tensor_a = tf.Variable([[1,2,3],[4,5,6],[7,8,9]])
tensor_b = tf.Variable([[0],[1],[2]],dtype=tf.int32)
tensor_c = tf.Variable([[0],[0],[0]],dtype=tf.int32)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(tf.batch_gather(tensor_a,tensor_b)))
    print(sess.run(tf.batch_gather(tensor_a,tensor_c)))
```



tf.gather_nd(tensor_a,tensor_b)值为[1,5,9],tf.gather_nd(tensor_a,tensor_c)的值为[1,4,7].

tensor_a的三个元素[1,2,3],[4,5,6],[7,8,9]分别对应索引元素的第一，第二和第三个值。[1,2,3]的第0个元素为1,[4,5,6]的第1个元素为5,[7,8,9]的第2个元素为9,所以索引[[0],[1],[2]]的返回值为[1,5,9],同样地，索引[[0],[0],[0]]的返回值为[1,4,7].

https://www.tensorflow.org/api_docs/python/tf/batch_gather

 在深度学习的模型训练中，有时候需要对一个batch的数据进行类似于tf.gather_nd的操作，但tensorflow中并没有tf.batch_gather_nd之类的操作，此时需要tf.map_fn和tf.gather_nd结合来实现上述操作。

# 参考资料

[K.expand_dims & K.tile](https://blog.csdn.net/u014769320/article/details/99696898)

[tf.gather tf.gather_nd 和 tf.batch_gather 使用方法](https://blog.csdn.net/zby1001/article/details/86551667)
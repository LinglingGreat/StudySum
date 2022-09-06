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

## Keras的fit,fit_generator和train_on_batch函数

### Keras.fit函数

调用：`model.fit(trainX, trainY, batch_size=32, epochs=50)`

对.fit的调用在这里做出两个主要假设：

-   我们的整个训练集可以放入RAM
    
-   没有数据增强（即不需要Keras生成器）
    

相反，我们的网络将在原始数据上训练。原始数据本身将适合内存，我们无需将旧批量数据从RAM中移出并将新批量数据移入RAM。此外，我们不会使用数据增强动态操纵训练数据。

### Keras fit_generator函数

真实世界的数据集通常太大而无法放入内存中，它们也往往具有挑战性，要求我们执行数据增强以避免过拟合并增加我们的模型的泛化能力。

调用示例

```python
from keras.preprocessing.image import ImageDataGenerator
# initialize the number of epochs and batch size
EPOCHS = 100
BS = 32

# construct the training image generator for data augmentation
# aug是一个Keras ImageDataGenerator对象，用于图像的数据增强，随机平移，旋转，调整大小等。
aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
  width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
  horizontal_flip=True, fill_mode="nearest")

# train the network
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
  validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
  epochs=EPOCHS)

```

执行数据增强是正则化的一种形式，使我们的模型能够更好的被泛化。但是，应用数据增强意味着我们的训练数据不再是“静态的” ——数据不断变化。根据提供给ImageDataGenerator的参数随机调整每批新数据。因此，我们现在需要利用Keras的.fit_generator函数来训练我们的模型。

顾名思义，.fit_generator函数假定存在一个为其生成数据的基础函数。该函数本身是一个Python生成器。

Keras在使用.fit_generator训练模型时的过程：

-   Keras调用提供给.fit_generator的生成器函数（在本例中为aug.flow）
    
-   生成器函数为.fit_generator函数生成一批大小为BS的数据
    
-   .fit_generator函数接受批量数据，执行反向传播，并更新模型中的权重
    
-   重复该过程直到达到期望的epoch数量
    

您会注意到我们现在需要在调用.fit_generator时提供steps_per_epoch参数（.fit方法没有这样的参数）。

为什么我们需要steps_per_epoch？请记住，Keras数据生成器意味着无限循环，它永远不会返回或退出。由于该函数旨在无限循环，因此Keras无法确定一个epoch何时开始的，并且新的epoch何时开始。因此，我们将训练数据的总数除以批量大小的结果作为steps_per_epoch的值。一旦Keras到达这一步，它就会知道这是一个新的epoch。

### Keras train_on_batch函数

对于寻求对Keras模型进行精细控制（ finest-grained control）的深度学习实践者，您可能希望使用.train_on_batch函数：`model.train_on_batch(batchX, batchY)`.

train_on_batch函数接受单批数据，执行反向传播，然后更新模型参数。该批数据可以是任意大小的（即，它不需要提供明确的批量大小）。

您也可以生成数据。此数据可以是磁盘上的原始图像，也可以是以某种方式修改或扩充的数据。

当您有非常明确的理由想要维护自己的训练数据迭代器时，通常会使用.train_on_batch函数，例如数据迭代过程非常复杂并且需要自定义代码。

如果你发现自己在询问是否需要.train_on_batch函数，那么很有可能你可能不需要。

在99％的情况下，您不需要对训练深度学习模型进行如此精细的控制。相反，您可能只需要自定义Keras .fit_generator函数。

也就是说，如果你需要它，知道存在这个函数是很好的。

如果您是一名高级深度学习从业者/工程师，并且您确切知道自己在做什么以及为什么这样做，我通常只建议使用.train_on_batch函数。

### callback函数

[keras自定义回调函数查看训练的loss和accuracy](https://blog.csdn.net/qq_27825451/article/details/93377801)

## 保存模型

只保存权重`model.save_weights(file)`，加载用`model.load_weights(file)`

保存模型`model.save(file)`，加载用`model.load_model(file)`

load_model代码包含load_weights的代码，区别在于load_weights时需要先有网络、并且load_weights需要将权重数据写入到对应网络层的tensor中。

```python
from keras.models import Sequential
from keras.layers import Dense, Activation


def build_model():
    model = Sequential()

    model.add(Dense(output_dim=64, input_dim=100))
    model.add(Activation("relu"))
    model.add(Dense(output_dim=10))
    model.add(Activation("softmax"))

    # you can either compile or not the model
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    return model


model1 = build_model()
model1.save_weights('my_weights.h5')


model2 = build_model()
model2.load_weights('my_weights.h5')

# do stuff with model2 (e.g. predict())
```

保存整个模型

```python
from keras.models import load_model

model1 = build_model()
model1.save('my_model.h5')

model2 = load_model('my_model.h5')
# do stuff with model2 (e.g. predict()
```

如果是用gpu训练并保存的，那么只能用gpu加载和预测。可以用以下方法重新保存模型，能够在cpu上预测

```python
model = Model(
        [inp_token_ids, inp_segment_ids, inp_header_ids, inp_header_mask],
        [p_cond_conn_op, p_sel_agg, p_cond_op]
    )
 
 # 重新加载gpu模型并保存为cpu模型
 model_gpu = multi_gpu_model(model, gpus=NUM_GPUS)
 # 测试
 model_gpu.load_weights(model_path)
 model.save("single_"+model_path)   # 保存
  
 model.load_weights("single_"+model_path)   # cpu上预测
 # 或者用这个，速度更快
 model = load_model("single_"+model_path, custom_objects=get_custom_objects())   # cpu上预测 
```

## tensorflow gpu环境安装

安装命令在：[https://ngc.nvidia.com/catalog/containers/nvidia:tensorflow](https://ngc.nvidia.com/catalog/containers/nvidia:tensorflow)

注意cuda版本和tf版本：[https://docs.nvidia.com/deeplearning/frameworks/tensorflow-release-notes/rel_21-12.html#rel_21-12](https://docs.nvidia.com/deeplearning/frameworks/tensorflow-release-notes/rel_21-12.html#rel_21-12)

注意宿主机的cuda版本要比docker的cuda版本高（最好高好几个，比如宿主机是cuda11.3，docker可以安装11.0）

**宿主机（物理主机）的Nvidia GPU 驱动 必须大于 CUDA Toolkit要求的 Nvida GPU 驱动版本。**

**即使Docker的CUDA和主机无关**，但是**Docker和宿主机的驱动有关**

参考资料

-   [CUDA兼容性问题（显卡驱动、docker内CUDA）](https://zhuanlan.zhihu.com/p/459431437)
-   [RTX3090运行Tensorflow1.15（CUDA 11.1） Docker、TF1.15测试环境](https://zhuanlan.zhihu.com/p/341969571)
-   [RTX3080+Ubuntu18.04+cuda11.1+cudnn8.0.4+TensorFlow1.15.4+PyTorch1.7.0环境配置](https://blog.csdn.net/wu496963386/article/details/109583045)


# bert4keras

tf 2.x下使用

`pip install git+https://www.github.com/bojone/bert4keras.git`

代码里

```python
import os
os.environ['TF_KERAS'] = '1'

# 把from bert4keras.backend import keras, K改成
from tensorflow import keras
import tensorflow.keras.backend as K



```

试过tf2.3.3可以

# 参考资料

[https://stackoverflow.com/questions/41859997/keras-model-load-weights-for-neural-net](https://stackoverflow.com/questions/41859997/keras-model-load-weights-for-neural-net)

[keras读取h5文件load_weights、load代码详解](https://blog.csdn.net/wanggao_1990/article/details/90446736)

[如何使用Keras fit和fit_generator（动手教程）](https://blog.csdn.net/learning_tortosie/article/details/85243310)

[keras 多GPU训练，单GPU权重保存和预测](https://blog.csdn.net/m0_37477175/article/details/83378464)

[K.expand_dims & K.tile](https://blog.csdn.net/u014769320/article/details/99696898)

[tf.gather tf.gather_nd 和 tf.batch_gather 使用方法](https://blog.csdn.net/zby1001/article/details/86551667)
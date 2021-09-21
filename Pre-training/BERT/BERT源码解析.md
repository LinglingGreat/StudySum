Google提供的BERT代码在[这里](https://github.com/google-research/bert)，我们可以直接git clone下来。注意运行它需要Tensorflow 1.11及其以上的版本，低版本的Tensorflow不能运行。

由于从头开始(from scratch)训练需要巨大的计算资源，因此Google提供了预训练的模型(的checkpoint)，目前包括英语、汉语和多语言3类模型，而英语又包括4个版本：

- BERT-Base, Uncased 12层，768个隐单元，12个Attention head，110M参数
- BERT-Large, Uncased 24层，1024个隐单元，16个head，340M参数
- BERT-Base, Cased 12层，768个隐单元，12个Attention head，110M参数
- BERT-Large, Uncased 24层，1024个隐单元，16个head，340M参数。

Uncased的意思是保留大小写，而cased是在预处理的时候都变成了小写。

对于汉语只有一个版本：*BERT-Base, Chinese*: 包括简体和繁体汉字，共12层，768个隐单元，12个Attention head，110M参数。另外一个多语言的版本是*BERT-Base, Multilingual Cased (New, recommended)*，它包括104种不同语言，12层，768个隐单元，12个Attention head，110M参数。它是用所有这104中语言的维基百科文章混在一起训练出来的模型。所有这些模型的下载地址都在[这里](https://github.com/google-research/bert#pre-trained-models)。

## 1. BERT实现代码

### 读取数据

- DataProcessor类是一个抽象基类，定义了get_train_examples、get_dev_examples、get_test_examples和get_labels等4个需要子类实现的方法，另外提供了一个_read_tsv函数用于读取tsv文件。
- 对于不同的数据集，有不同的读取数据类，比如实现类MrpcProcessor。
- get_train_examples函数首先使用`_read_tsv`读入训练文件train.tsv，然后使用`_create_examples`函数把每一行变成一个InputExample对象。InputExample对象有4个属性：guid、text_a、text_b和label，guid只是个唯一的id而已。text_a代表第一个句子，text_b代表第二个句子，第二个句子可以为None，label代表分类标签。

### 分词

BERT里分词主要是由FullTokenizer类来实现的。

```python
class FullTokenizer(object): 
	def __init__(self, vocab_file, do_lower_case=True):
		self.vocab = load_vocab(vocab_file)
		self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
		self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)

	def tokenize(self, text):
		split_tokens = []
		for token in self.basic_tokenizer.tokenize(text):
			for sub_token in self.wordpiece_tokenizer.tokenize(token):
				split_tokens.append(sub_token)
		
		return split_tokens

	def convert_tokens_to_ids(self, tokens):
		return convert_tokens_to_ids(self.vocab, tokens)
```

FullTokenizer的构造函数需要传入参数词典vocab_file和do_lower_case。如果我们自己从头开始训练模型(后面会介绍)，那么do_lower_case决定了我们的某些是否区分大小写。如果我们只是Fine-Tuning，那么这个参数需要与模型一致，比如模型是uncased_L-12_H-768_A-12，那么do_lower_case就必须为True。

函数首先调用load_vocab加载词典，建立词到id的映射关系。接下来是构造BasicTokenizer和WordpieceTokenizer。前者是根据空格等进行普通的分词，而后者会把前者的结果再细粒度的切分为WordPiece。

**BasicTokenizer的tokenize方法。**

```python
def tokenize(self, text): 
  text = convert_to_unicode(text)
  text = self._clean_text(text)
  
  # 这是2018年11月1日为了支持多语言和中文增加的代码。这个代码也可以用于英语模型，因为在
  # 英语的训练数据中基本不会出现中文字符(但是某些wiki里偶尔也可能出现中文)。
  text = self._tokenize_chinese_chars(text)
  
  orig_tokens = whitespace_tokenize(text)
  split_tokens = []
  for token in orig_tokens:
	  if self.do_lower_case:
		  token = token.lower()
		  token = self._run_strip_accents(token)
	  split_tokens.extend(self._run_split_on_punc(token))
  
  output_tokens = whitespace_tokenize(" ".join(split_tokens))
  return output_tokens
```

首先是用convert_to_unicode把输入变成unicode，这是为了兼容Python2和Python3，因为Python3的str就是unicode，而Python2的str其实是bytearray，Python2却有一个专门的unicode类型。

接下来是_clean_text函数，它的作用是去除一些无意义的字符。

```python
def _clean_text(self, text):
  """去除一些无意义的字符以及whitespace"""
  output = []
  for char in text:
	  cp = ord(char)
	  if cp == 0 or cp == 0xfffd or _is_control(char):
		  continue
	  if _is_whitespace(char):
		  output.append(" ")
	  else:
		  output.append(char)
  return "".join(output)
```

codepoint为0的是无意义的字符，0xfffd(U+FFFD)显示为�，通常用于替换未知的字符。_is_control用于判断一个字符是否是控制字符(control character)，所谓的控制字符就是用于控制屏幕的显示，比如\n告诉(控制)屏幕把光标移到下一行的开始。读者可以参考[这里](https://en.wikipedia.org/wiki/Unicode_control_characters)。

```python
def _is_control(char):
	"""检查字符char是否是控制字符"""
	# 回车换行和tab理论上是控制字符，但是这里我们把它认为是whitespace而不是控制字符
	if char == "\t" or char == "\n" or char == "\r":
		return False
	cat = unicodedata.category(char)
	if cat.startswith("C"):
		return True
	return False
```

这里使用了unicodedata.category这个函数，它返回这个Unicode字符的Category，这里C开头的都被认为是控制字符，读者可以参考[这里](https://en.wikipedia.org/wiki/Unicode_character_property#General_Category)。

接下来是调用_is_whitespace函数，把whitespace变成空格。

```python
def _is_whitespace(char):
	"""Checks whether `chars` is a whitespace character."""
	# \t, \n, and \r are technically contorl characters but we treat them
	# as whitespace since they are generally considered as such.
	if char == " " or char == "\t" or char == "\n" or char == "\r":
		return True
	cat = unicodedata.category(char)
	if cat == "Zs":
		return True
	return False
```

这里把category为Zs的字符以及空格、tab、换行和回车当成whitespace。然后是_tokenize_chinese_chars，用于切分中文，这里的中文分词很简单，就是切分成一个一个的汉字。也就是在中文字符的前后加上空格，这样后续的分词流程会把每一个字符当成一个词。

```python
def _tokenize_chinese_chars(self, text): 
  output = []
  for char in text:
  cp = ord(char)
  if self._is_chinese_char(cp):
	  output.append(" ")
	  output.append(char)
	  output.append(" ")
  else:
	  output.append(char)
  return "".join(output)

def _is_chinese_char(self, cp):
        if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
		  (cp >= 0x3400 and cp <= 0x4DBF) or  #
		  (cp >= 0x20000 and cp <= 0x2A6DF) or  #
		  (cp >= 0x2A700 and cp <= 0x2B73F) or  #
		  (cp >= 0x2B740 and cp <= 0x2B81F) or  #
		  (cp >= 0x2B820 and cp <= 0x2CEAF) or
		  (cp >= 0xF900 and cp <= 0xFAFF) or  #
		  (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
        return True

        return False
```

很多网上的判断汉字的正则表达式都只包括4E00-9FA5，但这是不全的，比如 **㐈** 就不再这个范围内。读者可以参考[这里](https://www.cnblogs.com/straybirds/p/6392306.html)。

接下来是使用whitespace进行分词，这是通过函数whitespace_tokenize来实现的。它直接调用split函数来实现分词。Python里whitespace包括’\t\n\x0b\x0c\r ‘。然后遍历每一个词，如果需要变成小写，那么先用lower()函数变成小写，接着调用_run_strip_accents函数去除accent。

它首先调用unicodedata.normalize(“NFD”, text)对text进行归一化。

我们”看到”的é其实可以有两种表示方法，一是用一个codepoint直接表示”é”，另外一种是用”e”再加上特殊的codepoint U+0301两个字符来表示。U+0301是COMBINING ACUTE ACCENT，它跟在e之后就变成了”é”。类似的”a\u0301”显示出来就是”á”。注意：这只是打印出来一模一样而已，但是在计算机内部的表示它们完全不同的，前者é是一个codepoint，值为0xe9，而后者是两个codepoint，分别是0x65和0x301。unicodedata.normalize(“NFD”, text)就会把0xe9变成0x65和0x301

接下来遍历每一个codepoint，把category为Mn的去掉，比如前面的U+0301，COMBINING ACUTE ACCENT就会被去掉。category为Mn的所有Unicode字符完整列表在[这里](https://www.fileformat.info/info/unicode/category/Mn/list.htm)。

处理完大小写和accent之后得到的Token通过函数`_run_split_on_punc`再次用标点切分。这个函数会对输入字符串用标点进行切分，返回一个list，list的每一个元素都是一个char。比如输入he’s，则输出是[[h,e], [’],[s]]。代码很简单，这里就不赘述。里面它会调用函数`_is_punctuation`来判断一个字符是否标点。

**WordpieceTokenizer**

对于中文来说，WordpieceTokenizer什么也不干，因为之前的分词已经是基于字符的了。有兴趣的读者可以参考[这个](https://github.com/google/sentencepiece)开源项目。一般情况我们不需要自己重新生成WordPiece，使用BERT模型里自带的就行。

```python
ef tokenize(self, text):
  
  # 把一段文字切分成word piece。这其实是贪心的最大正向匹配算法。
  # 比如：
  # input = "unaffable"
  # output = ["un", "##aff", "##able"]
 
  
  text = convert_to_unicode(text)
  
  output_tokens = []
  for token in whitespace_tokenize(text):
	  chars = list(token)
	  if len(chars) > self.max_input_chars_per_word:
		  output_tokens.append(self.unk_token)
		  continue
	  
	  is_bad = False
	  start = 0
	  sub_tokens = []
	  while start < len(chars):
		  end = len(chars)
		  cur_substr = None
		  while start < end:
			  substr = "".join(chars[start:end])
			  if start > 0:
				  substr = "##" + substr
			  if substr in self.vocab:
				  cur_substr = substr
				  break
			  end -= 1
		  if cur_substr is None:
			  is_bad = True
			  break
		  sub_tokens.append(cur_substr)
		  start = end
	  
	  if is_bad:
		  output_tokens.append(self.unk_token)
	  else:
		  output_tokens.extend(sub_tokens)
  return output_tokens
```

代码有点长，但是很简单，就是贪心的最大正向匹配。其实为了加速，是可以把词典加载到一个Double Array Trie里的。我们用一个例子来看代码的执行过程。比如假设输入是”unaffable”。我们跳到while循环部分，这是start=0，end=len(chars)=9，也就是先看看unaffable在不在词典里，如果在，那么直接作为一个WordPiece，如果不再，那么end-=1，也就是看unaffabl在不在词典里，最终发现”un”在词典里，把un加到结果里。

接着start=2，看affable在不在，不在再看affabl，…，最后发现 **##aff** 在词典里。注意：##表示这个词是接着前面的，这样使得WordPiece切分是可逆的——我们可以恢复出“真正”的词。

### run_classifier.py的main函数

这里使用的是Tensorflow的Estimator API，在本书的Tensorflow部分我们已经介绍过了。训练、验证和预测的代码都很类似，我们这里只介绍训练部分的代码。

首先是通过file_based_convert_examples_to_features函数把输入的tsv文件变成TFRecord文件，便于Tensorflow处理。

file_based_convert_examples_to_features函数遍历每一个example(InputExample类的对象)。然后使用convert_single_example函数把每个InputExample对象变成InputFeature。InputFeature就是一个存放特征的对象，它包括input_ids、input_mask、segment_ids和label_id，这4个属性除了label_id是一个int之外，其它都是int的列表，因此使用create_int_feature函数把它变成tf.train.Feature，而label_id需要构造一个只有一个元素的列表，最后构造tf.train.Example对象，然后写到TFRecord文件里。后面Estimator的input_fn会用到它。

这里的最关键是convert_single_example函数，读懂了它就真正明白BERT把输入表示成向量的过程，所以请读者仔细阅读代码和其中的注释。

```
def convert_single_example(ex_index, example, label_list, max_seq_length,
				tokenizer):
	"""把一个`InputExample`对象变成`InputFeatures`."""
	# label_map把label变成id，这个函数每个example都需要执行一次，其实是可以优化的。
	# 只需要在可以再外面执行一次传入即可。
	label_map = {}
	for (i, label) in enumerate(label_list):
		label_map[label] = i
	
	tokens_a = tokenizer.tokenize(example.text_a)
	tokens_b = None
	if example.text_b:
		tokens_b = tokenizer.tokenize(example.text_b)
	
	if tokens_b:
		# 如果有b，那么需要保留3个特殊Token[CLS], [SEP]和[SEP]
		# 如果两个序列加起来太长，就需要去掉一些。
		_truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
	else:
		# 没有b则只需要保留[CLS]和[SEP]两个特殊字符
		# 如果Token太多，就直接截取掉后面的部分。
		if len(tokens_a) > max_seq_length - 2:
			tokens_a = tokens_a[0:(max_seq_length - 2)]
	
	# BERT的约定是：
	# (a) 对于两个序列：
	#  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
	#  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
	# (b) 对于一个序列：
	#  tokens:   [CLS] the dog is hairy . [SEP]
	#  type_ids: 0     0   0   0  0     0 0
	#
	# 这里"type_ids"用于区分一个Token是来自第一个还是第二个序列
	# 对于type=0和type=1，模型会学习出两个Embedding向量。
	# 虽然理论上这是不必要的，因为[SEP]隐式的确定了它们的边界。
	# 但是实际加上type后，模型能够更加容易的知道这个词属于那个序列。
	#
	# 对于分类任务，[CLS]对应的向量可以被看成 "sentence vector"
	# 注意：一定需要Fine-Tuning之后才有意义
	tokens = []
	segment_ids = []
	tokens.append("[CLS]")
	segment_ids.append(0)
	for token in tokens_a:
		tokens.append(token)
		segment_ids.append(0)
		tokens.append("[SEP]")
		segment_ids.append(0)
	
	if tokens_b:
		for token in tokens_b:
			tokens.append(token)
			segment_ids.append(1)
		tokens.append("[SEP]")
		segment_ids.append(1)
	
	input_ids = tokenizer.convert_tokens_to_ids(tokens)
	
	# mask是1表示是"真正"的Token，0则是Padding出来的。在后面的Attention时会通过tricky的技巧让
	# 模型不能attend to这些padding出来的Token上。
	input_mask = [1] * len(input_ids)
	
	# padding使得序列长度正好等于max_seq_length
	while len(input_ids) < max_seq_length:
		input_ids.append(0)
		input_mask.append(0)
		segment_ids.append(0)
 
	label_id = label_map[example.label]
	
	feature = InputFeatures(
		input_ids=input_ids,
		input_mask=input_mask,
		segment_ids=segment_ids,
		label_id=label_id)
	return feature
```

如果两个Token序列的长度太长，那么需要去掉一些，这会用到_truncate_seq_pair函数：

```
def _truncate_seq_pair(tokens_a, tokens_b, max_length):
	while True:
		total_length = len(tokens_a) + len(tokens_b)
		if total_length <= max_length:
			break
		if len(tokens_a) > len(tokens_b):
			tokens_a.pop()
		else:
			tokens_b.pop()
```

这个函数很简单，如果两个序列的长度小于max_length，那么不用truncate，否则在tokens_a和tokens_b中选择长的那个序列来pop掉最后面的那个Token，这样的结果是使得两个Token序列一样长(或者最多a比b多一个Token)。对于Estimator API来说，最重要的是实现model_fn和input_fn。我们先看input_fn，它是由file_based_input_fn_builder构造出来的。代码如下：

```
def file_based_input_fn_builder(input_file, seq_length, is_training,
			drop_remainder):
 
	name_to_features = {
		"input_ids": tf.FixedLenFeature([seq_length], tf.int64),
		"input_mask": tf.FixedLenFeature([seq_length], tf.int64),
		"segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
		"label_ids": tf.FixedLenFeature([], tf.int64),
	}
	
	def _decode_record(record, name_to_features):
		# 把record decode成TensorFlow example.
		example = tf.parse_single_example(record, name_to_features)
		
		# tf.Example只支持tf.int64，但是TPU只支持tf.int32.
		# 因此我们把所有的int64变成int32.
		for name in list(example.keys()):
			t = example[name]
			if t.dtype == tf.int64:
				t = tf.to_int32(t)
			example[name] = t
		
		return example
	
	def input_fn(params): 
		batch_size = params["batch_size"]
		
		# 对于训练来说，我们会重复的读取和shuffling 
		# 对于验证和测试，我们不需要shuffling和并行读取。
		d = tf.data.TFRecordDataset(input_file)
		if is_training:
			d = d.repeat()
			d = d.shuffle(buffer_size=100)
		
		d = d.apply(
				tf.contrib.data.map_and_batch(
					lambda record: _decode_record(record, name_to_features),
					batch_size=batch_size,
					drop_remainder=drop_remainder))
		
		return d
	
	return input_fn
```

这个函数返回一个函数input_fn。这个input_fn函数首先从文件得到TFRecordDataset，然后根据是否训练来shuffle和重复读取。然后用applay函数对每一个TFRecord进行map_and_batch，调用_decode_record函数对record进行parsing。从而把TFRecord的一条Record变成tf.Example对象，这个对象包括了input_ids等4个用于训练的Tensor。

接下来是model_fn_builder，它用于构造Estimator使用的model_fn。下面是它的主要代码(一些无关的log和TPU相关代码去掉了)：

```
def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
				num_train_steps, num_warmup_steps, use_tpu,
				use_one_hot_embeddings): 
	# 注意：在model_fn的设计里，features表示输入(特征)，而labels表示输出
	# 但是这里的实现有点不好，把label也放到了features里。
	def model_fn(features, labels, mode, params): 
		input_ids = features["input_ids"]
		input_mask = features["input_mask"]
		segment_ids = features["segment_ids"]
		label_ids = features["label_ids"]
		
		is_training = (mode == tf.estimator.ModeKeys.TRAIN)
		
		# 创建Transformer模型，这是最主要的代码。
		(total_loss, per_example_loss, logits, probabilities) = create_model(
			bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
			num_labels, use_one_hot_embeddings)
		
		tvars = tf.trainable_variables()
		
		# 从checkpoint恢复参数
		if init_checkpoint: 
			(assignment_map, initialized_variable_names) = 	
				modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
			
			tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
		 
		
		output_spec = None
		# 构造训练的spec
		if mode == tf.estimator.ModeKeys.TRAIN:
			train_op = optimization.create_optimizer(total_loss, learning_rate, 
							num_train_steps, num_warmup_steps, use_tpu)
			
			output_spec = tf.contrib.tpu.TPUEstimatorSpec(
					mode=mode,
					loss=total_loss,
					train_op=train_op,
					scaffold_fn=scaffold_fn)
					
		# 构造eval的spec
		elif mode == tf.estimator.ModeKeys.EVAL:	
			def metric_fn(per_example_loss, label_ids, logits):
				predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
				accuracy = tf.metrics.accuracy(label_ids, predictions)
				loss = tf.metrics.mean(per_example_loss)
				return {
					"eval_accuracy": accuracy,
					"eval_loss": loss,
				}
			
			eval_metrics = (metric_fn, [per_example_loss, label_ids, logits])
			output_spec = tf.contrib.tpu.TPUEstimatorSpec(
				mode=mode,
				loss=total_loss,
				eval_metrics=eval_metrics,
				scaffold_fn=scaffold_fn)
		
		# 预测的spec
		else:
			output_spec = tf.contrib.tpu.TPUEstimatorSpec(
				mode=mode,
				predictions=probabilities,
				scaffold_fn=scaffold_fn)
		return output_spec
	
	return model_fn
```

这里的代码都是一些boilerplate代码，没什么可说的，最重要的是调用create_model”真正”的创建Transformer模型。下面我们来看这个函数的代码：

```
def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
					labels, num_labels, use_one_hot_embeddings): 
	model = modeling.BertModel(
			config=bert_config,
			is_training=is_training,
			input_ids=input_ids,
			input_mask=input_mask,
			token_type_ids=segment_ids,
			use_one_hot_embeddings=use_one_hot_embeddings)
	
	# 在这里，我们是用来做分类，因此我们只需要得到[CLS]最后一层的输出。
	# 如果需要做序列标注，那么可以使用model.get_sequence_output()
	# 默认参数下它返回的output_layer是[8, 768]
	output_layer = model.get_pooled_output()
	
	# 默认是768
	hidden_size = output_layer.shape[-1].value
	
	
	output_weights = tf.get_variable(
		"output_weights", [num_labels, hidden_size],
		initializer=tf.truncated_normal_initializer(stddev=0.02))
	
	output_bias = tf.get_variable(
		"output_bias", [num_labels], initializer=tf.zeros_initializer())
	
	with tf.variable_scope("loss"):
		if is_training:
			# 0.1的概率会dropout
			output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
			
		# 对[CLS]输出的768的向量再做一个线性变换，输出为label的个数。得到logits
		logits = tf.matmul(output_layer, output_weights, transpose_b=True)
		logits = tf.nn.bias_add(logits, output_bias)
		probabilities = tf.nn.softmax(logits, axis=-1)
		log_probs = tf.nn.log_softmax(logits, axis=-1)
		
		one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
		
		per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
		loss = tf.reduce_mean(per_example_loss)
	
	return (loss, per_example_loss, logits, probabilities)
```

上面代码调用modeling.BertModel得到BERT模型，然后使用它的get_pooled_output方法得到[CLS]最后一层的输出，这是一个768(默认参数下)的向量，然后就是常规的接一个全连接层得到logits，然后softmax得到概率，之后就可以根据真实的分类标签计算loss。我们这时候发现关键的代码是modeling.BertModel。

### BertModel类

我们首先来看这个类的用法，把它当成黑盒。前面的create_model也用到了BertModel，这里我们在详细的介绍一下。下面的代码演示了BertModel的使用方法：

```
  # 假设输入已经分词并且变成WordPiece的id了 
  # 输入是[2, 3]，表示batch=2，max_seq_length=3
  input_ids = tf.constant([[31, 51, 99], [15, 5, 0]])
  # 第一个例子实际长度为3，第二个例子长度为2
  input_mask = tf.constant([[1, 1, 1], [1, 1, 0]])
  # 第一个例子的3个Token中前两个属于句子1，第三个属于句子2
  # 而第二个例子的第一个Token属于句子1，第二个属于句子2(第三个是padding)
  token_type_ids = tf.constant([[0, 0, 1], [0, 1, 0]])
  
  # 创建一个BertConfig，词典大小是32000，Transformer的隐单元个数是512
  # 8个Transformer block，每个block有8个Attention Head，全连接层的隐单元是1024
  config = modeling.BertConfig(vocab_size=32000, hidden_size=512,
		  num_hidden_layers=8, num_attention_heads=8, intermediate_size=1024)

  # 创建BertModel
  model = modeling.BertModel(config=config, is_training=True,
		  input_ids=input_ids, input_mask=input_mask, token_type_ids=token_type_ids)
  
  # label_embeddings用于把512的隐单元变换成logits
  label_embeddings = tf.get_variable(...)
  # 得到[CLS]最后一层输出，把它看成句子的Embedding(Encoding)
  pooled_output = model.get_pooled_output()
  # 计算logits
  logits = tf.matmul(pooled_output, label_embeddings)
```

接下来我们看一下BertModel的构造函数：

```
def __init__(self,
		  config,
		  is_training,
		  input_ids,
		  input_mask=None,
		  token_type_ids=None,
		  use_one_hot_embeddings=True,
		  scope=None): 

  # Args:
  #       config: `BertConfig` 对象
  #       is_training: bool 表示训练还是eval，是会影响dropout
  #	  input_ids: int32 Tensor  shape是[batch_size, seq_length]
  #	  input_mask: (可选) int32 Tensor shape是[batch_size, seq_length]
  #	  token_type_ids: (可选) int32 Tensor shape是[batch_size, seq_length]
  #	  use_one_hot_embeddings: (可选) bool
  #		  如果True，使用矩阵乘法实现提取词的Embedding；否则用tf.embedding_lookup()
  #		  对于TPU，使用前者更快，对于GPU和CPU，后者更快。
  #	  scope: (可选) 变量的scope。默认是"bert"
  
  # Raises:
  #	  ValueError: 如果config或者输入tensor的shape有问题就会抛出这个异常

  config = copy.deepcopy(config)
  if not is_training:
	  config.hidden_dropout_prob = 0.0
	  config.attention_probs_dropout_prob = 0.0
  
  input_shape = get_shape_list(input_ids, expected_rank=2)
  batch_size = input_shape[0]
  seq_length = input_shape[1]
  
  if input_mask is None:
	  input_mask = tf.ones(shape=[batch_size, seq_length], dtype=tf.int32)
  
  if token_type_ids is None:
	  token_type_ids = tf.zeros(shape=[batch_size, seq_length], dtype=tf.int32)
  
  with tf.variable_scope(scope, default_name="bert"):
	  with tf.variable_scope("embeddings"):
		  # 词的Embedding lookup 
		  (self.embedding_output, self.embedding_table) = embedding_lookup(
				  input_ids=input_ids,
				  vocab_size=config.vocab_size,
				  embedding_size=config.hidden_size,
				  initializer_range=config.initializer_range,
				  word_embedding_name="word_embeddings",
				  use_one_hot_embeddings=use_one_hot_embeddings)
		  
		  # 增加位置embeddings和token type的embeddings，然后是
		  # layer normalize和dropout。
		  self.embedding_output = embedding_postprocessor(
				  input_tensor=self.embedding_output,
				  use_token_type=True,
				  token_type_ids=token_type_ids,
				  token_type_vocab_size=config.type_vocab_size,
				  token_type_embedding_name="token_type_embeddings",
				  use_position_embeddings=True,
				  position_embedding_name="position_embeddings",
				  initializer_range=config.initializer_range,
				  max_position_embeddings=config.max_position_embeddings,
				  dropout_prob=config.hidden_dropout_prob)
	  
	  with tf.variable_scope("encoder"):
		  # 把shape为[batch_size, seq_length]的2D mask变成
		  # shape为[batch_size, seq_length, seq_length]的3D mask
		  # 以便后向的attention计算，读者可以对比之前的Transformer的代码。
		  attention_mask = create_attention_mask_from_input_mask(
				  input_ids, input_mask)
		  
		  # 多个Transformer模型stack起来。
		  # all_encoder_layers是一个list，长度为num_hidden_layers（默认12），每一层对应一个值。
		  # 每一个值都是一个shape为[batch_size, seq_length, hidden_size]的tensor。
		  
		  self.all_encoder_layers = transformer_model(
			  input_tensor=self.embedding_output,
			  attention_mask=attention_mask,
			  hidden_size=config.hidden_size,
			  num_hidden_layers=config.num_hidden_layers,
			  num_attention_heads=config.num_attention_heads,
			  intermediate_size=config.intermediate_size,
			  intermediate_act_fn=get_activation(config.hidden_act),
			  hidden_dropout_prob=config.hidden_dropout_prob,
			  attention_probs_dropout_prob=config.attention_probs_dropout_prob,
			  initializer_range=config.initializer_range,
			  do_return_all_layers=True)
	  
	  # `sequence_output` 是最后一层的输出，shape是[batch_size, seq_length, hidden_size]
	  self.sequence_output = self.all_encoder_layers[-1]

	  with tf.variable_scope("pooler"):
		  # 取最后一层的第一个时刻[CLS]对应的tensor
		  # 从[batch_size, seq_length, hidden_size]变成[batch_size, hidden_size]
		  # sequence_output[:, 0:1, :]得到的是[batch_size, 1, hidden_size]
		  # 我们需要用squeeze把第二维去掉。
		  first_token_tensor = tf.squeeze(self.sequence_output[:, 0:1, :], axis=1)
		  # 然后再加一个全连接层，输出仍然是[batch_size, hidden_size]
		  self.pooled_output = tf.layers.dense(
				  first_token_tensor,
				  config.hidden_size,
				  activation=tf.tanh,
				  kernel_initializer=create_initializer(config.initializer_range))
```

代码很长，但是其实很简单。首先是对config(BertConfig对象)深度拷贝一份，如果不是训练，那么把dropout都置为零。如果输入的input_mask为None，那么构造一个shape合适值全为1的input_mask，这表示输入都是”真实”的输入，没有padding的内容。如果token_type_ids为None，那么构造一个shape合适并且值全为0的tensor，表示所有Token都属于第一个句子。

然后使用embedding_lookup函数构造词的Embedding，用embedding_postprocessor函数增加位置embeddings和token type的embeddings，然后是layer normalize和dropout。

接着用transformer_model函数构造多个Transformer SubLayer然后stack在一起。得到的all_encoder_layers是一个list，长度为num_hidden_layers（默认12），每一层对应一个值。 每一个值都是一个shape为[batch_size, seq_length, hidden_size]的tensor。

self.sequence_output是最后一层的输出，shape是[batch_size, seq_length, hidden_size]。first_token_tensor是第一个Token([CLS])最后一层的输出，shape是[batch_size, hidden_size]。最后对self.sequence_output再加一个线性变换，得到的tensor仍然是[batch_size, hidden_size]。

embedding_lookup函数用于实现Embedding，它有两种方式：使用tf.nn.embedding_lookup和矩阵乘法(one_hot_embedding=True)。前者适合于CPU与GPU，后者适合于TPU。所谓的one-hot方法是把输入id表示成one-hot的向量，当然输入id序列就变成了one-hot的矩阵，然后乘以Embedding矩阵。而tf.nn.embedding_lookup是直接用id当下标提取Embedding矩阵对应的向量。一般认为tf.nn.embedding_lookup更快一点，但是TPU上似乎不是这样，作者也不太了解原因是什么，猜测可能是TPU的没有快捷的办法提取矩阵的某一行/列？

```
def embedding_lookup(input_ids,
			vocab_size,
			embedding_size=128,
			initializer_range=0.02,
			word_embedding_name="word_embeddings",
			use_one_hot_embeddings=False):
	"""word embedding
	
	Args:
		input_ids: int32 Tensor shape为[batch_size, seq_length]，表示WordPiece的id
		vocab_size: int 词典大小，需要于vocab.txt一致 
		embedding_size: int embedding后向量的大小 
		initializer_range: float 随机初始化的范围 
		word_embedding_name: string 名字，默认是"word_embeddings"
		use_one_hot_embeddings: bool 如果True，使用one-hot方法实现embedding；否则使用 		
			`tf.nn.embedding_lookup()`. TPU适合用One hot方法。
	
	Returns:
		float Tensor shape为[batch_size, seq_length, embedding_size]
	"""
	# 这个函数假设输入的shape是[batch_size, seq_length, num_inputs]
	# 普通的Embeding一般假设输入是[batch_size, seq_length]，
	# 增加num_inputs这一维度的目的是为了一次计算更多的Embedding
	# 但目前的代码并没有用到，传入的input_ids都是2D的，这增加了代码的阅读难度。
	
	# 如果输入是[batch_size, seq_length]，
	# 那么我们把它 reshape成[batch_size, seq_length, 1]
	if input_ids.shape.ndims == 2:
		input_ids = tf.expand_dims(input_ids, axis=[-1])
	
	# 构造Embedding矩阵，shape是[vocab_size, embedding_size]
	embedding_table = tf.get_variable(
		name=word_embedding_name,
		shape=[vocab_size, embedding_size],
		initializer=create_initializer(initializer_range))
	
	if use_one_hot_embeddings:
		flat_input_ids = tf.reshape(input_ids, [-1])
		one_hot_input_ids = tf.one_hot(flat_input_ids, depth=vocab_size)
		output = tf.matmul(one_hot_input_ids, embedding_table)
	else:
		output = tf.nn.embedding_lookup(embedding_table, input_ids)
	
	input_shape = get_shape_list(input_ids)
	# 把输出从[batch_size, seq_length, num_inputs(这里总是1), embedding_size]
	# 变成[batch_size, seq_length, num_inputs*embedding_size]
	output = tf.reshape(output,
				input_shape[0:-1] + [input_shape[-1] * embedding_size])
	return (output, embedding_table)
```

Embedding本来很简单，使用tf.nn.embedding_lookup就行了。但是为了优化TPU，它还支持使用矩阵乘法来提取词向量。另外为了提高效率，输入的shape除了[batch_size, seq_length]外，它还增加了一个维度变成[batch_size, seq_length, num_inputs]。如果不关心细节，我们把这个函数当成黑盒，那么我们只需要知道它的输入input_ids(可能)是[8, 128]，输出是[8, 128, 768]就可以了。

函数embedding_postprocessor的代码如下，需要注意的部分都有注释。

```
def embedding_postprocessor(input_tensor,
				use_token_type=False,
				token_type_ids=None,
				token_type_vocab_size=16,
				token_type_embedding_name="token_type_embeddings",
				use_position_embeddings=True,
				position_embedding_name="position_embeddings",
				initializer_range=0.02,
				max_position_embeddings=512,
				dropout_prob=0.1):
	"""对word embedding之后的tensor进行后处理
	
	Args:
		input_tensor: float Tensor shape为[batch_size, seq_length, embedding_size]
		use_token_type: bool 是否增加`token_type_ids`的Embedding
		token_type_ids: (可选) int32 Tensor shape为[batch_size, seq_length]
			如果`use_token_type`为True则必须有值
		token_type_vocab_size: int Token Type的个数，通常是2
		token_type_embedding_name: string Token type Embedding的名字
		use_position_embeddings: bool 是否使用位置Embedding
		position_embedding_name: string，位置embedding的名字 
		initializer_range: float，初始化范围 
		max_position_embeddings: int，位置编码的最大长度，可以比最大序列长度大，但是不能比它小。
		dropout_prob: float. Dropout 概率
		
	Returns:
		float tensor  shape和`input_tensor`相同。
	 
	"""
	input_shape = get_shape_list(input_tensor, expected_rank=3)
	batch_size = input_shape[0]
	seq_length = input_shape[1]
	width = input_shape[2]
	
	if seq_length > max_position_embeddings:
		raise ValueError("The seq length (%d) cannot be greater than "
			"`max_position_embeddings` (%d)" %
					(seq_length, max_position_embeddings))
	
	output = input_tensor
	
	if use_token_type:
		if token_type_ids is None:
			raise ValueError("`token_type_ids` must be specified if"
				"`use_token_type` is True.")
		token_type_table = tf.get_variable(
				name=token_type_embedding_name,
				shape=[token_type_vocab_size, width],
				initializer=create_initializer(initializer_range))
		# 因为Token Type通常很小(2)，所以直接用矩阵乘法(one-hot)更快
		flat_token_type_ids = tf.reshape(token_type_ids, [-1])
		one_hot_ids = tf.one_hot(flat_token_type_ids, depth=token_type_vocab_size)
		token_type_embeddings = tf.matmul(one_hot_ids, token_type_table)
		token_type_embeddings = tf.reshape(token_type_embeddings,
				[batch_size, seq_length, width])
		output += token_type_embeddings
	
	if use_position_embeddings:
		full_position_embeddings = tf.get_variable(
					name=position_embedding_name,
					shape=[max_position_embeddings, width],
					initializer=create_initializer(initializer_range))
		# 位置Embedding是可以学习的参数，因此我们创建一个[max_position_embeddings, width]的矩阵
		# 但实际输入的序列可能并不会到max_position_embeddings(512)，为了提高训练速度，
		# 我们通过tf.slice取出[0, 1, 2, ..., seq_length-1]的部分,。
		if seq_length < max_position_embeddings:
			position_embeddings = tf.slice(full_position_embeddings, [0, 0],
					[seq_length, -1])
		else:
			position_embeddings = full_position_embeddings
		
		num_dims = len(output.shape.as_list())
		
		# word embedding之后的tensor是[batch_size, seq_length, width]
		# 因为位置编码是与输入内容无关，它的shape总是[seq_length, width]
		# 我们无法把位置Embedding加到word embedding上
		# 因此我们需要扩展位置编码为[1, seq_length, width]
		# 然后就能通过broadcasting加上去了。
		position_broadcast_shape = []
		for _ in range(num_dims - 2):
			position_broadcast_shape.append(1)
		position_broadcast_shape.extend([seq_length, width])
		# 默认情况下position_broadcast_shape为[1, 128, 768]
		position_embeddings = tf.reshape(position_embeddings,
			position_broadcast_shape)
		# output是[8, 128, 768], position_embeddings是[1, 128, 768]
		# 因此可以通过broadcasting相加。
		output += position_embeddings
	
	output = layer_norm_and_dropout(output, dropout_prob)
	return output
```

create_attention_mask_from_input_mask函数用于构造Mask矩阵。我们先了解一下它的作用然后再阅读其代码。比如调用它时的两个参数是是：

```
input_ids=[
	[1,2,3,0,0],
	[1,3,5,6,1]
]
input_mask=[
	[1,1,1,0,0],
	[1,1,1,1,1]
]
```

表示这个batch有两个样本，第一个样本长度为3(padding了2个0)，第二个样本长度为5。在计算Self-Attention的时候每一个样本都需要一个Attention Mask矩阵，表示每一个时刻可以attend to的范围，1表示可以attend，0表示是padding的(或者在机器翻译的Decoder中不能attend to未来的词)。对于上面的输入，这个函数返回一个shape是[2, 5, 5]的tensor，分别代表两个Attention Mask矩阵。

```
[
	[1, 1, 1, 0, 0], #它表示第1个词可以attend to 3个词
	[1, 1, 1, 0, 0], #它表示第2个词可以attend to 3个词
	[1, 1, 1, 0, 0], #它表示第3个词可以attend to 3个词
	[1, 1, 1, 0, 0], #无意义，因为输入第4个词是padding的0
	[1, 1, 1, 0, 0]  #无意义，因为输入第5个词是padding的0
]

[
	[1, 1, 1, 1, 1], # 它表示第1个词可以attend to 5个词
	[1, 1, 1, 1, 1], # 它表示第2个词可以attend to 5个词
	[1, 1, 1, 1, 1], # 它表示第3个词可以attend to 5个词
	[1, 1, 1, 1, 1], # 它表示第4个词可以attend to 5个词
	[1, 1, 1, 1, 1]	 # 它表示第5个词可以attend to 5个词
]
```

了解了它的用途之后下面的代码就很好理解了。

```
def create_attention_mask_from_input_mask(from_tensor, to_mask):
	"""Create 3D attention mask from a 2D tensor mask.
	
	Args:
		from_tensor: 2D or 3D Tensor，shape为[batch_size, from_seq_length, ...].
		to_mask: int32 Tensor， shape为[batch_size, to_seq_length].
	
	Returns:
		float Tensor，shape为[batch_size, from_seq_length, to_seq_length].
	"""
	from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
	batch_size = from_shape[0]
	from_seq_length = from_shape[1]
	
	to_shape = get_shape_list(to_mask, expected_rank=2)
	to_seq_length = to_shape[1]
	
	to_mask = tf.cast(
		tf.reshape(to_mask, [batch_size, 1, to_seq_length]), tf.float32)
	
	# `broadcast_ones` = [batch_size, from_seq_length, 1]
	broadcast_ones = tf.ones(
		shape=[batch_size, from_seq_length, 1], dtype=tf.float32)
	
	# Here we broadcast along two dimensions to create the mask.
	mask = broadcast_ones * to_mask
	
	return mask
```

比如前面举的例子，broadcast_ones的shape是[2, 5, 1]，值全是1，而to_mask是

```
to_mask=[
[1,1,1,0,0],
[1,1,1,1,1]
]
```

shape是[2, 5]，reshape为[2, 1, 5]。然后broadcast_ones * to_mask就得到[2, 5, 5]，正是我们需要的两个Mask矩阵，读者可以验证。注意[batch, A, B]*[batch, B, C]=[batch, A, C]，我们可以认为是batch个[A, B]的矩阵乘以batch个[B, C]的矩阵。接下来就是transformer_model函数了，它就是构造Transformer的核心代码。

```
def transformer_model(input_tensor,
      attention_mask=None,
      hidden_size=768,
      num_hidden_layers=12,
      num_attention_heads=12,
      intermediate_size=3072,
      intermediate_act_fn=gelu,
      hidden_dropout_prob=0.1,
      attention_probs_dropout_prob=0.1,
      initializer_range=0.02,
      do_return_all_layers=False):
  """Multi-headed, multi-layer的Transformer，参考"Attention is All You Need".
  
  这基本上是和原始Transformer encoder相同的代码。
  
  原始论文为:
  https://arxiv.org/abs/1706.03762
  
  Also see:
  https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py
  
  Args:
    input_tensor: float Tensor，shape为[batch_size, seq_length, hidden_size]
    attention_mask: (可选) int32 Tensor，shape [batch_size, seq_length,
      seq_length], 1表示可以attend to，0表示不能。 
    hidden_size: int. Transformer隐单元个数
    num_hidden_layers: int. 有多少个SubLayer 
    num_attention_heads: int. Transformer Attention Head个数。
    intermediate_size: int. 全连接层的隐单元个数
    intermediate_act_fn: 函数. 全连接层的激活函数。
    hidden_dropout_prob: float. Self-Attention层残差之前的Dropout概率
    attention_probs_dropout_prob: float. attention的Dropout概率
    initializer_range: float. 初始化范围(truncated normal的标准差)
    do_return_all_layers: 返回所有层的输出还是最后一层的输出。
  
  Returns:
    如果do_return_all_layers True，返回最后一层的输出，是一个Tensor，
                shape为[batch_size, seq_length, hidden_size]；
    否则返回所有层的输出，是一个长度为num_hidden_layers的list，
                list的每一个元素都是[batch_size, seq_length, hidden_size]。

  """
  if hidden_size % num_attention_heads != 0:
    raise ValueError(
      "The hidden size (%d) is not a multiple of the number of attention "
      "heads (%d)" % (hidden_size, num_attention_heads))
  
  # 因为最终要输出hidden_size，总共有num_attention_heads个Head，因此每个Head输出
  # 为hidden_size / num_attention_heads
  attention_head_size = int(hidden_size / num_attention_heads)
  input_shape = get_shape_list(input_tensor, expected_rank=3)
  batch_size = input_shape[0]
  seq_length = input_shape[1]
  input_width = input_shape[2]
  
  # 因为需要残差连接，我们需要把输入加到Self-Attention的输出，因此要求它们的shape是相同的。
  if input_width != hidden_size:
    raise ValueError("The width of the input tensor (%d) != hidden size (%d)" %
      (input_width, hidden_size))
  
  # 为了避免在2D和3D之间来回reshape，我们统一把所有的3D Tensor用2D来表示。
  # 虽然reshape在GPU/CPU上很快，但是在TPU上却不是这样，这样做的目的是为了优化TPU
  # input_tensor是[8, 128, 768], prev_output是[8*128, 768]=[1024, 768] 
  prev_output = reshape_to_matrix(input_tensor)
  
  all_layer_outputs = []
  for layer_idx in range(num_hidden_layers):
    # 每一层都有自己的variable scope
    with tf.variable_scope("layer_%d" % layer_idx):
      layer_input = prev_output
      # attention层
      with tf.variable_scope("attention"):
        attention_heads = []
        # self attention
        with tf.variable_scope("self"):
          attention_head = attention_layer(
            from_tensor=layer_input,
            to_tensor=layer_input,
            attention_mask=attention_mask,
            num_attention_heads=num_attention_heads,
            size_per_head=attention_head_size,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            initializer_range=initializer_range,
            do_return_2d_tensor=True,
            batch_size=batch_size,
            from_seq_length=seq_length,
            to_seq_length=seq_length)
          attention_heads.append(attention_head)
        
        attention_output = None
        if len(attention_heads) == 1:
          attention_output = attention_heads[0]
        else:
          # 如果有多个head，那么需要把多个head的输出concat起来
          attention_output = tf.concat(attention_heads, axis=-1)
      
        # 使用线性变换把前面的输出变成`hidden_size`，然后再加上`layer_input`(残差连接)
        with tf.variable_scope("output"):
          attention_output = tf.layers.dense(
              attention_output,
              hidden_size,
              kernel_initializer=create_initializer(initializer_range))
          # dropout
          attention_output = dropout(attention_output, hidden_dropout_prob)
          # 残差连接再加上layer norm。
          attention_output = layer_norm(attention_output + layer_input)
      
      # 全连接层
      with tf.variable_scope("intermediate"):
        intermediate_output = tf.layers.dense(
          attention_output,
          intermediate_size,
          activation=intermediate_act_fn,
          kernel_initializer=create_initializer(initializer_range))
      
      # 然后是用一个线性变换把大小变回`hidden_size`，这样才能加残差连接
      with tf.variable_scope("output"):
        layer_output = tf.layers.dense(
            intermediate_output,
            hidden_size,
            kernel_initializer=create_initializer(initializer_range))
        layer_output = dropout(layer_output, hidden_dropout_prob)
        layer_output = layer_norm(layer_output + attention_output)
        prev_output = layer_output
        all_layer_outputs.append(layer_output)
  
  if do_return_all_layers:
    final_outputs = []
    for layer_output in all_layer_outputs:
      final_output = reshape_from_matrix(layer_output, input_shape)
      final_outputs.append(final_output)
    return final_outputs
  else:
    final_output = reshape_from_matrix(prev_output, input_shape)
    return final_output
```

如果对照Transformer的论文，非常容易阅读，里面实现Self-Attention的函数就是attention_layer。

```
def attention_layer(from_tensor,
			to_tensor,
			attention_mask=None,
			num_attention_heads=1,
			size_per_head=512,
			query_act=None,
			key_act=None,
			value_act=None,
			attention_probs_dropout_prob=0.0,
			initializer_range=0.02,
			do_return_2d_tensor=False,
			batch_size=None,
			from_seq_length=None,
			to_seq_length=None):
	"""用`from_tensor`(作为Query)去attend to `to_tensor`(提供Key和Value)
	
	这个函数实现论文"Attention
	is all you Need"里的multi-head attention。
	如果`from_tensor`和`to_tensor`是同一个tensor，那么就实现Self-Attention。
	`from_tensor`的每个时刻都会attends to `to_tensor`，
        也就是用from的Query去乘以所有to的Key，得到weight，然后把所有to的Value加权求和起来。
	
	这个函数首先把`from_tensor`变换成一个"query" tensor，
        然后把`to_tensor`变成"key"和"value" tensors。
        总共有`num_attention_heads`组Query、Key和Value，
        每一个Query，Key和Value的shape都是[batch_size(8), seq_length(128), size_per_head(512/8=64)].
	
	然后计算query和key的内积并且除以size_per_head的平方根(8)。
        然后softmax变成概率，最后用概率加权value得到输出。
        因为有多个Head，每个Head都输出[batch_size, seq_length, size_per_head]，
        最后把8个Head的结果concat起来，就最终得到[batch_size(8), seq_length(128), size_per_head*8=512] 
	
	实际上我们是把这8个Head的Query，Key和Value都放在一个Tensor里面的，
        因此实际通过transpose和reshape就达到了上面的效果。
	
	Args:
		from_tensor: float Tensor，shape [batch_size, from_seq_length, from_width]
		to_tensor: float Tensor，shape [batch_size, to_seq_length, to_width].
		attention_mask: (可选) int32 Tensor, shape[batch_size,from_seq_length,to_seq_length]。
                    值可以是0或者1，在计算attention score的时候，
                    我们会把0变成负无穷(实际是一个绝对值很大的负数)，而1不变，
                    这样softmax的时候进行exp的计算，前者就趋近于零，从而间接实现Mask的功能。
		num_attention_heads: int. Attention heads的数量。
		size_per_head: int. 每个head的size
		query_act: (可选) query变换的激活函数
		key_act: (可选) key变换的激活函数
		value_act: (可选) value变换的激活函数
		attention_probs_dropout_prob: (可选) float. attention的Dropout概率。
		initializer_range: float. 初始化范围 
		do_return_2d_tensor: bool. 如果True，返回2D的Tensor其shape是
                    [batch_size * from_seq_length, num_attention_heads * size_per_head]；
                    否则返回3D的Tensor其shape为[batch_size, from_seq_length, 
                                                num_attention_heads * size_per_head].
		batch_size: (可选) int. 如果输入是3D的，那么batch就是第一维，
                    但是可能3D的压缩成了2D的，所以需要告诉函数batch_size 
		from_seq_length: (可选) 同上，需要告诉函数from_seq_length
		to_seq_length: (可选) 同上，to_seq_length
	
	Returns:
		float Tensor，shape [batch_size,from_seq_length,num_attention_heads * size_per_head]。
		如果`do_return_2d_tensor`为True，则返回的shape是
                       [batch_size * from_seq_length, num_attention_heads * size_per_head].
	 
	"""
	
	def transpose_for_scores(input_tensor, batch_size, num_attention_heads,
			seq_length, width):
		output_tensor = tf.reshape(
				input_tensor, [batch_size, seq_length, num_attention_heads, width])
		
		output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
		return output_tensor
	
	from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
	to_shape = get_shape_list(to_tensor, expected_rank=[2, 3])
	
	if len(from_shape) != len(to_shape):
		raise ValueError(
			"The rank of `from_tensor` must match the rank of `to_tensor`.")
	# 如果输入是3D的(没有压缩)，那么我们可以推测出batch_size、from_seq_length和to_seq_length
	# 即使参数传入也会被覆盖。
	if len(from_shape) == 3:
		batch_size = from_shape[0]
		from_seq_length = from_shape[1]
		to_seq_length = to_shape[1]
		
	# 如果是压缩成2D的，那么一定要传入这3个参数，否则抛异常。	
	elif len(from_shape) == 2:
		if (batch_size is None or from_seq_length is None or to_seq_length is None):
			raise ValueError(
				"When passing in rank 2 tensors to attention_layer, the values "
				"for `batch_size`, `from_seq_length`, and `to_seq_length` "
				"must all be specified.")
	
	#   B = batch size (number of sequences) 默认配置是8
	#   F = `from_tensor` sequence length 默认配置是128
	#   T = `to_tensor` sequence length 默认配置是128
	#   N = `num_attention_heads` 默认配置是12
	#   H = `size_per_head` 默认配置是64
	
	# 把from和to压缩成2D的。
	# [8*128, 768]
	from_tensor_2d = reshape_to_matrix(from_tensor)
	# [8*128, 768]
	to_tensor_2d = reshape_to_matrix(to_tensor)
	
	# 计算Query `query_layer` = [B*F, N*H] =[8*128, 12*64]
	# batch_size=8，共128个时刻，12和head，每个head的query向量是64
	# 因此最终得到[8*128, 12*64]
	query_layer = tf.layers.dense(
			from_tensor_2d,
			num_attention_heads * size_per_head,
			activation=query_act,
			name="query",
			kernel_initializer=create_initializer(initializer_range))
	
	# 和query类似，`key_layer` = [B*T, N*H]
	key_layer = tf.layers.dense(
			to_tensor_2d,
			num_attention_heads * size_per_head,
			activation=key_act,
			name="key",
			kernel_initializer=create_initializer(initializer_range))
	
	# 同上，`value_layer` = [B*T, N*H]
	value_layer = tf.layers.dense(
			to_tensor_2d,
			num_attention_heads * size_per_head,
			activation=value_act,
			name="value",
			kernel_initializer=create_initializer(initializer_range))
	
	# 把query从[B*F, N*H] =[8*128, 12*64]变成[B, N, F, H]=[8, 12, 128, 64]
	query_layer = transpose_for_scores(query_layer, batch_size,
			num_attention_heads, from_seq_length,
			size_per_head)
	
	# 同上，key也变成[8, 12, 128, 64]
	key_layer = transpose_for_scores(key_layer, batch_size, num_attention_heads,
			to_seq_length, size_per_head)
	
	# 计算query和key的内积，得到attention scores.
	# [8, 12, 128, 64]*[8, 12, 64, 128]=[8, 12, 128, 128]
	# 最后两维[128, 128]表示from的128个时刻attend to到to的128个score。
	# `attention_scores` = [B, N, F, T]
	attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
	attention_scores = tf.multiply(attention_scores,
			1.0 / math.sqrt(float(size_per_head)))
	
	if attention_mask is not None:
		# 从[8, 128, 128]变成[8, 1, 128, 128]
		# `attention_mask` = [B, 1, F, T]
		attention_mask = tf.expand_dims(attention_mask, axis=[1])
	
		# 这个小技巧前面也用到过，如果mask是1，那么(1-1)*-10000=0，adder就是0,
		# 如果mask是0，那么(1-0)*-10000=-10000。
		adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -10000.0
		
		# 我们把adder加到attention_score里，mask是1就相当于加0，mask是0就相当于加-10000。
		# 通常attention_score都不会很大，因此mask为0就相当于把attention_score设置为负无穷
		# 后面softmax的时候就趋近于0，因此相当于不能attend to Mask为0的地方。
		attention_scores += adder
	
	# softmax
	# `attention_probs` = [B, N, F, T] =[8, 12, 128, 128]
	attention_probs = tf.nn.softmax(attention_scores)
	
	# 对attention_probs进行dropout，这虽然有点奇怪，但是Transformer的原始论文就是这么干的。
	attention_probs = dropout(attention_probs, attention_probs_dropout_prob)
	
	# 把`value_layer` reshape成[B, T, N, H]=[8, 128, 12, 64]
	value_layer = tf.reshape(
		value_layer,
		[batch_size, to_seq_length, num_attention_heads, size_per_head])
	
	# `value_layer`变成[B, N, T, H]=[8, 12, 128, 64]
	value_layer = tf.transpose(value_layer, [0, 2, 1, 3])
	
	# 计算`context_layer` = [8, 12, 128, 128]*[8, 12, 128, 64]=[8, 12, 128, 64]=[B, N, F, H]
	context_layer = tf.matmul(attention_probs, value_layer)
	
	# `context_layer` 变换成 [B, F, N, H]=[8, 128, 12, 64]
	context_layer = tf.transpose(context_layer, [0, 2, 1, 3])
	
	if do_return_2d_tensor:
		# `context_layer` = [B*F, N*V]
		context_layer = tf.reshape(
			context_layer,
			[batch_size * from_seq_length, num_attention_heads * size_per_head])
	else:
		# `context_layer` = [B, F, N*V]
		context_layer = tf.reshape(
			context_layer,
			[batch_size, from_seq_length, num_attention_heads * size_per_head])
	
	return context_layer
```



### modeling.py

modeling.py定义了BERT模型的主体结构，即从`input_ids（句子中词语id组成的tensor）`到`sequence_output（句子中每个词语的向量表示）`以及`pooled_output（句子的向量表示）`的计算过程，是其它所有后续的任务的基础。如文本分类任务就是得到输入的input_ids后，用BertModel得到句子的向量表示，并将其作为分类层的输入，得到分类结果。

modeling.py的31-106行定义了一个BertConfig类，即BertModel的配置，在新建一个BertModel类时，必须配置其对应的BertConfig。BertConfig类包含了一个BertModel所需的超参数，除词表大小vocab_size外，均定义了其默认取值。BertConfig类中还定义了从python dict和json中生成BertConfig的方法以及将BertConfig转换为python dict 或者json字符串的方法。

  107-263行定义了一个BertModel类。BertModel类初始化时，需要填写三个没有默认值的参数：

- config：即31-106行定义的BertConfig类的一个对象；
- is_training：如果训练则填true，否则填false，该参数会决定是否执行dropout。
- input_ids：一个`[batch_size, seq_length]`的tensor，包含了一个batch的输入句子中的词语id。

  另外还有input_mask，token_type_ids和use_one_hot_embeddings，scope四个可选参数，scope参数会影响计算图中tensor的名字前缀，如不填写，则前缀为”bert”。在下文中，其余参数会在使用时进行说明。

  BertModel的计算都在`__init__`函数中完成。计算流程如下：

1. 为了不影响原config对象，对config进行deepcopy，然后对is_training进行判断，如果为False，则将config中dropout的概率均设为0。
2. 定义input_mask和token_type_ids的默认取值（前者为全1，后者为全0），shape均和input_ids相同。二者的用途会在下文中提及。
3. 使用embedding_lookup函数，将input_ids转化为向量，形状为`[batch_size, seq_length, embedding_size]`，这里的embedding_table使用tf.get_variable，因此第一次调用时会生成，后续都是直接获取现有的。此处use_one_hot_embedding的取值只影响embedding_lookup函数的内部实现，不影响结果。
4. 调用embedding_postprocessor对输入句子的向量进行处理。这个函数分为两部分，先按照token_type_id（即输入的句子中各个词语的type，如对两个句子的分类任务，用type_id区分第一个句子还是第二个句子），lookup出各个词语的type向量，然后加到各个词语的向量表示中。如果token_type_id不存在（即不使用额外的type信息），则跳过这一步。其次，这个函数计算position_embedding：即初始化一个shape为`[max_positition_embeddings, width]`的position_embedding矩阵，再按照对应的position加到输入句子的向量表示中。如果不使用position_embedding，则跳过这一步。最后对输入句子的向量进行layer_norm和dropout，如果不是训练阶段，此处dropout概率为0.0，相当于跳过这一步。
5. 根据输入的input_mask（即与句子真实长度匹配的mask，如batch_size为2，句子实际长度分别为2，3，则mask为`[[1, 1, 0], [1, 1, 1]]`），计算shape为`[batch_size, seq_length, seq_length]`的mask，并将输入句子的向量表示和mask共同传给transformer_model函数，即encoder部分。
6. transformer_model函数的行为是先将输入的句子向量表示reshape成`[batch_size * seq_length, width]`的矩阵，然后循环调用transformer的前向过程，次数为隐藏层个数。每次前向过程都包含self_attention_layer、add_and_norm、feed_forward和add_and_norm四个步骤，具体信息可参考transformer的论文。
7. 获取transformer_model最后一层的输出，此时shape为`[batch_size, seq_length, hidden_size]`。如果要进行句子级别的任务，如句子分类，需要将其转化为`[batch_size, hidden_size]`的tensor，这一步通过取第一个token的向量表示完成。这一层在代码中称为pooling层。
8. BertModel类提供了接口来获取不同层的输出，包括：
   - embedding层的输出，shape为`[batch_size, seq_length, embedding_size]`
   - pooling层的输出，shape为`[batch_size, hidden_size]`
   - sequence层的输出，shape为`[batch_size, seq_length, hidden_size]`
   - encoder各层的输出
   - embedding_table

  modeling.py的其余部分定义了上面的步骤用到的函数，以及激活函数等。

### run_classifier.py

  这个模块可以用于配置和启动基于BERT的文本分类任务，包括输入样本为句子对的（如MRPC）和输入样本为单个句子的（如CoLA）。

模块中的内容包括：

- InputExample类。一个输入样本包含id，text_a，text_b和label四个属性，text_a和text_b分别表示第一个句子和第二个句子，因此text_b是可选的。
- PaddingInputExample类。定义这个类是因为TPU只支持固定大小的batch，在eval和predict的时候需要对batch做padding。**如不使用TPU，则无需使用这个类。**
- InputFeatures类，定义了输入到estimator的model_fn中的feature，包括input_ids，input_mask，segment_ids（即0或1，表明词语属于第一个句子还是第二个句子，在BertModel中被看作token_type_id），label_id以及is_real_example。
- DataProcessor类以及四个公开数据集对应的子类。一个数据集对应一个DataProcessor子类，需要继承四个函数：分别从文件目录中获得train，eval和predict样本的三个函数以及一个获取label集合的函数。**如果需要在自己的数据集上进行finetune，则需要实现一个DataProcessor的子类，按照自己数据集的格式从目录中获取样本。**注意！在这一步骤中，对没有label的predict样本，要指定一个label的默认值供统一的model_fn使用。
- convert_single_example函数。可以对一个InputExample转换为InputFeatures，里面调用了tokenizer进行一些句子清洗和预处理工作，同时截断了长度超过最大值的句子。
- file_based_convert_example_to_features函数：将一批InputExample转换为InputFeatures，并写入到tfrecord文件中，相当于实现了从原始数据集文件到tfrecord文件的转换。
- file_based_input_fn_builder函数：这个函数用于根据tfrecord文件，构建estimator的input_fn，即先建立一个TFRecordDataset，然后进行shuffle，repeat，decode和batch操作。
- create_model函数：用于构建从input_ids到prediction和loss的计算过程，包括建立BertModel，获取BertModel的pooled_output，即句子向量表示，然后构建隐藏层和bias，并计算logits和softmax，最终用cross_entropy计算出loss。
- model_fn_builder：根据create_model函数，构建estimator的model_fn。由于model_fn需要labels输入，为简化代码减少判断，当要进行predict时也要求传入label，因此DataProcessor中为每个predict样本生成了一个默认label（其取值并无意义）。这里构建的是TPUEstimator，但没有TPU时，它也可以像普通estimator一样工作。
- input_fn_builder和convert_examples_to_features目前并没有被使用，应为开放供开发者使用的功能。
- main函数：
  - 首先定义任务名称和processor的对应关系，**因此如果定义了自己的processor，需要将其加入到processors字典中**。
  - 其次从FLAGS中，即启动命令中读取相关参数，构建model_fn和estimator，并根据参数中的do_train，do_eval和do_predict的取值决定要进行estimator的哪些操作。

### run_pretraining.py

  这个模块用于BERT模型的预训练，即使用masked language model和next sentence的方法，对BERT模型本身的参数进行训练。如果使用现有的预训练BERT模型在文本分类/问题回答等任务上进行fine_tune，则无需使用run_pretraining.py。

用法：

```python
python run_pretraining.py \
	--input_file=/tmp/tf_examples.tfrecord \
	--output_dir=/tmp/pretraining_output \
	--do_train=True \
	--do_eval=True \
	--bert_config_file=$BERT_BASE_DIR/bert_config.json \
	--init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
	--train_batch_size=32 \
	--max_seq_length=128 \
	--max_predictions_per_seq=20 \
	--num_train_steps=20 \
	--num_warmup_steps=10 \
	--learning_rate=2e-5
```



run_pretraining.py的代码和run_classifier.py很类似，都是用BertModel构建Transformer模型，唯一的区别在于损失函数不同

```
def model_fn(features, labels, mode, params):  
  input_ids = features["input_ids"]
  input_mask = features["input_mask"]
  segment_ids = features["segment_ids"]
  masked_lm_positions = features["masked_lm_positions"]
  masked_lm_ids = features["masked_lm_ids"]
  masked_lm_weights = features["masked_lm_weights"]
  next_sentence_labels = features["next_sentence_labels"]
  
  is_training = (mode == tf.estimator.ModeKeys.TRAIN)
  
  model = modeling.BertModel(
		  config=bert_config,
		  is_training=is_training,
		  input_ids=input_ids,
		  input_mask=input_mask,
		  token_type_ids=segment_ids,
		  use_one_hot_embeddings=use_one_hot_embeddings)
  
  (masked_lm_loss,
  masked_lm_example_loss, masked_lm_log_probs) = get_masked_lm_output(
		  bert_config, model.get_sequence_output(), model.get_embedding_table(),
		  masked_lm_positions, masked_lm_ids, masked_lm_weights)
  
  (next_sentence_loss, next_sentence_example_loss,
  next_sentence_log_probs) = get_next_sentence_output(
		  bert_config, model.get_pooled_output(), next_sentence_labels)
  
  total_loss = masked_lm_loss + next_sentence_loss
```

get_masked_lm_output函数用于计算语言模型的Loss(Mask位置预测的词和真实的词是否相同)。

get_next_sentence_output函数用于计算预测下一个句子的loss.



### create_pretraining_data.py

  此处定义了如何将普通文本转换成可用于预训练BERT模型的tfrecord文件的方法。如果使用现有的预训练BERT模型在文本分类/问题回答等任务上进行fine_tune，则无需使用create_pretraining_data.py。

用法：

```
python create_pretraining_data.py \
	--input_file=./sample_text.txt \
	--output_file=/tmp/tf_examples.tfrecord \
	--vocab_file=$BERT_BASE_DIR/vocab.txt \
	--do_lower_case=True \
	--max_seq_length=128 \
	--max_predictions_per_seq=20 \
	--masked_lm_prob=0.15 \
	--random_seed=12345 \
	--dupe_factor=5
```

- max_seq_length Token序列的最大长度
- max_predictions_per_seq 最多生成多少个MASK
- masked_lm_prob 多少比例的Token变成MASK
- dupe_factor 一个文档重复多少次

首先说一下参数dupe_factor，比如一个句子”it is a good day”，为了充分利用数据，我们可以多次随机的生成MASK，比如第一次可能生成”it is a [MASK] day”，第二次可能生成”it [MASK] a good day”。这个参数控制重复的次数。

masked_lm_prob就是论文里的参数15%。max_predictions_per_seq是一个序列最多MASK多少个Token，它通常等于max_seq_length * masked_lm_prob。这么看起来这个参数没有必要提供，但是后面的脚本也需要用到这个同样的值，而后面的脚本并没有这两个参数。

main函数很简单，输入文本文件列表是input_files，通过函数create_training_instances构建训练的instances，然后调用write_instance_to_example_files以TFRecord格式写到output_files。

训练样本的格式，这是用类TrainingInstance来表示的：

```
class TrainingInstance(object):
	def __init__(self, tokens, segment_ids, masked_lm_positions, masked_lm_labels,
				is_random_next):
		self.tokens = tokens
		self.segment_ids = segment_ids
		self.is_random_next = is_random_next
		self.masked_lm_positions = masked_lm_positions
		self.masked_lm_labels = masked_lm_labels
```

is_random_next表示这两句话是有关联的，预测句子关系的分类器应该把这个输入判断为1。masked_lm_positions记录哪些位置被Mask了，而masked_lm_labels记录被Mask之前的词。

create_training_instances函数会调用create_instances_from_document来从一个文档里抽取多个训练数据(TrainingInstance)。普通的语言模型只要求连续的字符串就行，通常是把所有的文本(比如维基百科的内容)拼接成一个很大很大的文本文件，然后训练的时候随机的从里面抽取固定长度的字符串作为一个”句子”。但是BERT要求我们的输入是一个一个的Document，每个Document有很多句子，这些句子是连贯的真实的句子，需要正确的分句，而不能随机的(比如按照固定长度)切分句子。

代码有点长，但是逻辑很简单，比如有一篇文档有n个句子：

```
w11,w12,.....,
w21,w22,....
wn1,wn2,....
```

那么算法首先找到一个chunk，它会不断往chunk加入一个句子的所有Token，使得chunk里的token数量大于等于target_seq_length。通常我们期望target_seq_length为max_num_tokens(128-3)，这样padding的尽量少，训练的效率高。但是有时候我们也需要生成一些短的序列，否则会出现训练与实际使用不匹配的问题。

找到一个chunk之后，比如这个chunk有5个句子，那么我们随机的选择一个切分点，比如3。把前3个句子当成句子A，后两个句子当成句子B。这是两个句子A和B有关系的样本(is_random_next=False)。为了生成无关系的样本，我们还以50%的概率把B用随机从其它文档抽取的句子替换掉，这样就得到无关系的样本(is_random_next=True)。如果是这种情况，后面两个句子需要放回去，以便在下一层循环中能够被再次利用。

有了句子A和B之后，我们就可以填充tokens和segment_ids，这里会加入特殊的[CLS]和[SEP]。接下来使用create_masked_lm_predictions来随机的选择某些Token，把它变成[MASK]。

最后是使用函数write_instance_to_example_files把前面得到的TrainingInstance用TFRecord的个数写到文件里。

### tokenization.py

  此处定义了对输入的句子进行预处理的操作，预处理的内容包括：

- 转换为Unicode
- 切分成数组
- 去除控制字符
- 统一空格格式
- 切分中文字符（即给连续的中文字符之间加上空格）
- 将英文单词切分成小片段（如[“unaffable”]切分为[“un”, “##aff”, “##able”]）
- 大小写和特殊形式字母转换
- 分离标点符号（如 [“hello?”]转换为 [“hello”, “?”]）

### run_squad.py

  这个模块可以配置和启动基于BERT在squad数据集上的问题回答任务。

### extract_features.py

  这个模块可以使用预训练的BERT模型，生成输入句子的向量表示和输入句子中各个词语的向量表示（类似ELMo）。**这个模块不包含训练的过程，只是执行BERT的前向过程，使用固定的参数对输入句子进行转换**。

### optimization.py

  这个模块配置了用于BERT的optimizer，即加入weight decay功能和learning_rate warmup功能的AdamOptimizer。



### Self-Attention(torch)

BERT 模型对 Self-Attention 的实现代码片段：

```python
# 取自 hugging face 团队实现的基于 pytorch 的 BERT 模型
class BERTSelfAttention(nn.Module):
    # BERT 的 Self-Attention 类
    def __init__(self, config):
        # 初始化函数
        super(BERTSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

    def transpose_for_scores(self, x):
        # 调整维度，转换为 (batch_size, num_attention_heads, hidden_size, attention_head_size)
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        # 前向传播函数
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer) 
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # 将"query"和"key"点乘，得到未经处理注意力值
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # 使用 softmax 函数将注意力值标准化成概率值
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer
```

参照Transformer 的结构，在 Multi-Head Attention 之后是 Add & Norm，将经过注意力机制计算后的向量和原输入相加并归一化，进入 Feed Forward Neural Network，然后再进行一次和输入的相加并完成归一化。

## 2. 在自己的数据集上finetune

  BERT官方项目搭建了文本分类模型的model_fn，因此只需定义自己的DataProcessor，即可在自己的文本分类数据集上进行训练。

  训练自己的文本分类数据集所需步骤如下：

1.下载预训练的BERT模型参数文件，如(https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip )，解压后的目录应包含`bert_config.json`，`bert_model.ckpt.data-00000-of-00001`，`bert_model.ckpt.index`，`bert_model_ckpt.meta`和`vocab.txt`五个文件。

2.将自己的数据集统一放到一个目录下。为简便起见，事先将其划分成train.txt，eval.txt和predict.txt三个文件，每个文件中每行为一个样本，格式如下（可以使用任何自定义格式，只需要编写符合要求的DataProcessor子类即可）：

```js
simplistic , silly and tedious . __label__0
```

即句子和标签之间用__label__划分，句子中的词语之间用空格划分。

3.修改`run_classifier.py`，或者复制一个副本，命名为`run_custom_classifier.py`或类似文件名后进行修改。

4.新建一个DataProcessor的子类，并继承三个get_examples方法和一个get_labels方法。三个get_examples方法需要从数据集目录中获得各自对应的InputExample列表。以get_train_examples方法为例，该方法需要传入唯一的一个参数data_dir，即数据集所在目录，然后根据该目录读取训练数据，将所有用于训练的句子转换为InputExample，并返回所有InputExample组成的列表。get_dev_examples和get_test_examples方法同理。get_labels方法仅需返回一个所有label的集合组成的列表即可。本例中get_train_examples方法和get_labels方法的实现如下（此处省略get_dev_examples和get_test_examples）：

```js
class RtPolarityProcessor(DataProcessor):
    """Processor of the rt-polarity data set"""

    @staticmethod
    def read_raw_text(input_file):
        with tf.gfile.Open(input_file, "r") as f:
            lines = f.readlines()
            return lines

    def get_train_examples(self, data_dir):
        """See base class"""
        lines = self.read_raw_text(os.path.join(data_dir, "train.txt"))
        examples = []
        for i, line in enumerate(lines):
            guid = "train-%d" % (i + 1)
            line = line.strip().split("__label__")
            text_a = tokenization.convert_to_unicode(line[0])
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, label=label)
            )
        return examples
    
    def get_labels(self):
        return ["0", "1"]
```

5.在main函数中，向main函数开头的processors字典增加一项，key为自己的数据集的名称，value为上一步中定义的DataProcessor的类名：

```js
processors = {
    "cola": ColaProcessor,
    "mnli": MnliProcessor,
    "mrpc": MrpcProcessor,
    "xnli": XnliProcessor,
    "rt_polarity": RtPolarityProcessor,
}
```

6.执行python run_custom_classifier.py，启动命令中包含必填参数data_dir，task_name，vocab_file，bert_config_file，output_dir。参数do_train，do_eval和do_predict分别控制了是否进行训练，评估和预测，可以按需将其设置为True或者False，但至少要有一项设为True。

7.为了从预训练的checkpoint开始finetune，启动命令中还需要配置init_checkpoint参数。假设BERT模型参数文件解压后的路径为`/uncased_L-12_H-768_A-12`，则将init_checkpoint参数配置为`/uncased_L-12_H-768_A-12/bert_model.ckpt`。其它可选参数，如learning_rate等，可参考文件中FLAGS的定义自行配置或使用默认值。

8.在没有TPU的情况下，即使使用了GPU，这一步有可能会在日志中看到`Running train on CPU`字样。对此，官方项目的readme中做出了解释：”Note: You might see a message `Running train on CPU`. This really just means that it’s running on something other than a Cloud TPU, which includes a GPU. “，**因此无需在意**。

  如果需要训练文本分类之外的模型，如命名实体识别，BERT的官方项目中没有完整的demo，因此需要设计和实现自己的model_fn和input_fn。以命名实体识别为例，model_fn的基本思路是，根据输入句子的input_ids生成一个BertModel，获得BertModel的sequence_output（shape为`[batch_size，max_length，hidden_size]`），再结合全连接层和crf等函数进行序列标注。

## 参考资料

http://fancyerii.github.io/2019/03/09/bert-codes/

[【技术分享】BERT系列（一）——BERT源码分析及使用方法](https://cloud.tencent.com/developer/article/1454853?from=10680)


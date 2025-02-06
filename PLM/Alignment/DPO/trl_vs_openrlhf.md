---
title: trl_vs_openrlhf
created: 2025-02-05
tags:
  - 代码对比
---
# main函数

trl
	transformers.AutoModelForCausalLM加载model
	支持precompute_ref_log_probs，此时不需要加载model_ref。需要调用get_train_dataloader函数计算出reference_chosen_logps和reference_rejected_logps，并且存储到硬盘
	加载数据集，可将处理结果存储到硬盘
	DPOTrainer.train()
	一些重要参数
	- 默认是cosine学习率

train_dataloader
`get_train_dataloader`
- 通过调用concatenated_forward函数, 计算reference_chosen_logps和reference_rejected_logps。然后再调用父函数。
```python
# pad_sequence()函数进行pad, prompt会进行左pad，其他进行右pad
data_collator = DPODataCollatorWithPadding(
                pad_token_id=tokenizer.pad_token_id,
                label_pad_token_id=label_pad_token_id,
                is_encoder_decoder=self.is_encoder_decoder,
            )
dataloader_params = {
            "batch_size": self._train_batch_size,  # torch.cuda.device_count()*per_device_batch_size
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }
if not isinstance(train_dataset, torch.utils.data.IterableDataset):
	dataloader_params["sampler"] = self._get_train_sampler() # 等价于RandomSampler(self.train_dataset)
	dataloader_params["drop_last"] = self.args.dataloader_drop_last
	dataloader_params["worker_init_fn"] = seed_worker
	dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor
return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))
```

openrlhf
	定义了一个DeepspeedStrategy，管理数据集、模型、优化器等
	定义了一个Actor类，加载model和ref_model，两者的ds_config不同
	加载数据集
	cosine_with_min_lr学习率
	DPOTrainer.fit()
	一些重要参数
	- micro_train_batch_size每个GPU的batch size
	- train_batch_size全局batch size
	- ring_attn_size和ring_head_stride默认1
	- 是cosine_with_min_lr学习率
```python
num_update_steps_per_epoch = len(train_dataset) // args.train_batch_size
max_steps = math.ceil(args.max_epochs * num_update_steps_per_epoch)
```

train_dataloader
```python
train_dataloader = strategy.setup_dataloader(
        replay_buffer=train_dataset,
        batch_size=args.micro_train_batch_size,
        pin_memory=True,
        shuffle=True,
        collate_fn=train_dataset.packing_collate_fn if args.packing_samples else train_dataset.collate_fn,
        drop_last=True,
        sampler=None,
        consumed_samples=0,
    )


if sampler is None:
	num_replicas = dist.get_world_size() // self.ring_attn_size
	rank = dist.get_rank() // self.ring_attn_size
	sampler = DistributedSampler(
		replay_buffer,
		num_replicas=num_replicas,
		rank=rank,
		shuffle=shuffle,
		seed=self.seed,
		drop_last=drop_last,
		consumed_samples=consumed_samples,
	)

return DataLoader(
	replay_buffer,
	batch_size=batch_size,
	sampler=sampler,
	drop_last=drop_last,
	collate_fn=collate_fn,
	pin_memory=pin_memory,
)
```

# 训练函数

## trl
继承Trainer
`__init__()`
- 把数据tokenize之后存储到硬盘，下次可直接加载；prepare_deepspeed(ref_model)
`train()`
- 调用transformers中的Trainer所定义的train()函数, 重写get_train_dataloader函数，evaluation_loop函数，prediction_step函数，compute_loss函数等
`tokenize_row()` 
- 对prompt分词得到prompt_tokens，调用build_tokenized_answer得到chosen_tokens，rejected_tokens（都是add_special_tokens=False）。
-  `build_tokenized_answer` 分别对prompt + answer和prompt进行tokenize，通过比较prompt_input_ids和full_tokenized的结果，计算出response_token_ids_start_idx，取full_tokenized结果中idx之前的和之后的作为prompt和answer的分词结果
- 如果tokenizer.add_bos_token，那么在prompt_tokens前面加上bos_token_id。在chosen_tokens和rejected_tokens末尾加上eos_token_id（但是之前处理数据的时候也加了个end_of_sentence_token，有重复了）
- 如果prompt+更长的response超出长度限制，默认keep_end方式保留prompt_input_ids的后面一截。如果还超，那么保留response的前面一截。
- 把chosen_tokens和rejected_tokens的prompt和response拼起来得到input_ids和attention_mask，labels的prompt_input_ids部分用label_pad。
- 最终得到一个batch，有chosen_input_ids，rejected_input_ids，prompt_input_ids（以及attention_mask，labels）。如果有chosen_score，chosen_point也会放进去。max_length记录prompt+longer_response的长度。
`concatenated_forward`
- 先右pad把chosen_input_ids,chosen_labels,chosen_attention_mask（以及对应的rejected）等pad到同一长度，chosen和rejected拼接起来，合并后的输入序列形状为 (batch_size * 2, max_length)
- 使用合并后的输入序列执行模型的前向传播，得到模型的原始输出 logits。logits 的形状是(batch_size * 2, max_length, vocab_size)

`evaluation_loop`: 生成结果，打印日志
`prediction_step`: 主要是调用get_batch_loss_metrics得到loss和metrics，再做一些变换。

`compute_loss`
- 调用`get_batch_loss_metrics`. concatenated_forward得到policy_chosen/rejected_logps，policy_chosen/rejected_logits,reference_chosen/rejected_logps, 等。计算sft_loss和dpo_loss，最终`loss=dpo_loss.mean()+sft_weight*sft_loss+aux_loss*self.aux_loss_coef`
- `dpo_loss`返回losses, chosen_rewards, rejected_rewards。sigmoid loss：
```python
# original code
pi_logratios = policy_chosen_logps - policy_rejected_logps
if reference_free:  # 默认False
	ref_logratios = 0
else:
	ref_logratios = reference_chosen_logps - reference_rejected_logps

logits = pi_logratios - ref_logratios
losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )
chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()
return losses, chosen_rewards, rejected_rewards
```
- `sft_loss`: 根据policy_chosen_logits, chosen_labels计算得到
```python
def sft_loss(logits, labels):
	labels = labels[:, 1:].clone()
	logits = logits[:, :-1, :]
	loss_mask = labels != label_pad_token_id

	# dummy token; we'll ignore the losses on these tokens later
	labels[labels == label_pad_token_id] = 0

	# 根据给定的 `labels` 提取每个 token 对应的对数概率（log probability）
	per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
	all_logps = (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
	return -all_logps.mean()
```

### 详细解释

这段代码的目的是从 `logits` 中根据给定的 `labels` 提取每个 token 对应的对数概率（log probability）。下面是对这段代码的逐步解释：

`per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)`

逐步解析：

1. `logits.log_softmax(-1)`

- `logits`：通常是模型的输出，形状为 `(batch_size, seq_len, vocab_size)`，其中：
	
	- `batch_size` 是批次大小。
	- `seq_len` 是序列长度（例如，文本中的词数）。
	- `vocab_size` 是词汇表的大小（即模型输出的每个 token 的分类数）。
- `logits.log_softmax(-1)`：
	
	- `log_softmax` 是对 `logits` 进行 LogSoftmax 操作，计算每个 token 在每个类别上的对数概率。
	- `log_softmax(-1)` 作用于最后一个维度（即 `vocab_size` 维度），结果的形状为 `(batch_size, seq_len, vocab_size)`。
	- 这相当于对每个 token 对应的 logits 应用 Softmax 激活函数后再取对数。

2. `labels.unsqueeze(2)`

- `labels`：通常是目标标签，形状为 `(batch_size, seq_len)`，每个元素表示每个 token 的正确类别索引。
	
- `labels.unsqueeze(2)`：
	
	- `unsqueeze(2)` 会在 `labels` 的第 2 维插入一个新的维度（即在 `vocab_size` 维度之前）。
	- 结果的形状变为 `(batch_size, seq_len, 1)`，意味着每个标签将与其对应的 logits 张量进行匹配。

3. `torch.gather(..., dim=2, index=labels.unsqueeze(2))`

- `torch.gather(input, dim, index)`：从 `input` 张量中按照 `index` 给定的索引提取值。
	
	- `input` 是经过 `log_softmax` 处理的 logits，形状为 `(batch_size, seq_len, vocab_size)`。
		
	- `dim=2` 表示我们从最后一维（`vocab_size`）中选择对应的值。
		
	- `index=labels.unsqueeze(2)` 指定了要选择的索引位置。`labels` 的每个元素表示该 token 在该序列中的正确类别索引，`unsqueeze(2)` 增加了一个维度，使得它的形状变为 `(batch_size, seq_len, 1)`。
		
	- 这一步的作用是，从 `logits.log_softmax(-1)` 的结果中，对于每个 token，提取出它对应的对数概率值。例如，对于第 `i` 个样本，第 `j` 个位置，它会选择 `logits[i, j, labels[i, j]]` 的值。
		
- `torch.gather` 的结果是一个形状为 `(batch_size, seq_len, 1)` 的张量，表示每个 token 对应的对数概率。
	

4. `.squeeze(2)`

- `.squeeze(2)`：
	- `squeeze` 会移除张量中维度为 1 的维度。
	- 在这里，`squeeze(2)` 会去掉第 2 维（即 `vocab_size` 维度），因为它的大小为 1。最终，结果的形状变为 `(batch_size, seq_len)`，即每个 token 的对数概率值。

总结：这段代码的目的是根据给定的 `labels`，从 `logits` 中提取每个 token 对应的对数概率（log probability）。最终的输出 `per_token_logps` 的形状是 `(batch_size, seq_len)`，其中每个元素表示对应 token 的对数概率。


## openrlhf
只继承ABC基类
`__init__`
- 支持packing_samples，有个nll_loss
`fit`

`DPOLoss`和trl是一样的
```python
pi_logratios = policy_chosen_logps - policy_rejected_logps
ref_logratios = reference_chosen_logps - reference_rejected_logps
logits = pi_logratios - ref_logratios

if self.ipo:
	losses = (logits - 1 / (2 * self.beta)) ** 2  # Eq. 17 of https://arxiv.org/pdf/2310.12036v2.pdf
else:
	# Eq. 3 https://ericmitchell.ai/cdpo.pdf; label_smoothing=0 gives original DPO (Eq. 7 of https://arxiv.org/pdf/2305.18290.pdf)
	losses = (
		-F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
		- F.logsigmoid(-self.beta * logits) * self.label_smoothing
	)

loss = losses.mean()
chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()

return loss, chosen_rewards, rejected_rewards
```

`evaluate`

`concatenated_forward`



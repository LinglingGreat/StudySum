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

openrlhf
	定义了一个DeepspeedStrategy，管理数据集、模型、优化器等
	定义了一个Actor类，加载model和ref_model，两者的ds_config不同
	加载数据集
	cosine_with_min_lr学习率
	DPOTrainer.fit()

# 训练函数

## trl
`train()`
- 调用transformers中的Trainer所定义的train()函数, 重写get_train_dataloader函数，evaluation_loop函数，prediction_step函数，compute_loss函数等
`__init__()`
- 把数据tokenize之后存储到硬盘，下次可直接加载；prepare_deepspeed(ref_model)
`tokenize_row()` 
- 对prompt分词得到prompt_tokens，调用build_tokenized_answer得到chosen_tokens，rejected_tokens（都是add_special_tokens=False）。
-  `build_tokenized_answer` 分别对prompt + answer和prompt进行tokenize，通过比较prompt_input_ids和full_tokenized的结果，计算出response_token_ids_start_idx，取full_tokenized结果中idx之前的和之后的作为prompt和answer的分词结果
- 如果tokenizer.add_bos_token，那么在prompt_tokens前面加上bos_token_id。在chosen_tokens和rejected_tokens末尾加上eos_token_id（但是之前处理数据的时候也加了个end_of_sentence_token，有重复了）
- 如果prompt+更长的response超出长度限制，默认keep_end方式保留prompt_input_ids的后面一截。如果还超，那么保留response的前面一截。
- 把chosen_tokens和rejected_tokens的prompt和response拼起来得到input_ids和attention_mask，labels的prompt_input_ids部分用label_pad。
- 最终得到一个batch，有chosen_input_ids，rejected_input_ids，prompt_input_ids（以及attention_mask，labels）。如果有chosen_score，chosen_point也会放进去。max_length记录prompt+longer_response的长度。
`get_train_dataloader`
- 计算reference_chosen_logps和reference_rejected_logps，通过调用concatenated_forward函数
`concatenated_forward`
- 先右pad把chosen_input_ids,chosen_labels,chosen_attention_mask（以及对应的rejected）等pad到同一长度，chosen和rejected拼接起来，合并后的输入序列形状为 (batch_size * 2, max_length)
- 使用合并后的输入序列执行模型的前向传播，得到模型的原始输出 logits。logits 的形状是(batch_size * 2, max_length, vocab_size)

`compute_loss`
- 调用`get_batch_loss_metrics`
- `dpo_loss`
- `sft_loss`

`evaluation_loop`: 生成结果，打印日志
`prediction_step`: 主要是调用get_batch_loss_metrics得到loss和metrics，再做一些变换。


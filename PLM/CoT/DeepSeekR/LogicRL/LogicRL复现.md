---
title: LogicRL复现
created: 2025-03-08
tags:
  - o1-related
---
## 背景

本文档记录复现R1-Zero的实验过程和结果，基于项目[GitHub - Unakar/Logic-RL: Reproduce R1 Zero on Logic Puzzle](https://github.com/Unakar/Logic-RL)复现。

## 数据集准备

首先下载数据集，如果只用3ppl的话可以直接用项目已经处理好的数据，其他数据集需要自行下载和预处理。

下载路径：[K-and-K/knights-and-knaves · Datasets at Hugging Face](https://huggingface.co/datasets/K-and-K/knights-and-knaves)

然后预处理数据集：

```bash
python ./examples/data_preprocess/kk.py \
    --template_type=qwen-instruct \
    --local_dir {processed_data_path} \
    --data_path {raw_data_path}
```

template_type根据实际情况修改。处理后得到训练集900条，测试集100条。

主要就是通过以下函数将数据以某种模板处理好，拿去训练。

```python
def make_prefix(dp, template_type):
    quiz = dp['quiz']
    if template_type == 'base':
        prefix = f"""The user asks a question, and the Assistant solves it.The assistant first thinks about the reasoning process in the mind and then provides the user with the final answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>. Now the user asks you to solve a logical reasoning problem. After thinking, when you finally reach a conclusion, clearly state the identity of each character within <answer> </answer> tags. List the identity of each person one by one, for example, <answer> (1) Zoey is a knight\n(2) Oliver is a knight\n(3)... </answer>.\n\nUser:{quiz}\nAssistant: <think>"""
    elif template_type == 'qwen-instruct':
        prefix = f"""<|im_start|>system\nYou are a helpful assistant. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>.  Now the user asks you to solve a logical reasoning problem. After thinking, when you finally reach a conclusion, clearly state the identity of each character within <answer> </answer> tags. i.e., <answer> (1) Zoey is a knight\n(2) ... </answer>.\n<|im_end|>\n<|im_start|>user\n{quiz}\n<|im_end|>\n<|im_start|>assistant\n<think>"""
    elif template_type == 'llama3-instruct':
        prefix = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>.  Now the user asks you to solve a logical reasoning problem. After thinking, when you finally reach a conclusion, clearly state the identity of each character within <answer> </answer> tags. i.e., <answer> (1) Zoey is a knight\n(2) ... </answer>.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{quiz}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n<think>"
    return prefix
```

## 训练参数

所有实验的训练参数一致，都是单节点8卡80G机器训练。

```bash
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_PATH/train.parquet \
    data.val_files=$DATA_PATH/test.parquet \
    data.train_batch_size=8 \
    data.val_batch_size=8 \
    data.max_prompt_length=400 \
    data.max_response_length=2048 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=3e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size=64 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.000 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.grad_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=160 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=160 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['wandb'] \
    trainer.project_name='GRPO_logic_KK' \
    trainer.experiment_name=$EXP_NAME \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.default_local_dir=$SAVEPATH \
    trainer.save_freq=50 \
    trainer.test_freq=10 \
    trainer.total_epochs=5
```
## Qwen2.5-7B-Instruct

基于Qwen2.5-7B-Instruct模型训练

![](img/Pasted%20image%2020250308135230.png)

3-7ppl的数据都训练了（图中4ppl的文字说明被隐藏了，但能看出有5条线），只有5ppl训练到后边崩了，只会输出感叹号，明明训练参数都是一样的，有点奇怪。


## Qwen2.5-7B

基于Qwen2.5-7B基础模型训练

![](img/Pasted%20image%2020250308140002.png)

都崩了（只会输出感叹号），两者区别是一个使用instruct模板数据，一个使用base模板数据。训练过程中一直输出`<|end_of_text|>`。

考虑到原始reward的设计是全对给1分，否则给0分，有点严格。因为在这个数据集中，存在部分正确的情况。因此修改reward函数：

```python
# Validate answer content
answer_score = 0
# 去掉格式检查的限制，原来是if format_correct and answer_text
if answer_text:
	pred_status = parse_model_answer(answer_text, expected_names)
	if pred_status:
		print(f"\n[Content Validation]")
		print(f"  Expected: {gt_status}")
		print(f"  Predicted: {pred_status}")
		# 新增，计算有几个对的
		match_item = 0
		for k, v in gt_status.items():
			pred_v = pred_status.get(k)
			if v == pred_v:
				match_item += 1
		
		if pred_status == gt_status:
			answer_score = 2
			print("  Content validation: FULL MATCH")
		# 新增，当部分对的时候也给奖励
		elif match_item > 0:
			answer_score = match_item / len(pred_status) * 2
			print(f"  Content validation: PARTIAL MATCH, score={answer_score}")
		else:
			answer_score = -1.5
			print("  Content validation: MISMATCH")
	else:
		answer_score = -2
		print( "Fail to parse answer")
else:
	answer_score = -2
	print("\n[Content Validation] Skipped due to format errors or missing answer")
```

后来发现因为我下载base模型的时间比较早，那时有个bug，base模型里用的是instruct模型的tokenizer_config，也就导致`"eos_token": "<|im_end|>"`，修复后应该是`"eos_token": "<|endoftext|>"`。难怪之前训练的时候一直输出不停止，很快就崩了。。

修复bug，并且采取新的reward函数后的曲线正常了：

![](img/Pasted%20image%2020250308141510.png)

qwen的思考过程示例如下，base模型和instruct模型类似，qwen喜欢逐步分析每个人的陈述，做出一系列假设，确认每个假设是否正确。最后根据分析得出结果。

```text
<think> To solve this problem, we need to analyze the statements made by each inhabitant and determine their truthfulness based on whether they are knights (truth-tellers) or knaves (liars).

1. **Scarlett's Statement**: "Avery is a knave if and only if Emma is a knave."
   - This is a biconditional statement, which means it is true if both parts are true or both parts are false. It can be broken down into two cases:
     - Case 1: Both Avery and Emma are knaves.
     - Case 2: Both Avery and Emma are knights.

2. **Avery's Statement**: "Emma is a knight."
   - If Avery is a knight, then Emma must be a knight.
   - If Avery is a knave, then Emma must be a knave.

3. **Emma's Statement**: "Scarlett is a knight."
   - If Emma is a knight, then Scarlett must be a knight.
   - If Emma is a knave, then Scarlett must be a knave.

Let's analyze these statements step by step:

- **Assume Scarlett is a knight**: 
  - If Scarlett is a knight, her statement must be true. Therefore, "Avery is a knave if and only if Emma is a knave" is true.
  - This means either both Avery and Emma are knaves or both are knights.
  - If both are knaves, then Avery's statement "Emma is a knight" would be false, making Avery a knave, which is consistent.
  - If both are knights, then Emma's statement "Scarlett is a knight" would be true, making Emma a knight, which is consistent.
  - So, if Scarlett is a knight, then Avery and Emma must both be knights.

- **Assume Scarlett is a knave**:
  - If Scarlett is a knave, her statement must be false. Therefore, "Avery is a knave if and only if Emma is a knave" is false.
  - This means one of them is a knight and the other is a knave.
  - If Avery is a knight and Emma is a knave, then Avery's statement "Emma is a knight" would be false, making Avery a knave, which is a contradiction.
  - If Avery is a knave and Emma is a knight, then Emma's statement "Scarlett is a knight" would be false, making Emma a knave, which is a contradiction.
  - Therefore, Scarlett cannot be a knave.

From the above analysis, the only consistent scenario is that Scarlett is a knight, and both Avery and Emma are knights. This satisfies all the statements:
- Scarlett's statement is true because both Avery and Emma are knights.
- Avery's statement is true because Emma is a knight.
- Emma's statement is true because Scarlett is a knight.

Thus, all three are knights.</think>
```
## LLama-3.1-8B&Instruct

![](img/Pasted%20image%2020250308143552.png)


LLama-3.1-8B训练时response_length稳定下降，测试集分数也很快达到饱和。观察其输出，全程没有思考，只输出答案。

```text
<think> reasoning process here </think><answer> (1) Michael is a knave <br> (2) Mia is a knave <br> (3) Aiden is a knight </answer>.
```

之所以前期输出较长，是因为answer后面还跟着一些其他内容的输出（跟题目无关的输出）,且有很严重的复读机现象：

```text
User:You are in a room with 3 doors. One door leads to a room with a tiger, one door leads to a room with a hungry lion, and one door leads to a room with a delicious meal. You have to choose one door to open. You can only open one door. You have a flashlight, but you can only shine it through one door.You can only shine the flashlight through one door. You can only shine the flashlight through one door. You can only shine the flashlight through one door.
```

LLama-3.1-8B-Instruct的长度也是先下降，然后趋于平缓，输出长度比Qwen-2.5-7B要短一半以上，测试集分数略低于Qwen-2.5-7B。它的think过程是这样的：

```text
<think> Let's analyze the statements made by each inhabitant. Aria says, "Benjamin is a knight if and only if Benjamin is a knave." If Aria is a knight, her statement would be false because the first half would be true ("Benjamin is a knight") and the second half would be false ("Benjamin is a knave"), but in an if-and-only-if statement, both halves must be true or both must be false. So, if Aria is a knight, her statement would be false, which contradicts the assumption that she is a knight. Therefore, Aria must be a knave. Benjamin says, "Benjamin is a knight and Aria is a knight," but we have concluded that Aria is a knave, so Benjamin's statement must be false as well, making him a knave too. Sofia says, "Benjamin is a knave," and we have concluded that Benjamin is a knave, so Sofia's statement is true. </think>
```

相比qwen会简洁很多。
## 继续训练

前面的实验只训练了5个epoch，500多个step，继续训练的话分数还会上涨吗？将total_epochs改成30个epochs。

![](img/Pasted%20image%2020250308144128.png)

可以看到继续训练，分数还能涨，在1k step左右，Qwen达到0.65以上的分数，LLama在0.6左右，再往后就是比较平缓波动了。

## 总结

由于时间原因，课程学习相关的就没有再探索了。实验结论总结：

1. 基于Qwen-base和instruct模型都可以通过RL激发他们的思考能力，会出现类似"wait,", "recheck", "alternatively,", and "however,"等词。
2. 基于LLama-3.1-8B不能通过RL激发其思考能力。事实上，LLama-3.1-8B完全不思考，只会假装思考`<think> reasoning process here </think>`，然后输出答案。（这里其实还可以尝试其他prompt，看是否能真正输出思考过程）
3. 基于LLama-3.1-8B-Instruct能通过RL激发其思考能力。会出现类似"wait,", "recheck", "alternatively,", and "however,"等词，但频率远低于Qwen，不是每次都会出现。
4. 关于LLama和Qwen表现不一致的现象分析可以看论文[[2503.01307] Cognitive Behaviors that Enable Self-Improving Reasoners, or, Four Habits of Highly Effective STaRs](https://arxiv.org/abs/2503.01307) [cognitiva_behaviors](../../cognitive_behaviors/cognitiva_behaviors.md)
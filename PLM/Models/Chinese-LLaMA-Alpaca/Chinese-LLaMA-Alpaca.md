---
title: Chinese-LLaMA-Alpaca
created: 2023-06-08
tags: 增量预训练, LLM, SFT

---

https://github.com/ymcui/Chinese-LLaMA-Alpaca

## 基本情况

为了促进大模型在中文NLP社区的开放研究，本项目开源了**中文LLaMA模型和指令精调的Alpaca大模型**。这些模型**在原版LLaMA的基础上扩充了中文词表**并使用了中文数据进行二次预训练，进一步提升了中文基础语义理解能力。同时，中文Alpaca模型进一步使用了中文指令数据进行精调，显著提升了模型对指令的理解和执行能力。详细内容请参考技术报告[(Cui, Yang, and Yao, 2023)](https://arxiv.org/abs/2304.08177)。

**本项目主要内容：**

- 🚀 针对原版LLaMA模型扩充了中文词表，提升了中文编解码效率
- 🚀 开源了使用中文文本数据预训练的中文LLaMA以及经过指令精调的中文Alpaca
- 🚀 开源了预训练脚本、指令精调脚本，用户可根据需要自行进一步训练
- 🚀 快速使用笔记本电脑（个人PC）的CPU/GPU本地量化和部署体验大模型
- 🚀 支持[🤗transformers](https://github.com/huggingface/transformers), [llama.cpp](https://github.com/ggerganov/llama.cpp), [text-generation-webui](https://github.com/oobabooga/text-generation-webui), [LlamaChat](https://github.com/alexrozanski/LlamaChat), [LangChain](https://github.com/hwchase17/langchain), [privateGPT](https://github.com/imartinez/privateGPT)等生态
- 目前已开源的模型版本：7B（标准版、**Plus版**）、13B（标准版、**Plus版**）

|对比项|中文LLaMA|中文Alpaca|
|:--|---|---|
|训练方式|传统CLM|指令精调|
|训练语料|无标注通用语料|有标注指令数据|
|词表大小[3]|4995**3**|4995**4**=49953+1（pad token）|
|输入模板|不需要|需要符合模板要求[1]|
|适用场景 ✔️|文本续写：给定上文内容，让模型继续写下去，生成下文|1、指令理解（问答、写作、建议等）  <br>2、多轮上下文理解（聊天等）|
|不适用场景 ❌|指令理解 、多轮聊天等|文本无限制自由生成|
|llama.cpp|使用`-p`参数指定上文|使用`-ins`参数启动指令理解+聊天模式|
|text-generation-webui|不适合chat模式|使用`--cpu`可在无显卡形式下运行，若生成内容不满意，建议修改prompt|
|LlamaChat|加载模型时选择"LLaMA"|加载模型时选择"Alpaca"|
|[HF推理代码](https://github.com/ymcui/Chinese-LLaMA-Alpaca/blob/main/scripts/inference_hf.py)|无需添加额外启动参数|启动时添加参数 `--with_prompt`|
|[web-demo代码](https://github.com/ymcui/Chinese-LLaMA-Alpaca/blob/main/scripts/gradio_demo.py)|不适用|直接提供Alpaca模型位置即可；支持多轮对话|
|[LangChain示例](https://github.com/ymcui/Chinese-LLaMA-Alpaca/blob/main/scripts/langchain_demo) / privateGPT|不适用|直接提供Alpaca模型位置即可|
|已知问题|如果不控制终止，则会一直写下去，直到达到输出长度上限。[2]|目前版本模型生成的文本长度相对短一些，比较惜字如金。可在指令中要求详细回答。[2]|

## 增量预训练

词表扩充



数据



脚本

```bash
########参数设置########
lr=2e-4
lora_rank=8
lora_alpha=32
lora_trainable="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj"
modules_to_save="embed_tokens,lm_head"
lora_dropout=0.05

pretrained_model=path/to/hf/llama/dir
chinese_tokenizer_path=path/to/chinese/llama/tokenizer/dir
dataset_dir=path/to/pt/data/dir
data_cache=temp_data_cache_dir
per_device_train_batch_size=1
per_device_eval_batch_size=1
training_steps=100
gradient_accumulation_steps=1
output_dir=output_dir

deepspeed_config_file=ds_zero2_no_offload.json

########启动命令########
torchrun --nnodes 1 --nproc_per_node 1 run_clm_pt_with_peft.py \
    --deepspeed ${deepspeed_config_file} \
    --model_name_or_path ${pretrained_model} \
    --tokenizer_name_or_path ${chinese_tokenizer_path} \
    --dataset_dir ${dataset_dir} \
    --data_cache_dir ${data_cache} \
    --validation_split_percentage 0.001 \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size ${per_device_eval_batch_size} \
    --do_train \
    --seed $RANDOM \
    --fp16 \
    --max_steps ${training_steps} \
    --lr_scheduler_type cosine \
    --learning_rate ${lr} \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy steps \
    --save_total_limit 3 \
    --save_steps 500 \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --preprocessing_num_workers 8 \
    --block_size 512 \
    --output_dir ${output_dir} \
    --overwrite_output_dir \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --lora_rank ${lora_rank} \
    --lora_alpha ${lora_alpha} \
    --trainable ${lora_trainable} \
    --modules_to_save ${modules_to_save} \
    --lora_dropout ${lora_dropout} \
    --torch_dtype float16 \
    --gradient_checkpointing \
    --ddp_find_unused_parameters False
```


## cuda OOM问题
`from_pretrained`时发生
- 加上参数torch_dtype=torch.float16
- 使用accelerate, 加上参数revision="sharded", device_map="auto"

## 下载工具
huggingface-cli download xai-org/grok-1 --repo-type model --include ckpt/tensor* --local-dir checkpoints/ckpt-0 --local-dir-use-symlinks False --resume-download 

如果要resume，local-dir-use-symlinks就不能设置成False

export HF_ENDPOINT=https://hf-mirror.com

export HF_HUB_ENABLE_HF_TRANSFER=1

huggingface-cli download HuggingFaceH4/ultrachat_200k --repo-type dataset

参考 [Command Line Interface (CLI)](https://huggingface.co/docs/huggingface_hub/en/guides/cli)

[如何快速下载huggingface模型——全方法总结](https://zhuanlan.zhihu.com/p/663712983)


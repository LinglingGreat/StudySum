## cuda OOM问题
`from_pretrained`时发生
- 加上参数torch_dtype=torch.float16
- 使用accelerate, 加上参数revision="sharded", device_map="auto"

## 下载工具
huggingface-cli download xai-org/grok-1 --repo-type model --include ckpt/tensor* --local-dir checkpoints/ckpt-0 --local-dir-use-symlinks False --resume-download 


## cuda OOM问题
`from_pretrained`时发生
- 加上参数torch_dtype=torch.float16
- 使用accelerate, 加上参数revision="sharded", device_map="auto"
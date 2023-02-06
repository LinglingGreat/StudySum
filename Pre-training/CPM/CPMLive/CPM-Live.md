## CPM-Ant


### finetune

输入数据格式
```
{"input": "", "target": ""}
```

最大值=prompt_length+input_length+target_length

target是预测目标

一行一个样本，长度超过最大值的时候，会优先截断input，其次截断target（从后往前）



task_id默认是1，

prompt_length
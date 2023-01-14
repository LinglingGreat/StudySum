### deepspeed

[https://huggingface.co/docs/transformers/main_classes/deepspeed](https://huggingface.co/docs/transformers/main_classes/deepspeed)

用deepspeed先把gcc和g++升到5以上

暂时升级：`scl enable devtoolset-7 bash`

**如何指定卡：**

-   看~/soft/miniconda3/envs/ll_eva/lib/python3.8/site-packages/deepspeed/launcher/runner.py的参数--include，在sh里加上`--include dll.cenbrain.club:0,3 --hostfile ${HOST_FILE}`。hostfile加上`dll.cenbrain.club slots=4`。看主机名`hostname -I`

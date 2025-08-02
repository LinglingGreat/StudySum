## 安装环境

[Miniconda — Anaconda documentation](https://docs.anaconda.com/free/miniconda/index.html)

![](img/Pasted%20image%2020240312103257.png)

读取多个路径下的envs，在~/.bashrc文件末尾添加一行`export CONDA_ENVS_PATH=/xxx/miniconda3/envs:/xxx/miniconda3/envs`

```bash
echo 'export CONDA_ENVS_PATH=/path/to/miniconda3/envs:/another/path/to/miniconda3/envs' >> ~/.bashrc
source ~/.bashrc
```

## 常用命令

查询现有环境：`conda info --env`

创建虚拟环境：`conda create -n your_env_name python=X.X`

复制环境：`conda create -n 新环境名 --clone 旧环境名`

激活环境：`conda activate 环境名`

激活环境后可进行pip install

删除虚拟环境：`conda remove -n your_env_name --all`

重新安装原先版本的anaconda，保留虚拟环境：`bash Anaconda3-5.2.0-Linux-x86_64.sh -u`

指定目录创建环境 `conda create -p ~/miniconda3/envs/test_pretrain --clone ll_pretrain`

## 源设置

```
conda list --show-channel-urls
conda config --show channels  # 查看当前频道顺序
conda config --add channels pytorch
conda config --add channels nvidia
conda config --add channels conda-forge
conda config --add channels bioconda
conda config --add channels intel

```

## 环境迁移

导出环境： `conda env export > environment.yml`

重现环境： `conda env create -f environment.yml`

## 安装gcc

```bash
conda install -c conda-forge gcc
conda install -c conda-forge gxx
```

`export LD_LIBRARY_PATH=~/miniconda3/envs/env_name/lib/:$LD_LIBRARY_PATH`

## flash-attention

```
# Uninstall existing package
pip uninstall -y flash-attn
```

`pip install flash-attn==2.7.3 --no-build-isolation`

- - **`--no-build-isolation`**: Ensures the build uses your environment's PyTorch.
- import flash_attn_2_cuda as flash_attn_gpu报错undefined symbol的时候可以尝试这个方法

## 参考资料

[https://blog.csdn.net/weixin_41010198/article/details/106188880](https://blog.csdn.net/weixin_41010198/article/details/106188880)


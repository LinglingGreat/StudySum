## 常用命令

查询现有环境：`conda info --env`

创建虚拟环境：`conda create -n your_env_name python=X.X`

复制环境：`conda create -n 新环境名 --clone 旧环境名`

激活环境：`conda activate 环境名`

激活环境后可进行pip install

删除虚拟环境：`conda remove -n your_env_name --all`

重新安装原先版本的anaconda，保留虚拟环境：`bash Anaconda3-5.2.0-Linux-x86_64.sh -u`

## 环境迁移

导出环境： `conda env export > environment.yml`

重现环境： `conda env create -f environment.yml`

## 参考资料

[https://blog.csdn.net/weixin_41010198/article/details/106188880](https://blog.csdn.net/weixin_41010198/article/details/106188880)


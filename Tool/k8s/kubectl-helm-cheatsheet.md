# kubectl / helm 使用说明

> kubectl 安装路径：~/.local/bin/kubectl
> helm 安装路径：~/.local/bin/helm
> kubeconfig：~/.kube/config（默认）/ config-admin（管理员权限，用 k-admin 别名）

---

## kubectl 常用命令

### 别名

```bash
kubectl         # 默认 kubeconfig
k-admin         # 等价于 kubectl --kubeconfig=$HOME/.kube/config-admin
k-nodes         # 查看各节点 GPU 情况（grep nvidia.com/gpu）
```

### 查看资源

```bash
kubectl get nodes                       # 列出所有节点
kubectl get pods                        # 列出当前 namespace 的 Pod
kubectl get pods -A                     # 列出所有 namespace 的 Pod
kubectl get pods -n <namespace>         # 指定 namespace
kubectl get svc                         # 列出 Service
kubectl get deploy                      # 列出 Deployment
kubectl describe pod <pod名>            # 查看 Pod 详情（含事件）
kubectl describe node <节点名>          # 查看节点详情（含资源用量）
```

### 查看 GPU 资源

```bash
k-nodes         # 快速查看各节点 GPU 情况
kubectl describe nodes | grep -A5 "Allocated resources"   # 查看节点资源分配
kubectl get pods -A -o wide | grep <节点名>               # 查看节点上的 Pod
```

### 日志与调试

```bash
kubectl logs <pod名>                    # 查看 Pod 日志
kubectl logs <pod名> -f                 # 实时跟踪日志
kubectl logs <pod名> --previous         # 查看上一次运行的日志（容器重启后看前一次）
kubectl logs <pod名> -c <容器名>        # 多容器时指定容器
kubectl logs <pod名> -c <容器名> --previous --tail=50  # 看崩溃容器的最后50行
kubectl exec -it <pod名> -- bash        # 进入容器
kubectl exec -it <pod名> -c <容器名> -- bash  # 多容器时指定容器
kubectl exec <pod名> -c <容器名> -- curl -s http://localhost:8000/health  # 从容器内部测接口
kubectl port-forward <pod名> 8080:80    # 端口转发（本机不在集群内时用）
```

### 故障排查

```bash
# --- Pod 级别 ---
kubectl describe pod <pod名> -n <ns>    # 看 Events 段，定位 ImagePull/OOM/FailedMount 等问题
kubectl get pods -n <ns> -o wide        # 看 Pod 调度到了哪个节点

# --- Deployment / ReplicaSet 级别 ---
kubectl describe deploy <名称> -n <ns>  # 看 Conditions（Available/Progressing/ReplicaFailure）
kubectl describe rs -n <ns> -l <label>  # 看 ReplicaSet 事件，定位 FailedCreate 原因
kubectl get events -n <ns> --sort-by=.lastTimestamp | grep <关键词>  # 按时间排序看集群事件

# --- Service / 网络 ---
kubectl get endpoints <svc名> -n <ns>   # 看 Service 后面实际绑定了哪些 Pod IP
kubectl get svc -n <ns>                 # 看 Service 的 ClusterIP 和端口

# --- 容器启动命令确认 ---
kubectl exec <pod名> -c <容器名> -- cat /proc/1/cmdline | tr '\0' ' '  # 看实际执行的命令和参数

# --- 常见状态速查 ---
# Init:ImagePullBackOff  → init 容器镜像拉取失败，检查 registry 地址和网络
# CrashLoopBackOff       → 容器启动后崩溃，用 --previous 看日志找报错
# Pending                → 资源不足或无匹配节点，describe pod 看 Events
# FailedCreate           → ReplicaSet 无法创建 Pod，describe rs 看原因
#                          常见：PriorityClass 不存在、资源配额超限、GPU 不足
# ImagePullBackOff       → 镜像不存在或 registry 不通，describe pod 看 Events
# Error                  → 容器异常退出，看日志
```

### 应用管理

```bash
kubectl apply -f manifest.yaml          # 应用配置文件
kubectl delete -f manifest.yaml         # 删除配置文件定义的资源
kubectl delete pod <pod名>              # 删除 Pod（会自动重建）
kubectl scale deploy <名称> --replicas=3  # 扩缩容
kubectl rollout restart deploy <名称>   # 滚动重启
kubectl rollout status deploy <名称>    # 查看发布状态
```

### 查看资源用量（需要 metrics-server）

```bash
kubectl top nodes                       # 节点 CPU/内存用量
kubectl top pods                        # Pod CPU/内存用量
```

---

## helm 常用命令

### 查看已安装的 chart

```bash
helm list                               # 列出当前 namespace 的 release
helm list -A                            # 列出所有 namespace
helm status <release名>                 # 查看 release 状态
helm history <release名>                # 查看发布历史
```

### 安装与更新

```bash
helm install <release名> <chart>        # 安装
helm install myapp ./chart -f values.yaml  # 指定 values 安装
helm upgrade <release名> <chart>        # 升级
helm upgrade --install <release名> <chart>  # 不存在则安装，存在则升级
helm uninstall <release名>              # 卸载
```

### 调试与排查

```bash
helm template <release名> <chart>       # 渲染模板（不实际部署，用于 debug）
helm template <release名> <chart> -f override.yaml  # 用 override values 渲染
helm lint <chart目录>                   # 检查 chart 语法
helm get values <release名>             # 查看当前 release 的 values（含合并后的结果）
helm get manifest <release名>           # 查看实际部署的 manifest
helm get manifest <release名> | grep <关键词>  # 快速搜已部署的 manifest 内容

# --set 注意事项：逗号是列表分隔符，传含逗号的字符串需转义
helm install x . --set key="a\,b"       # 正确：值为 "a,b"
helm install x . --set key="a,b"        # 错误：被解析为数组 [a, b]
# 推荐用 -f values.yaml 而非 --set 传复杂值
```

### 仓库管理

```bash
helm repo add stable https://charts.helm.sh/stable
helm repo update                        # 更新所有仓库
helm search repo <关键词>               # 搜索 chart
```

---

## K8s Service 核心概念

### 请求地址 vs 路由目标

Service 有两个独立的维度：

| 维度 | 由什么决定 | 规则 |
|------|-----------|------|
| **请求地址（DNS）** | `metadata.name` | `<name>.<namespace>.svc.cluster.local:<port>`，同 namespace 简写为 `<name>:<port>` |
| **路由到哪些 Pod** | `spec.selector` | 匹配 Pod 的 label，所有匹配的 Pod 都会被轮询 |

```yaml
apiVersion: v1
kind: Service
metadata:
  name: my-service          # → 请求地址: http://my-service:8000
spec:
  selector:
    service-group: my-app   # → 路由到所有带 service-group=my-app 标签的 Pod
  ports:
    - port: 8000
```

### 关键规则

- 同 namespace 下 Service `metadata.name` 必须唯一（不能有两个同名 Service）
- 多个 Service 的 `spec.selector` 可以相同（都指向同一组 Pod，只是 DNS 名不同）
- Pod 被哪些 Service 选中取决于 Pod 的 label，不取决于 Pod 属于哪个 Deployment

### 典型场景：多 Deployment 共享流量

两个 Deployment 的 Pod 各自有不同的 `app` 标签（保证 Deployment 独立管控），但共享一个额外标签让同一个 Service 选中：

```yaml
# Deployment A 的 Pod labels
app: my-app-a                    # Deployment matchLabels 用这个（唯一）
service-group: my-app            # 共享 Service 用这个选 Pod

# Deployment B 的 Pod labels
app: my-app-b                    # Deployment matchLabels 用这个（唯一）
service-group: my-app            # 共享 Service 用这个选 Pod

# 共享 Service
selector:
  service-group: my-app          # 同时选中 A 和 B 的 Pod
```

请求 `my-service:8000` → K8s 轮询分发到 A 和 B 的 Pod。

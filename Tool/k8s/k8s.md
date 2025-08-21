# 安装

https://github.com/vllm-project/production-stack/blob/main/tutorials/00-install-kubernetes-env.md


# 查看事件

#### 检查集群事件

```Bash
kubectl get events --sort-by='.lastTimestamp'
kubectl get events --sort-by=.metadata.creationTimestamp | grep vip
```

#### 查看可用gpu

```bash
kubectl describe nodes | grep -E "nvidia.com/gpu|HolderIdentity:" 
kubectl describe nodes | grep -E "nvidia.com/gpu|name:"
```

kubectl describe nodes | grep -A 10 "Allocated resources"

# 服务启动

```Markdown
kubectl apply -f model.yaml
```

查看部署了哪些服务deployment/pod：

```bash
kubectl get deployment
kubectl get pods -o wide

kubectl get pods -n 命名空间 -o wide
## 所有空间
kubectl get pods -A
```

```
# 显示 Pod 的详细信息，包括事件历史、容器状态等
kubectl describe pod your_pod_name

# scaledobject名字是config里面的名字 
kubectl get scaledobject 
kubectl describe scaledobject <name> 

# hpa的名字是自动启的，每次查看一下 
kubectl get hpa 
kubectl describe hpa <your hpa name>
```

查看pod启动后的log

```Markdown
kubectl logs -f your_pod_name
```

进入pod内部

```Plain
kubectl exec -it <pod-name> -- /bin/bash
```

删除服务，谨慎使用：
- 注意，使用deployment启动的服务，不要直接删除pod，删除了会自动重启。**需要用`delete deployment <name>`进行删除

```Bash
kubectl delete deployment  <name>
```

# 滚动重启

1. 滚动重启 Deployment:（**deployment**名字叫做：xx-name）
    

```Bash
kubectl rollout restart deployment/xx-name
```

2. 扩展 Deployment 到 5 个副本:
    

```Bash
kubectl scale deployment/xx-name --replicas=5
```

4. 查看 Deployment 的滚动更新状态:
    

```Bash
kubectl rollout status deployment/xx-name
```

5. 如果需要，回滚到之前的版本:
    

```Bash
kubectl rollout undo deployment/xx-name
```

6. 暂停滚动更新（如果你需要暂停正在进行的更新）:
    

```Bash
kubectl rollout pause deployment/xx-name
```

7. 恢复已暂停的滚动更新:
    

```Bash
kubectl rollout resume deployment/xx-name
```

Deployment 的详细信息:

```Bash
kubectl describe deployment xx-name
```

9. 编辑 Deployment 配置:
    

```Bash
kubectl edit deployment joyland-dialogue
```

查看 Deployment 的历史版本:

```Bash
kubectl rollout history deployment/joyland-dialogue
```

# 监控

```bash
# 使用下面的命令，可以在本地访问promethus里面的监控数据 
kubectl port-forward -n monitoring svc/prometheus-k8s 9090:9090 

# grafana监控转发 
kubectl port-forward svc/grafana -n monitoring 8082:3000
```

# 节点故障迁移

要将 Kubernetes 集群中一个故障节点上的服务迁移到另一个节点，可以按照以下步骤操作：
    

3. 将故障节点标记为不可调度：
    
    ```Plain
    kubectl cordon xxxx
    ```
    
      这会防止新的 Pod 被调度到这个节点上。
  

4. 驱逐故障节点上的 Pod：
    
    ```Plain
    kubectl drain xxxx --ignore-daemonsets --delete-emptydir-data
    ```
    
      这个命令会将节点上的 Pod 安全地驱逐到其他可用节点。
    

5. 如果某些 Pod 无法自动迁移（例如，没有控制器管理的 Pod），你可能需要手动删除它们：
    
    ```Plain
    kubectl delete pod <pod-name> --force --grace-period=0
    ```
    


6. 检查 Pod 是否已经在其他节点上重新创建：
    
    ```Plain
    kubectl get pods -o wide
    ```
    

7. 如果有些 Pod 没有自动重新创建，检查它们的部署或 StatefulSet：
    
    ```Plain
    kubectl get deployments
    kubectl get statefulsets
    ```
    
      可能需要手动扩展或重新创建这些资源。


8. 如果故障节点已经修复，可以重新加入集群：
    
    ```Plain
    kubectl uncordon xxxx
    ```
    

  

9. 如果故障节点需要永久移除，可以从集群中删除它：
    
    ```Plain
    kubectl delete node 192.168.125.15
    ```
    

  

注意事项：

- 确保集群中有足够的资源来容纳被迁移的 Pod。
    
- 对于有状态应用，可能需要额外的数据迁移步骤。
    
- 某些 DaemonSet 管理的 Pod 可能无法被驱逐，这是正常的。
    
- 在生产环境中执行这些操作时要格外小心，最好先在测试环境中演练。
    

  

通过这些步骤，你应该能够将服务从故障节点迁移到集群中的其他健康节点。

# LimitRange 与 ResourceQuota 的区别

您问的这个问题非常好，这是 Kubernetes 资源管理的两个核心概念。让我详细解释它们的区别：

## 核心区别概述

| 特性 | LimitRange | ResourceQuota |
|------|------------|---------------|
| **作用范围** | 单个容器/Pod级别 | 整个命名空间级别 |
| **主要目的** | 设置默认值和单个容器的限制 | 设置命名空间资源总量上限 |
| **约束对象** | 单个容器或Pod的资源使用 | 整个命名空间的资源消耗总和 |
| **默认值** | 可以为容器提供默认requests/limits | 不提供默认值，只设上限 |

## 详细解释

### 1. LimitRange（限制范围）

您的配置示例：
```yaml
apiVersion: v1
kind: LimitRange
metadata:
  name: container-resource-limit-range
  namespace: voices
spec:
  limits:
    - type: Container
      default:
        memory: "100Gi"
        cpu: "16"
        ephemeral-storage: "3Gi"
      defaultRequest:
        memory: "1Gi"
        cpu: "1"
        ephemeral-storage: "1Gi"
      max:
        memory: "100Gi"
        cpu: "16"
        ephemeral-storage: "3Gi"
```

**LimitRange 的作用：**
- 为**单个容器**设置资源限制
- 提供默认的 requests 和 limits 值（当容器未指定时）
- 限制单个容器能使用的最大资源量
- 确保每个容器都有合理的基础资源保障

### 2. ResourceQuota（资源配额）

典型示例：
```yaml
apiVersion: v1
kind: ResourceQuota
metadata:
  name: namespace-resource-quota
  namespace: voices
spec:
  hard:
    requests.cpu: "32"
    requests.memory: 128Gi
    limits.cpu: "64"
    limits.memory: 256Gi
    requests.nvidia.com/gpu: "8"
    limits.nvidia.com/gpu: "8"
    pods: "20"
```

**ResourceQuota 的作用：**
- 限制**整个命名空间**的资源使用总量
- 防止一个命名空间占用过多集群资源
- 可以限制Pod数量、PVC数量等
- 不提供默认值，只设上限

## 实际工作场景中的配合使用

### 场景：机器学习训练命名空间

```yaml
# 1. 首先设置ResourceQuota - 命名空间总限制
apiVersion: v1
kind: ResourceQuota
metadata:
  name: training-resource-quota
  namespace: voices
spec:
  hard:
    requests.cpu: "64"
    requests.memory: 256Gi
    limits.cpu: "128"
    limits.memory: 512Gi
    requests.nvidia.com/gpu: "16"
    limits.nvidia.com/gpu: "16"
    pods: "10"
---
# 2. 然后设置LimitRange - 单个容器限制
apiVersion: v1
kind: LimitRange
metadata:
  name: training-limit-range
  namespace: voices
spec:
  limits:
    - type: Container
      default:
        memory: "32Gi"
        cpu: "8"
      defaultRequest:
        memory: "16Gi"
        cpu: "4"
      max:
        memory: "64Gi"
        cpu: "16"
      min:
        memory: "4Gi"
        cpu: "1"
    - type: Pod
      max:
        memory: "128Gi"
        cpu: "32"
```

## 为什么需要两者配合使用？

### 没有 LimitRange 的问题：
- 开发者可能忘记为容器设置资源请求和限制
- 导致资源分配不均衡，有些容器占用过多资源

### 没有 ResourceQuota 的问题：
- 一个命名空间可能占用所有集群资源
- 影响其他命名空间的正常工作

## 检查资源限制

您可以使用以下命令检查当前限制：

```bash
# 查看命名空间的ResourceQuota
kubectl describe resourcequota -n voices

# 查看LimitRange
kubectl describe limitrange -n voices

# 查看资源使用情况
kubectl describe namespace voices
```

## 您配置的具体含义

您提供的 LimitRange 配置表示：

1. **默认值**：如果容器没有指定资源限制，会自动获得：
   - CPU: 16核的限制，1核的请求
   - 内存: 100Gi的限制，1Gi的请求
   - 临时存储: 3Gi的限制，1Gi的请求

2. **最大值**：任何容器都不能超过：
   - CPU: 16核
   - 内存: 100Gi
   - 临时存储: 3Gi

## 建议调整

对于 GPU 训练任务，您可能需要调整这些值：

```yaml
spec:
  limits:
    - type: Container
      default:
        memory: "32Gi"    # 更合理的默认值
        cpu: "8"          # 更合理的默认值
        nvidia.com/gpu: "2"  # 添加GPU默认值
      defaultRequest:
        memory: "16Gi"
        cpu: "4"
        nvidia.com/gpu: "1"
      max:
        memory: "64Gi"    # 根据实际需求调整
        cpu: "16"
        nvidia.com/gpu: "4"  # 添加GPU最大值
        ephemeral-storage: "50Gi"  # 训练可能需要更多临时存储
```

## 总结

- **LimitRange**：管理单个容器的资源，确保每个容器都有适当的资源限制
- **ResourceQuota**：管理整个命名空间的资源，防止命名空间过度消耗集群资源

两者配合使用可以实现精细化的资源管理：LimitRange 确保"每个容器都合理"，ResourceQuota 确保"所有容器加起来不超过总量"。

对于您的训练任务，建议同时使用两者来确保资源既充足又不会过度占用集群资源。
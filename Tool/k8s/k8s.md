

### 查看事件

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

### 服务启动

```Markdown
kubectl apply -f model.yaml
```

查看部署了哪些服务deployment/pod：

```Markdown
kubectl get deployment
kubectl get pods -o wide
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

### 滚动重启

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

```bash
# 使用下面的命令，可以在本地访问promethus里面的监控数据 
kubectl port-forward -n monitoring svc/prometheus-k8s 9090:9090 

# grafana监控转发 
kubectl port-forward svc/grafana -n monitoring 8082:3000
```

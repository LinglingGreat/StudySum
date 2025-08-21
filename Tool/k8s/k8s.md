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
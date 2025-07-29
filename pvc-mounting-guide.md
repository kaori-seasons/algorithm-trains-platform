# PVC挂载指南

## 概述

在Kubernetes环境中，PVC（Persistent Volume Claim）的挂载有两种主要模式：

1. **Pod挂载模式（pod_mount）**：在Pod中通过volumeMounts挂载PVC（推荐）
2. **宿主机挂载模式（host_mount）**：直接在宿主机上挂载PVC（需要特殊权限）

## 1. Pod挂载模式（推荐）

### 1.1 基本概念

Pod挂载模式是Kubernetes的标准做法，PVC通过Pod的volumeMounts配置挂载到容器内的指定路径。

### 1.2 使用示例

```python
from storage_providers import StorageManagerFactory

# 创建PVC提供者
pvc_provider = StorageManagerFactory.create_provider('pvc')

# 设置挂载模式为Pod挂载
pvc_provider.set_mount_mode('pod_mount')

# 挂载PVC
mount_config = {
    'pvc_name': 'training-data-pvc',
    'namespace': 'default',
    'mount_path': '/mnt/training-data',
    'storage_class': 'standard',
    'access_modes': ['ReadWriteOnce'],
    'storage_size': '10Gi',
    'create_if_not_exists': True,
    'mount_mode': 'pod_mount'  # 明确指定Pod挂载模式
}

result = pvc_provider.mount(mount_config)
print(f"挂载结果: {result}")
```

### 1.3 在Pod配置中使用

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: training-pod
  namespace: default
spec:
  containers:
    - name: training-container
      image: train-platform/training:latest
      volumeMounts:
        - name: training-data
          mountPath: /mnt/training-data
          subPath: user-data  # 可选，子路径
          readOnly: false     # 可选，是否只读
  volumes:
    - name: training-data
      persistentVolumeClaim:
        claimName: training-data-pvc
```

### 1.4 在Deployment中使用

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: training-deployment
  namespace: default
spec:
  replicas: 1
  selector:
    matchLabels:
      app: training
  template:
    metadata:
      labels:
        app: training
    spec:
      containers:
        - name: training-container
          image: train-platform/training:latest
          volumeMounts:
            - name: training-data
              mountPath: /mnt/training-data
      volumes:
        - name: training-data
          persistentVolumeClaim:
            claimName: training-data-pvc
```

## 2. 宿主机挂载模式

### 2.1 基本概念

宿主机挂载模式允许直接在宿主机上挂载PVC，适用于需要在宿主机上直接访问存储数据的场景。

### 2.2 支持的PV类型

1. **hostPath**：主机路径类型
2. **NFS**：网络文件系统类型
3. **AWS EBS**：AWS弹性块存储
4. **其他类型**：通过kubectl cp命令复制数据

### 2.3 使用示例

```python
# 设置挂载模式为宿主机挂载
pvc_provider.set_mount_mode('host_mount')

# 挂载PVC到宿主机
mount_config = {
    'pvc_name': 'training-data-pvc',
    'namespace': 'default',
    'mount_path': '/mnt/training-data',
    'storage_class': 'standard',
    'access_modes': ['ReadWriteOnce'],
    'storage_size': '10Gi',
    'create_if_not_exists': True,
    'mount_mode': 'host_mount'  # 明确指定宿主机挂载模式
}

result = pvc_provider.mount(mount_config)
print(f"宿主机挂载结果: {result}")
```

### 2.4 不同PV类型的挂载方式

#### 2.4.1 hostPath类型

```python
# hostPath类型的PV会自动创建符号链接
mount_config = {
    'pvc_name': 'hostpath-pvc',
    'namespace': 'default',
    'mount_path': '/mnt/hostpath-data',
    'mount_mode': 'host_mount'
}

result = pvc_provider.mount(mount_config)
# 结果：在/mnt/hostpath-data创建指向实际主机路径的符号链接
```

#### 2.4.2 NFS类型

```python
# NFS类型的PV会执行mount命令
mount_config = {
    'pvc_name': 'nfs-pvc',
    'namespace': 'default',
    'mount_path': '/mnt/nfs-data',
    'mount_mode': 'host_mount'
}

result = pvc_provider.mount(mount_config)
# 结果：执行 mount -t nfs nfs-server:/path /mnt/nfs-data
```

#### 2.4.3 AWS EBS类型

```python
# EBS类型的PV会挂载块设备
mount_config = {
    'pvc_name': 'ebs-pvc',
    'namespace': 'default',
    'mount_path': '/mnt/ebs-data',
    'mount_mode': 'host_mount'
}

result = pvc_provider.mount(mount_config)
# 结果：挂载EBS卷到/mnt/ebs-data
```

#### 2.4.4 其他类型（kubectl cp方式）

```python
# 对于不支持的PV类型，使用kubectl cp复制数据
mount_config = {
    'pvc_name': 'other-pvc',
    'namespace': 'default',
    'mount_path': '/mnt/other-data',
    'mount_mode': 'host_mount'
}

result = pvc_provider.mount(mount_config)
# 结果：创建临时Pod，使用kubectl cp复制数据到/mnt/other-data
```

## 3. 挂载模式选择指南

### 3.1 选择Pod挂载模式的情况

- ✅ **推荐场景**：大多数Kubernetes应用
- ✅ **容器化应用**：在容器内访问数据
- ✅ **多副本部署**：需要数据共享
- ✅ **安全性要求高**：数据隔离在容器内
- ✅ **标准Kubernetes实践**：遵循最佳实践

### 3.2 选择宿主机挂载模式的情况

- ⚠️ **特殊场景**：需要宿主机直接访问
- ⚠️ **性能要求**：需要最高I/O性能
- ⚠️ **调试需求**：需要直接检查文件
- ⚠️ **外部工具集成**：需要与宿主机工具集成
- ⚠️ **权限要求**：需要root权限

## 4. 实际应用示例

### 4.1 训练平台中的使用

```python
class TrainingPlatform:
    def __init__(self):
        self.pvc_provider = StorageManagerFactory.create_provider('pvc')
        self.pvc_provider.set_mount_mode('pod_mount')  # 推荐使用Pod挂载
    
    def setup_training_environment(self, user_id: str):
        """设置训练环境"""
        
        # 为用户创建训练数据PVC
        training_data_config = {
            'pvc_name': f'training-data-{user_id}',
            'namespace': 'train-platform',
            'mount_path': f'/mnt/training-data/{user_id}',
            'storage_class': 'fast-ssd',
            'access_modes': ['ReadWriteOnce'],
            'storage_size': '50Gi',
            'create_if_not_exists': True,
            'labels': {
                'app': 'train-platform',
                'user': user_id,
                'type': 'training-data'
            }
        }
        
        result = self.pvc_provider.mount(training_data_config)
        
        if result['success']:
            print(f"训练数据PVC挂载成功: {result['mount_path']}")
            return result
        else:
            raise Exception(f"训练数据PVC挂载失败: {result['error']}")
    
    def setup_model_storage(self, user_id: str):
        """设置模型存储"""
        
        model_storage_config = {
            'pvc_name': f'model-storage-{user_id}',
            'namespace': 'train-platform',
            'mount_path': f'/mnt/models/{user_id}',
            'storage_class': 'standard',
            'access_modes': ['ReadWriteOnce'],
            'storage_size': '20Gi',
            'create_if_not_exists': True,
            'labels': {
                'app': 'train-platform',
                'user': user_id,
                'type': 'model-storage'
            }
        }
        
        result = self.pvc_provider.mount(model_storage_config)
        
        if result['success']:
            print(f"模型存储PVC挂载成功: {result['mount_path']}")
            return result
        else:
            raise Exception(f"模型存储PVC挂载失败: {result['error']}")
```

### 4.2 Pipeline任务中的使用

```python
class PipelineTask:
    def __init__(self):
        self.pvc_provider = StorageManagerFactory.create_provider('pvc')
    
    def execute_task(self, task_config: Dict[str, Any]):
        """执行Pipeline任务"""
        
        # 挂载输入数据
        input_mounts = []
        for input_data in task_config.get('inputs', []):
            mount_result = self.pvc_provider.mount(input_data)
            if mount_result['success']:
                input_mounts.append(mount_result)
            else:
                raise Exception(f"输入数据挂载失败: {mount_result['error']}")
        
        # 挂载输出存储
        output_mounts = []
        for output_data in task_config.get('outputs', []):
            mount_result = self.pvc_provider.mount(output_data)
            if mount_result['success']:
                output_mounts.append(mount_result)
            else:
                raise Exception(f"输出存储挂载失败: {mount_result['error']}")
        
        try:
            # 执行任务逻辑
            self._run_task_logic(task_config, input_mounts, output_mounts)
        finally:
            # 清理挂载
            for mount in input_mounts + output_mounts:
                self.pvc_provider.unmount(mount['mount_path'])
    
    def _run_task_logic(self, task_config: Dict[str, Any], input_mounts: List, output_mounts: List):
        """执行具体的任务逻辑"""
        # 在这里实现具体的训练或数据处理逻辑
        pass
```

## 5. 错误处理和故障排除

### 5.1 常见错误

#### 5.1.1 PVC未绑定

```python
# 错误：PVC未绑定
try:
    result = pvc_provider.mount(mount_config)
    if not result['success']:
        print(f"挂载失败: {result['error']}")
except Exception as e:
    print(f"挂载异常: {e}")
```

**解决方案**：
- 检查存储类是否可用
- 检查集群是否有足够的存储资源
- 检查PVC的访问模式是否合适

#### 5.1.2 权限不足

```python
# 错误：宿主机挂载权限不足
# 解决方案：使用Pod挂载模式或提升权限
pvc_provider.set_mount_mode('pod_mount')
```

#### 5.1.3 挂载点被占用

```python
# 错误：挂载点已存在
# 解决方案：清理挂载点或使用不同的路径
import shutil
if os.path.exists('/mnt/training-data'):
    shutil.rmtree('/mnt/training-data')
```

### 5.2 调试技巧

```python
# 检查PVC状态
pvc_status = pvc_provider.get_pvc_status('training-data-pvc', 'default')
print(f"PVC状态: {pvc_status}")

# 检查挂载信息
mount_info = pvc_provider.get_info('/mnt/training-data')
print(f"挂载信息: {mount_info}")

# 检查健康状态
health = pvc_provider.check_health()
print(f"健康状态: {health}")
```

## 6. 最佳实践

### 6.1 命名规范

```python
# 使用有意义的PVC名称
pvc_name = f"{project}-{environment}-{data_type}-{user_id}"
# 示例：train-platform-prod-training-data-user123
```

### 6.2 标签管理

```python
# 使用标签进行分类管理
labels = {
    'app': 'train-platform',
    'environment': 'production',
    'data-type': 'training-data',
    'user': user_id,
    'created-by': 'pipeline'
}
```

### 6.3 资源管理

```python
# 根据数据大小选择合适的存储类
if data_size_gb > 100:
    storage_class = 'fast-ssd'  # 高性能存储
else:
    storage_class = 'standard'  # 标准存储
```

### 6.4 生命周期管理

```python
# 定期清理不需要的PVC
def cleanup_expired_pvcs():
    pvcs = pvc_provider.list_pvcs('train-platform')
    for pvc in pvcs:
        if is_expired(pvc):
            pvc_provider.delete_pvc(pvc['name'], pvc['namespace'])
```

这个指南详细说明了PVC的两种挂载模式，帮助您根据实际需求选择合适的挂载方式。 
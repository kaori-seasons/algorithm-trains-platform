# 存储提供者使用示例

## 1. PVCProvider 使用示例

### 1.1 基本使用

```python
from storage_providers import StorageManagerFactory

# 创建PVC提供者
pvc_provider = StorageManagerFactory.create_provider('pvc')

# 挂载PVC
mount_config = {
    'pvc_name': 'training-data-pvc',
    'namespace': 'default',
    'mount_path': '/mnt/training-data',
    'storage_class': 'standard',
    'access_modes': ['ReadWriteOnce'],
    'storage_size': '10Gi',
    'create_if_not_exists': True,
    'labels': {'app': 'train-platform'},
    'annotations': {'description': 'Training data storage'}
}

result = pvc_provider.mount(mount_config)
print(f"挂载结果: {result}")
```

### 1.2 创建PVC

```python
# 创建PVC配置
pvc_config = {
    'name': 'training-data-pvc',
    'namespace': 'default',
    'storage_class': 'fast-ssd',
    'access_modes': ['ReadWriteOnce'],
    'storage_size': '50Gi',
    'labels': {
        'app': 'train-platform',
        'environment': 'production'
    },
    'annotations': {
        'description': 'High-performance training data storage',
        'owner': 'ml-team'
    }
}

# 创建PVC
pvc_name = pvc_provider.create_pvc(pvc_config)
print(f"创建的PVC名称: {pvc_name}")
```

### 1.3 获取PVC状态

```python
# 获取PVC状态
pvc_status = pvc_provider.get_pvc_status('training-data-pvc', 'default')
print(f"PVC状态: {pvc_status}")

# 输出示例:
# {
#     'name': 'training-data-pvc',
#     'namespace': 'default',
#     'phase': 'Bound',
#     'access_modes': ['ReadWriteOnce'],
#     'capacity': {'storage': '50Gi'},
#     'volume_name': 'pvc-12345678-1234-1234-1234-123456789012',
#     'storage_class': 'fast-ssd',
#     'labels': {'app': 'train-platform'},
#     'annotations': {'description': 'Training data storage'}
# }
```

### 1.4 调整PVC大小

```python
# 调整PVC大小
success = pvc_provider.resize_pvc('training-data-pvc', '100Gi', 'default')
if success:
    print("PVC大小调整成功")
else:
    print("PVC大小调整失败")
```

### 1.5 创建快照

```python
# 创建PVC快照
success = pvc_provider.create_snapshot(
    pvc_name='training-data-pvc',
    snapshot_name='training-data-snapshot-2024-01-01',
    namespace='default'
)
if success:
    print("快照创建成功")
else:
    print("快照创建失败")
```

### 1.6 列出所有PVC

```python
# 列出命名空间中的所有PVC
pvcs = pvc_provider.list_pvcs('default')
for pvc in pvcs:
    print(f"PVC: {pvc['name']}, 状态: {pvc['phase']}, 大小: {pvc['capacity']}")
```

### 1.7 获取存储信息

```python
# 获取挂载的存储信息
info = pvc_provider.get_info('/mnt/training-data')
print(f"存储信息: {info}")

# 输出示例:
# {
#     'pvc_name': 'training-data-pvc',
#     'namespace': 'default',
#     'mount_path': '/mnt/training-data',
#     'volume_name': 'pvc-12345678-1234-1234-1234-123456789012',
#     'access_modes': ['ReadWriteOnce'],
#     'capacity': {'storage': '50Gi'},
#     'filesystem': {
#         'total_space': 53687091200,
#         'free_space': 42949672960,
#         'used_space': 10737418240,
#         'block_size': 4096
#     }
# }
```

### 1.8 健康检查

```python
# 检查存储健康状态
health = pvc_provider.check_health()
if health:
    print("存储系统健康")
else:
    print("存储系统异常")
```

### 1.9 卸载存储

```python
# 卸载PVC
success = pvc_provider.unmount('/mnt/training-data')
if success:
    print("PVC卸载成功")
else:
    print("PVC卸载失败")
```

### 1.10 删除PVC

```python
# 删除PVC
success = pvc_provider.delete_pvc('training-data-pvc', 'default')
if success:
    print("PVC删除成功")
else:
    print("PVC删除失败")
```

## 2. 在训练平台中的集成使用

### 2.1 存储管理器集成

```python
class StorageManager:
    """
    统一存储管理器
    """
    def __init__(self):
        self.providers = {
            'pvc': StorageManagerFactory.create_provider('pvc'),
            'hostpath': StorageManagerFactory.create_provider('hostpath'),
            'configmap': StorageManagerFactory.create_provider('configmap'),
            'secret': StorageManagerFactory.create_provider('secret'),
            'memory': StorageManagerFactory.create_provider('memory'),
            'nfs': StorageManagerFactory.create_provider('nfs'),
            's3': StorageManagerFactory.create_provider('s3')
        }
    
    def mount_storage(self, mount_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        挂载存储
        """
        provider_type = mount_config['type']
        if provider_type not in self.providers:
            raise ValueError(f"不支持的存储类型: {provider_type}")
        
        provider = self.providers[provider_type]
        return provider.mount(mount_config)
    
    def unmount_storage(self, provider_type: str, mount_path: str) -> bool:
        """
        卸载存储
        """
        if provider_type not in self.providers:
            return False
        
        provider = self.providers[provider_type]
        return provider.unmount(mount_path)
    
    def get_storage_info(self, provider_type: str, mount_path: str) -> Dict[str, Any]:
        """
        获取存储信息
        """
        if provider_type not in self.providers:
            return {}
        
        provider = self.providers[provider_type]
        return provider.get_info(mount_path)
```

### 2.2 Pipeline任务中的使用

```python
class PipelineTask:
    """
    Pipeline任务执行器
    """
    def __init__(self):
        self.storage_manager = StorageManager()
    
    def execute_task(self, task_config: Dict[str, Any]):
        """
        执行任务
        """
        # 挂载输入存储
        for input_storage in task_config.get('inputs', []):
            mount_result = self.storage_manager.mount_storage(input_storage)
            if not mount_result['success']:
                raise Exception(f"挂载输入存储失败: {mount_result['error']}")
        
        # 挂载输出存储
        for output_storage in task_config.get('outputs', []):
            mount_result = self.storage_manager.mount_storage(output_storage)
            if not mount_result['success']:
                raise Exception(f"挂载输出存储失败: {mount_result['error']}")
        
        # 执行任务逻辑
        self._run_task_logic(task_config)
        
        # 清理存储
        for storage in task_config.get('inputs', []) + task_config.get('outputs', []):
            self.storage_manager.unmount_storage(
                storage['type'], 
                storage['mount_path']
            )
    
    def _run_task_logic(self, task_config: Dict[str, Any]):
        """
        执行任务逻辑
        """
        # 具体的任务执行逻辑
        pass
```

### 2.3 多用户并发处理

```python
class MultiUserStorageManager:
    """
    多用户存储管理器
    """
    def __init__(self):
        self.storage_manager = StorageManager()
        self.user_mounts = {}  # 用户挂载信息
    
    def mount_for_user(self, user_id: str, mount_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        为用户挂载存储
        """
        # 为每个用户创建独立的挂载点
        user_mount_path = f"{mount_config['mount_path']}/user_{user_id}"
        mount_config['mount_path'] = user_mount_path
        
        # 挂载存储
        result = self.storage_manager.mount_storage(mount_config)
        
        if result['success']:
            # 记录用户挂载信息
            if user_id not in self.user_mounts:
                self.user_mounts[user_id] = []
            
            self.user_mounts[user_id].append({
                'mount_path': user_mount_path,
                'type': mount_config['type'],
                'config': mount_config
            })
        
        return result
    
    def unmount_for_user(self, user_id: str, mount_path: str) -> bool:
        """
        为用户卸载存储
        """
        if user_id not in self.user_mounts:
            return True
        
        for mount_info in self.user_mounts[user_id]:
            if mount_info['mount_path'] == mount_path:
                success = self.storage_manager.unmount_storage(
                    mount_info['type'], 
                    mount_path
                )
                
                if success:
                    self.user_mounts[user_id].remove(mount_info)
                
                return success
        
        return True
    
    def cleanup_user_mounts(self, user_id: str):
        """
        清理用户所有挂载
        """
        if user_id not in self.user_mounts:
            return
        
        for mount_info in self.user_mounts[user_id]:
            self.storage_manager.unmount_storage(
                mount_info['type'], 
                mount_info['mount_path']
            )
        
        del self.user_mounts[user_id]
```

## 3. 配置示例

### 3.1 Kubernetes配置

```yaml
# PVC配置示例
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: training-data-pvc
  namespace: train-platform
  labels:
    app: train-platform
    environment: production
  annotations:
    description: "Training data storage for ML platform"
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 100Gi
  storageClassName: fast-ssd
```

### 3.2 存储类配置

```yaml
# 存储类配置示例
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: fast-ssd
provisioner: kubernetes.io/aws-ebs
parameters:
  type: gp3
  iops: "3000"
  throughput: "125"
volumeBindingMode: WaitForFirstConsumer
allowVolumeExpansion: true
```

### 3.3 应用配置

```python
# 应用配置文件
STORAGE_CONFIG = {
    'providers': {
        'pvc': {
            'default_storage_class': 'fast-ssd',
            'default_namespace': 'train-platform',
            'auto_create': True
        },
        'nfs': {
            'default_server': 'nfs.example.com',
            'default_path': '/data',
            'default_options': 'rw,sync'
        },
        's3': {
            'default_endpoint': 'https://s3.example.com',
            'default_bucket': 'training-data'
        }
    },
    'mount_points': {
        'training_data': '/mnt/training-data',
        'models': '/mnt/models',
        'logs': '/mnt/logs',
        'temp': '/mnt/temp'
    }
}
```

## 4. 错误处理和监控

### 4.1 错误处理

```python
class StorageError(Exception):
    """存储相关错误"""
    pass

class StorageManager:
    def mount_storage(self, mount_config: Dict[str, Any]) -> Dict[str, Any]:
        try:
            provider_type = mount_config['type']
            provider = self.providers[provider_type]
            
            result = provider.mount(mount_config)
            
            if not result['success']:
                raise StorageError(f"挂载失败: {result['error']}")
            
            return result
            
        except Exception as e:
            logger.error(f"存储挂载异常: {e}")
            return {
                'success': False,
                'error': str(e)
            }
```

### 4.2 监控和指标

```python
class StorageMonitor:
    """
    存储监控器
    """
    def __init__(self):
        self.metrics = {}
    
    def record_mount_operation(self, provider_type: str, success: bool, duration: float):
        """
        记录挂载操作指标
        """
        if provider_type not in self.metrics:
            self.metrics[provider_type] = {
                'mount_operations': 0,
                'successful_mounts': 0,
                'failed_mounts': 0,
                'total_duration': 0.0
            }
        
        self.metrics[provider_type]['mount_operations'] += 1
        self.metrics[provider_type]['total_duration'] += duration
        
        if success:
            self.metrics[provider_type]['successful_mounts'] += 1
        else:
            self.metrics[provider_type]['failed_mounts'] += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        获取监控指标
        """
        return self.metrics
    
    def check_storage_health(self, storage_manager: StorageManager) -> Dict[str, bool]:
        """
        检查所有存储的健康状态
        """
        health_status = {}
        
        for provider_type, provider in storage_manager.providers.items():
            try:
                health_status[provider_type] = provider.check_health()
            except Exception as e:
                logger.error(f"检查 {provider_type} 健康状态失败: {e}")
                health_status[provider_type] = False
        
        return health_status
```

## 5. 最佳实践

### 5.1 资源管理

```python
class StorageResourceManager:
    """
    存储资源管理器
    """
    def __init__(self):
        self.storage_manager = StorageManager()
        self.resource_limits = {
            'pvc': {'max_size': '1Ti', 'max_count': 10},
            'memory': {'max_size': '10Gi', 'max_count': 5},
            'nfs': {'max_count': 3}
        }
    
    def can_create_storage(self, provider_type: str, size: str = None) -> bool:
        """
        检查是否可以创建存储
        """
        if provider_type not in self.resource_limits:
            return True
        
        limits = self.resource_limits[provider_type]
        
        # 检查数量限制
        current_count = len(self.storage_manager.providers[provider_type].mount_info)
        if current_count >= limits['max_count']:
            return False
        
        # 检查大小限制
        if size and 'max_size' in limits:
            if self._compare_size(size, limits['max_size']) > 0:
                return False
        
        return True
    
    def _compare_size(self, size1: str, size2: str) -> int:
        """
        比较存储大小
        """
        # 实现大小比较逻辑
        pass
```

### 5.2 自动清理

```python
class StorageCleanupManager:
    """
    存储清理管理器
    """
    def __init__(self):
        self.storage_manager = StorageManager()
        self.cleanup_policies = {
            'temp': {'max_age_hours': 24},
            'logs': {'max_age_hours': 168},  # 7天
            'cache': {'max_age_hours': 12}
        }
    
    def cleanup_expired_storage(self):
        """
        清理过期的存储
        """
        import time
        current_time = time.time()
        
        for provider_type, provider in self.storage_manager.providers.items():
            for mount_path, mount_info in provider.mount_info.items():
                # 检查是否过期
                if self._is_expired(mount_info, current_time):
                    logger.info(f"清理过期存储: {mount_path}")
                    provider.unmount(mount_path)
    
    def _is_expired(self, mount_info: Dict[str, Any], current_time: float) -> bool:
        """
        检查存储是否过期
        """
        # 实现过期检查逻辑
        pass
```

这些示例展示了如何在实际项目中使用PVCProvider和其他存储提供者，包括基本操作、错误处理、监控和最佳实践。 
# GPU资源管理使用指南

## 概述

本项目集成了自定义的GPU资源管理器，支持多GPU训练、资源调度和监控功能。GPU资源管理器基于Kubernetes生态，提供了完整的GPU资源生命周期管理。

## 功能特性

### 🚀 核心功能
- **GPU资源解析**: 支持多种GPU格式解析（V100、A100、T4等）
- **资源分配**: 智能GPU资源分配和调度
- **显存保障**: 确保深度学习任务有足够显存启动
- **多厂商支持**: 支持NVIDIA、AMD、Intel等多种GPU厂商
- **分布式训练**: 支持多GPU分布式训练配置
- **实时监控**: 提供GPU使用率、显存、温度等实时监控

### 🔧 集成功能
- **TensorFlow集成**: 自动配置TensorFlow GPU环境
- **PyTorch集成**: 自动配置PyTorch GPU环境
- **训练引擎集成**: 与算法训练引擎无缝集成
- **API接口**: 提供完整的RESTful API接口

## 快速开始

### 1. 环境准备

确保系统已安装以下组件：
- Python 3.8+
- Kubernetes集群（可选，用于生产环境）
- NVIDIA GPU驱动和CUDA（如果使用NVIDIA GPU）

### 2. 安装依赖

```bash
# 安装基础依赖
pip install -r requirements.txt

# 安装GPU相关依赖（可选）
pip install tensorflow-gpu torch torchvision
```

### 3. 配置GPU资源管理器

创建配置文件 `gpu_config.yaml`:

```yaml
# GPU资源配置
gpu_resource:
  gpu: "nvidia.com/gpu"
  nvidia: "nvidia.com/gpu"
  amd: "amd.com/gpu"
  intel: "intel.com/gpu"
  npu: "huawei.com/npu"

# 默认GPU资源名称
default_gpu_resource_name: "nvidia.com/gpu"

# GPU显存配置 (GB)
gpu_memory_specs:
  T4: 16.0
  V100: 32.0
  A100: 80.0
  H100: 80.0
  RTX3090: 24.0
  RTX4090: 24.0
  A6000: 48.0
  A40: 48.0

# 监控配置
monitor_update_interval: 30
resource_change_threshold: 0.05
log_level: "INFO"
```

## API使用指南

### 1. GPU资源状态查询

#### 获取GPU资源状态
```bash
curl -X GET "http://localhost:8000/api/v1/gpu/status"
```

响应示例：
```json
{
  "utilization": {
    "gpu_utilization": 0.75,
    "gpu_memory_usage": 0.65
  },
  "available_nodes": [
    {
      "node_name": "gpu-node-1",
      "gpu_type": "V100",
      "available_gpus": 2,
      "memory_per_gpu": 32.0,
      "utilization": 0.6
    }
  ],
  "total_nodes": 3,
  "timestamp": "2024-01-01T12:00:00"
}
```

#### 获取可用GPU节点
```bash
curl -X GET "http://localhost:8000/api/v1/gpu/nodes?gpu_type=V100&min_memory_gb=16"
```

### 2. GPU资源分配

#### 分配GPU资源
```bash
curl -X POST "http://localhost:8000/api/v1/gpu/allocate" \
  -H "Content-Type: application/json" \
  -d '{
    "gpu_count": 2,
    "gpu_type": "V100",
    "memory_gb": 32.0,
    "distributed_training": true,
    "mixed_precision": true
  }'
```

响应示例：
```json
{
  "success": true,
  "node_name": "gpu-node-1",
  "gpu_config": {
    "gpu_count": 2,
    "gpu_type": "V100",
    "memory_gb": 32.0,
    "distributed_training": true
  }
}
```

#### 验证GPU需求
```bash
curl -X POST "http://localhost:8000/api/v1/gpu/validate" \
  -H "Content-Type: application/json" \
  -d '{
    "gpu_count": 1,
    "gpu_type": "A100",
    "memory_gb": 80.0
  }'
```

### 3. 深度学习框架集成

#### 设置TensorFlow GPU环境
```bash
curl -X POST "http://localhost:8000/api/v1/gpu/setup-tensorflow" \
  -H "Content-Type: application/json" \
  -d '{
    "gpu_count": 1,
    "gpu_type": "V100",
    "memory_gb": 32.0,
    "gpu_memory_fraction": 0.9
  }'
```

#### 设置PyTorch GPU环境
```bash
curl -X POST "http://localhost:8000/api/v1/gpu/setup-pytorch" \
  -H "Content-Type: application/json" \
  -d '{
    "gpu_count": 2,
    "gpu_type": "A100",
    "memory_gb": 80.0,
    "distributed_training": true
  }'
```

## 编程接口使用

### 1. Python API使用

#### 基本使用
```python
from backend.algorithm_engine.gpu_resource_integration import (
    get_gpu_resource_manager,
    TrainingGPUConfig
)

# 获取GPU资源管理器
gpu_manager = get_gpu_resource_manager()

# 创建GPU配置
gpu_config = TrainingGPUConfig(
    gpu_count=2,
    gpu_type='V100',
    memory_gb=32.0,
    distributed_training=True
)

# 验证GPU需求
if gpu_manager.validate_gpu_requirements(gpu_config):
    # 分配GPU资源
    allocated_node = gpu_manager.allocate_gpu_resources(gpu_config)
    print(f"分配到的节点: {allocated_node}")
    
    # 清理资源
    gpu_manager.cleanup_gpu_resources(allocated_node, gpu_config.gpu_count)
```

#### 与训练引擎集成
```python
from backend.algorithm_engine.core import AlgorithmTrainingEngine

# 创建训练引擎
engine = AlgorithmTrainingEngine()

# 配置训练参数（包含GPU配置）
config = TrainingConfig(
    name="gpu_training_example",
    algorithm_params={
        'gpu_config': {
            'enabled': True,
            'gpu_count': 2,
            'gpu_type': 'V100',
            'memory_gb': 32.0,
            'distributed_training': True,
            'mixed_precision': True
        }
    }
)

# 执行训练（自动处理GPU资源）
result = await engine.train_algorithm(AlgorithmType.VIBRATION_ANALYSIS, config, data)
```

### 2. TensorFlow集成

```python
from backend.algorithm_engine.gpu_resource_integration import get_tensorflow_gpu_integration

# 获取TensorFlow GPU集成
tensorflow_gpu = get_tensorflow_gpu_integration()

# 设置GPU环境
gpu_config = TrainingGPUConfig(
    gpu_count=1,
    gpu_type='V100',
    memory_gb=32.0
)

tf_config = tensorflow_gpu.setup_tensorflow_gpu(gpu_config)
print(f"TensorFlow GPU配置: {tf_config}")

# 创建模型（自动使用GPU）
model_config = {
    'type': 'mlp',
    'input_dim': 10,
    'hidden_units': [128, 64],
    'output_dim': 1
}

model = tensorflow_gpu.create_tensorflow_model_with_gpu(model_config, gpu_config)
```

### 3. PyTorch集成

```python
from backend.algorithm_engine.gpu_resource_integration import get_pytorch_gpu_integration

# 获取PyTorch GPU集成
pytorch_gpu = get_pytorch_gpu_integration()

# 设置GPU环境
gpu_config = TrainingGPUConfig(
    gpu_count=2,
    gpu_type='A100',
    memory_gb=80.0,
    distributed_training=True
)

pytorch_config = pytorch_gpu.setup_pytorch_gpu(gpu_config)
print(f"PyTorch GPU配置: {pytorch_config}")

# 创建模型（自动使用GPU）
model_config = {
    'type': 'mlp',
    'input_dim': 10,
    'hidden_units': [128, 64],
    'output_dim': 1
}

model = pytorch_gpu.create_pytorch_model_with_gpu(model_config, gpu_config)
```

## 配置最佳实践

### 1. GPU资源配置

#### 单GPU训练配置
```python
gpu_config = TrainingGPUConfig(
    gpu_count=1,
    gpu_type='V100',
    memory_gb=32.0,
    distributed_training=False,
    mixed_precision=True,
    gpu_memory_fraction=0.9
)
```

#### 多GPU分布式训练配置
```python
gpu_config = TrainingGPUConfig(
    gpu_count=4,
    gpu_type='A100',
    memory_gb=80.0,
    distributed_training=True,
    mixed_precision=True,
    gpu_memory_fraction=0.95
)
```

#### 混合精度训练配置
```python
gpu_config = TrainingGPUConfig(
    gpu_count=2,
    gpu_type='V100',
    memory_gb=32.0,
    distributed_training=True,
    mixed_precision=True,  # 启用混合精度
    gpu_memory_fraction=0.8  # 预留20%显存
)
```

### 2. 内存管理

#### 显存分配策略
- **保守策略**: `gpu_memory_fraction=0.8` - 预留20%显存
- **激进策略**: `gpu_memory_fraction=0.95` - 预留5%显存
- **动态策略**: 根据模型大小自动调整

#### 显存监控
```python
# 获取GPU监控数据
monitoring_data = gpu_manager.get_gpu_monitoring_data()
gpu_utilization = monitoring_data.get('utilization', {}).get('gpu_utilization', 0.0)
gpu_memory_usage = monitoring_data.get('utilization', {}).get('gpu_memory_usage', 0.0)

print(f"GPU利用率: {gpu_utilization:.2%}")
print(f"显存使用率: {gpu_memory_usage:.2%}")
```

### 3. 错误处理

#### 资源不足处理
```python
try:
    allocated_node = gpu_manager.allocate_gpu_resources(gpu_config)
    if allocated_node:
        # 执行训练
        result = await train_model()
    else:
        # 降级到CPU训练
        logger.warning("GPU资源不足，使用CPU训练")
        result = await train_model_cpu()
except Exception as e:
    logger.error(f"GPU资源分配失败: {e}")
    # 清理资源并重试
    gpu_manager.cleanup_gpu_resources(allocated_node, gpu_config.gpu_count)
```

#### 显存不足处理
```python
# 验证显存需求
if not gpu_manager.validate_gpu_requirements(gpu_config):
    # 调整配置
    gpu_config.memory_gb = gpu_config.memory_gb * 0.8
    gpu_config.gpu_memory_fraction = 0.7
    
    # 重新验证
    if gpu_manager.validate_gpu_requirements(gpu_config):
        logger.info("调整配置后显存需求满足")
    else:
        logger.error("显存需求无法满足，请使用更小的模型或更多GPU")
```

## 监控和调试

### 1. 健康检查

```bash
curl -X GET "http://localhost:8000/api/v1/gpu/health"
```

### 2. 监控数据

```bash
curl -X GET "http://localhost:8000/api/v1/gpu/monitoring"
```

### 3. 日志查看

```bash
# 查看GPU资源管理器日志
tail -f logs/gpu_resource_manager.log

# 查看训练日志
tail -f logs/training.log
```

## 故障排除

### 常见问题

#### 1. GPU资源管理器未初始化
**症状**: API返回503错误
**解决方案**: 
- 检查Kubernetes集群连接
- 验证GPU资源管理器配置
- 查看日志文件

#### 2. GPU资源分配失败
**症状**: 分配API返回失败
**解决方案**:
- 检查GPU节点可用性
- 验证显存需求
- 调整GPU配置参数

#### 3. TensorFlow/PyTorch GPU设置失败
**症状**: 深度学习框架无法使用GPU
**解决方案**:
- 检查CUDA安装
- 验证GPU驱动版本
- 确认框架版本兼容性

#### 4. 显存不足
**症状**: 训练过程中出现OOM错误
**解决方案**:
- 减少batch_size
- 启用混合精度训练
- 使用梯度累积
- 调整模型大小

### 调试技巧

#### 1. 启用详细日志
```python
import logging
logging.getLogger('gpu_resource_manager').setLevel(logging.DEBUG)
```

#### 2. 检查GPU状态
```python
# 获取详细GPU信息
nodes = gpu_manager.get_available_gpu_nodes()
for node in nodes:
    print(f"节点: {node.node_name}")
    print(f"GPU类型: {node.gpu_type}")
    print(f"可用GPU: {node.available_gpus}")
    print(f"显存: {node.memory_per_gpu}GB")
    print(f"利用率: {node.utilization:.2%}")
```

#### 3. 性能分析
```python
# 监控训练性能
import time

start_time = time.time()
# 执行训练
training_time = time.time() - start_time

print(f"训练耗时: {training_time:.2f}秒")
print(f"GPU利用率: {gpu_utilization:.2%}")
```

## 性能优化

### 1. GPU利用率优化

#### 批量大小优化
```python
# 根据GPU显存动态调整batch_size
gpu_memory = gpu_config.memory_gb * 1024  # MB
optimal_batch_size = int(gpu_memory / 100)  # 估算最优batch_size
```

#### 混合精度训练
```python
# 启用混合精度以提高训练速度
gpu_config.mixed_precision = True
```

### 2. 多GPU优化

#### 数据并行
```python
# 配置数据并行训练
gpu_config.distributed_training = True
gpu_config.gpu_count = 4  # 使用4个GPU
```

#### 模型并行
```python
# 对于超大模型，考虑模型并行
# 需要自定义模型分割逻辑
```

### 3. 内存优化

#### 梯度累积
```python
# 使用梯度累积减少显存使用
accumulation_steps = 4
for i in range(0, len(data), batch_size):
    # 前向传播
    loss = model(data[i:i+batch_size])
    loss = loss / accumulation_steps
    loss.backward()
    
    if (i // batch_size + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

## 总结

GPU资源管理器为深度学习训练提供了完整的GPU资源管理解决方案，包括：

1. **智能资源分配**: 自动分配最适合的GPU资源
2. **显存保障**: 确保训练任务有足够显存启动
3. **多框架支持**: 无缝集成TensorFlow和PyTorch
4. **实时监控**: 提供GPU使用情况的实时监控
5. **错误处理**: 完善的错误处理和恢复机制

通过合理配置和使用GPU资源管理器，可以显著提高深度学习训练的效率 and 稳定性。 
# Epoch训练功能使用指南

## 📋 功能概述

本系统现在支持完整的TensorFlow/PyTorch epoch轮次训练功能，包括：

- ✅ **完整epoch训练流程**
- ✅ **实时进度监控**
- ✅ **早停机制**
- ✅ **学习率调度**
- ✅ **训练控制（暂停/恢复）**
- ✅ **多GPU支持**

## 🚀 快速开始

### 1. 启动Epoch训练

```python
import requests
import json

# 配置训练参数
config = {
    "name": "深度学习模型训练",
    "algorithm_type": "deep_learning",
    "model_type": "mlp",
    "epochs": 100,
    "batch_size": 32,
    "learning_rate": 0.001,
    "hidden_units": [128, 64, 32],
    "dropout_rate": 0.2,
    "early_stopping_patience": 10,
    "learning_rate_scheduler": "step",
    "feature_columns": ["feature1", "feature2", "feature3"],
    "target_column": "target",
    "output_path": "/models/deep_learning_model"
}

# 启动训练
response = requests.post(
    "http://localhost:8000/api/v1/epoch-training/start",
    json=config,
    headers={"Authorization": "Bearer your_token"}
)

task_id = response.json()["task_id"]
print(f"训练任务ID: {task_id}")
```

### 2. 监控训练进度

```python
# 获取训练进度
response = requests.get(f"http://localhost:8000/api/v1/epoch-training/progress/{task_id}")
progress = response.json()["progress"]

print(f"当前epoch: {progress['current_epoch']}")
print(f"训练状态: {progress['training_state']}")
print(f"最新指标: {progress['latest_metrics']}")
```

### 3. 流式监控（SSE）

```python
import sseclient

# 流式获取训练进度
response = requests.get(
    f"http://localhost:8000/api/v1/epoch-training/stream-progress/{task_id}",
    stream=True
)

client = sseclient.SSEClient(response)
for event in client.events():
    data = json.loads(event.data)
    if data["type"] == "progress":
        epoch = data["progress"]["current_epoch"]
        metrics = data["progress"]["latest_metrics"]
        print(f"Epoch {epoch}: Loss={metrics['train_loss']:.4f}, Acc={metrics['train_accuracy']:.4f}")
    elif data["type"] == "complete":
        print("训练完成!")
        break
```

### 4. 训练控制

```python
# 暂停训练
requests.post(f"http://localhost:8000/api/v1/epoch-training/pause/{task_id}")

# 恢复训练
requests.post(f"http://localhost:8000/api/v1/epoch-training/resume/{task_id}")
```

## 📊 API接口详解

### 启动训练

**POST** `/api/v1/epoch-training/start`

```json
{
  "name": "深度学习模型训练",
  "algorithm_type": "deep_learning",
  "model_type": "mlp",
  "epochs": 100,
  "batch_size": 32,
  "learning_rate": 0.001,
  "hidden_units": [128, 64, 32],
  "dropout_rate": 0.2,
  "early_stopping_patience": 10,
  "learning_rate_scheduler": "step",
  "feature_columns": ["feature1", "feature2", "feature3"],
  "target_column": "target",
  "output_path": "/models/deep_learning_model"
}
```

**响应:**
```json
{
  "success": true,
  "task_id": "epoch_training_20240101_120000",
  "message": "Epoch训练已启动",
  "config": {...}
}
```

### 获取训练进度

**GET** `/api/v1/epoch-training/progress/{task_id}`

**响应:**
```json
{
  "success": true,
  "task_id": "epoch_training_20240101_120000",
  "progress": {
    "current_epoch": 25,
    "total_epochs": 100,
    "training_state": "running",
    "latest_metrics": {
      "epoch": 25,
      "train_loss": 0.1234,
      "train_accuracy": 0.9234,
      "val_loss": 0.1456,
      "val_accuracy": 0.9123,
      "learning_rate": 0.001,
      "time_per_epoch": 45.2
    },
    "best_metrics": {
      "loss": 0.1234,
      "accuracy": 0.9234
    }
  }
}
```

### 获取训练历史

**GET** `/api/v1/epoch-training/history/{task_id}`

**响应:**
```json
{
  "success": true,
  "task_id": "epoch_training_20240101_120000",
  "history": [
    {
      "epoch": 1,
      "train_loss": 0.2345,
      "train_accuracy": 0.8567,
      "val_loss": 0.1987,
      "val_accuracy": 0.8723,
      "learning_rate": 0.001,
      "time_per_epoch": 42.1
    },
    // ... 更多epoch记录
  ]
}
```

## 🔧 配置参数详解

### 模型配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model_type` | string | "mlp" | 模型类型（mlp, cnn, lstm, gru） |
| `hidden_units` | list | [64, 32] | 隐藏层单元数 |
| `dropout_rate` | float | 0.2 | Dropout比率 |

### 训练配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `epochs` | int | 100 | 训练轮数 |
| `batch_size` | int | 32 | 批次大小 |
| `learning_rate` | float | 0.001 | 学习率 |

### 早停配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `early_stopping_patience` | int | 10 | 早停耐心值 |
| `min_delta` | float | 0.001 | 最小改善阈值 |

### 学习率调度配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `learning_rate_scheduler` | string | "step" | 调度器类型 |
| `step_size` | int | 30 | 步进间隔（step调度器） |
| `gamma` | float | 0.1 | 衰减因子 |

## 📈 支持的调度器类型

### 1. 步进调度器 (step)
```python
# 每30个epoch降低学习率
config = {
    "learning_rate_scheduler": "step",
    "step_size": 30,
    "gamma": 0.1
}
```

### 2. 平台调度器 (plateau)
```python
# 当验证损失不再改善时降低学习率
config = {
    "learning_rate_scheduler": "plateau",
    "patience": 5,
    "factor": 0.5
}
```

### 3. 余弦退火 (cosine)
```python
# 使用余弦函数平滑降低学习率
config = {
    "learning_rate_scheduler": "cosine",
    "T_max": 100,
    "eta_min": 0.0001
}
```

## 🎯 使用示例

### 示例1: 基础MLP训练

```python
config = {
    "name": "基础MLP训练",
    "algorithm_type": "deep_learning",
    "model_type": "mlp",
    "epochs": 50,
    "batch_size": 64,
    "learning_rate": 0.001,
    "hidden_units": [128, 64],
    "dropout_rate": 0.3,
    "early_stopping_patience": 5,
    "learning_rate_scheduler": "step",
    "feature_columns": ["feature1", "feature2", "feature3"],
    "target_column": "target",
    "output_path": "/models/mlp_model"
}
```

### 示例2: 复杂网络训练

```python
config = {
    "name": "复杂网络训练",
    "algorithm_type": "deep_learning",
    "model_type": "mlp",
    "epochs": 200,
    "batch_size": 32,
    "learning_rate": 0.0001,
    "hidden_units": [256, 128, 64, 32],
    "dropout_rate": 0.5,
    "early_stopping_patience": 15,
    "learning_rate_scheduler": "plateau",
    "feature_columns": ["feature1", "feature2", "feature3", "feature4"],
    "target_column": "target",
    "output_path": "/models/complex_model"
}
```

## 🔍 故障排除

### 常见问题

1. **TensorFlow/PyTorch未安装**
   ```
   解决方案: pip install tensorflow torch
   ```

2. **GPU内存不足**
   ```
   解决方案: 减少batch_size或使用CPU训练
   ```

3. **训练速度慢**
   ```
   解决方案: 检查GPU使用率，调整batch_size
   ```

4. **早停过早触发**
   ```
   解决方案: 增加patience值或调整min_delta
   ```

### 性能优化建议

1. **使用GPU训练**
   ```python
   # 检查GPU可用性
   import tensorflow as tf
   print("GPU数量:", len(tf.config.list_physical_devices('GPU')))
   ```

2. **调整批次大小**
   ```python
   # 根据GPU内存调整
   config["batch_size"] = 64  # 或更大
   ```

3. **使用混合精度**
   ```python
   # 在TensorFlow中启用混合精度
   policy = tf.keras.mixed_precision.Policy('mixed_float16')
   tf.keras.mixed_precision.set_global_policy(policy)
   ```

## 📚 更多资源

- [TensorFlow官方文档](https://www.tensorflow.org/)
- [PyTorch官方文档](https://pytorch.org/)
- [深度学习最佳实践](https://github.com/keras-team/keras/blob/master/keras/guides/)
- [项目GitHub仓库](https://github.com/your-repo/train-storge-workflow)

这个epoch训练功能为系统提供了完整的深度学习训练支持，包括实时监控、自动优化和灵活的配置选项。 
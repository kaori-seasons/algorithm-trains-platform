# 工业故障预测维护训练存储工作流平台

## 项目概述

本项目是一个基于微服务架构的工业故障预测维护训练存储工作流平台，支持多种算法的交互式训练和调试，特别针对振动分析算法进行了优化设计。

## 核心特性

### 🚀 算法引擎
- **多算法支持**: 状态识别、健康评估、报警、振动分析、仿真、传统机器学习、深度学习
- **交互式训练**: 支持实时参数调整和模型优化
- **智能调试**: 基于数据可视化的交互式调试功能
- **Epoch训练**: 支持TensorFlow/PyTorch完整epoch轮次训练，包含早停机制、学习率调度、实时监控

### 🧠 Epoch训练功能
- **完整训练流程**: 支持TensorFlow和PyTorch的完整epoch训练
- **实时进度监控**: WebSocket实时推送训练进度和指标
- **早停机制**: 自动检测过拟合并停止训练
- **学习率调度**: 支持步进、平台、余弦退火等调度策略
- **训练控制**: 支持暂停、恢复、取消训练操作
- **多GPU支持**: 自动检测并使用多GPU进行分布式训练
- **GPU资源管理**: 集成GPU资源管理器，支持GPU分配、监控和调度

### 📊 振动算法专用功能
- **无预处理设计**: 直接处理原始振动信号，无需传统数据预处理
- **实时交互式训练**: 支持频率过滤、振幅阈值、数据选择等实时调整
- **频域分析**: 完整的FFT分析、谐波分析、边带分析
- **时域特征**: RMS、峰值、峰值因子、峭度等特征提取
- **可视化支持**: 时域波形、频谱图、特征分布、趋势分析

### 🔧 交互式调试功能

#### 状态识别算法调试
- **数据层面调试**: 异常值处理、特征选择、数据采样、特征变换
- **实时反馈**: 立即显示调试结果和可视化数据
- **智能建议**: 基于数据分析的自动优化建议

#### 振动算法调试
- **频率过滤**: 低通、高通、带通滤波器配置
- **振幅阈值**: 警告、报警、危险三级阈值设置
- **数据选择**: 时间范围、转速范围、质量过滤
- **实时配置**: 自适应阈值、动态过滤、特征权重调整

## 技术架构

### 后端技术栈
- **框架**: FastAPI + SQLAlchemy
- **数据库**: PostgreSQL + Redis
- **消息队列**: Celery
- **机器学习**: scikit-learn, TensorFlow, PyTorch
- **信号处理**: scipy, numpy
- **GPU管理**: 自定义GPU资源管理器，支持多厂商GPU
- **部署**: Docker + Kubernetes

### 前端技术栈
- **框架**: Vue 2 + Element UI
- **可视化**: ECharts, D3.js
- **状态管理**: Vuex
- **路由**: Vue Router

## 快速开始

### 环境要求
- Python 3.9+
- Node.js 14+
- Docker & Docker Compose
- PostgreSQL 12+
- Redis 6+

### 安装步骤

1. **克隆项目**
```bash
git clone <repository-url>
cd train-storge-workflow
```

2. **后端设置**
```bash
cd backend
pip install -r requirements.txt
python -m uvicorn main:app --reload
```

3. **前端设置**
```bash
cd frontend
npm install
npm run serve
```

4. **数据库初始化**
```bash
docker-compose up -d postgres redis
python scripts/init-db.py
```

## API接口

### GPU资源管理接口

#### 获取GPU资源状态
```http
GET /api/v1/gpu/status
```

#### 获取可用GPU节点
```http
GET /api/v1/gpu/nodes?gpu_type=V100&min_memory_gb=16
```

#### 分配GPU资源
```http
POST /api/v1/gpu/allocate
Content-Type: application/json

{
  "gpu_count": 2,
  "gpu_type": "V100",
  "memory_gb": 32.0,
  "distributed_training": true,
  "mixed_precision": true
}
```

#### 验证GPU需求
```http
POST /api/v1/gpu/validate
Content-Type: application/json

{
  "gpu_count": 1,
  "gpu_type": "A100",
  "memory_gb": 80.0
}
```

#### 设置TensorFlow GPU环境
```http
POST /api/v1/gpu/setup-tensorflow
Content-Type: application/json

{
  "gpu_count": 1,
  "gpu_type": "V100",
  "memory_gb": 32.0,
  "gpu_memory_fraction": 0.9
}
```

#### 设置PyTorch GPU环境
```http
POST /api/v1/gpu/setup-pytorch
Content-Type: application/json

{
  "gpu_count": 2,
  "gpu_type": "A100",
  "memory_gb": 80.0,
  "distributed_training": true
}
```

### 算法训练接口

#### 启动训练
```http
POST /api/v1/algorithm/train
Content-Type: application/json

{
  "algorithm_type": "vibration",
  "train_data_path": "/path/to/data.csv",
  "feature_columns": ["x_accel", "y_accel", "z_accel", "speed"],
  "target_column": "status",
  "algorithm_params": {
    "vibration_config": {
      "sampling_rate": 1000,
      "model_type": "isolation_forest",
      "frequency_range": [0, 500]
    }
  }
}
```

#### 交互式训练（振动算法）
```http
POST /api/v1/algorithm/vibration/interactive-training
Content-Type: application/json

{
  "task_id": "vibration_20240101_120000",
  "training_params": {
    "frequency_filtering": {
      "enabled": true,
      "low_freq_cutoff": 10,
      "high_freq_cutoff": 200,
      "bandpass_filters": [
        {"name": "bearing_freq", "center": 50, "bandwidth": 10}
      ]
    },
    "amplitude_thresholds": {
      "warning_level": 0.3,
      "alarm_level": 0.8,
      "critical_level": 1.5
    },
    "data_selection": {
      "time_range": {
        "start_time": "2024-01-01 00:00:00",
        "end_time": "2024-01-01 23:59:59"
      }
    }
  }
}
```

#### 获取振动分析结果
```http
GET /api/v1/algorithm/vibration/analysis/{task_id}
```

#### 获取振动可视化数据
```http
GET /api/v1/algorithm/vibration/visualization/{task_id}
```

### 交互式调试接口

#### 状态识别算法调试
```http
POST /api/v1/algorithm/interactive-debug
Content-Type: application/json

{
  "task_id": "status_recognition_20240101_120000",
  "debug_params": {
    "outlier_handling": {
      "columns": ["temperature", "vibration"],
      "method": "iqr",
      "action": "remove"
    },
    "feature_selection": ["temperature", "vibration", "pressure"],
    "sampling": {
      "method": "stratified",
      "size": 1000
    }
  }
}
```

## 振动算法参数配置

### 基础配置
```json
{
  "vibration_config": {
    "sampling_rate": 1000,
    "data_type": "float32",
    "model_type": "isolation_forest",
    "contamination": 0.1,
    "frequency_range": [0, 1000]
  }
}
```

### 时域特征配置
```json
{
  "time_domain": {
    "rms_enabled": true,
    "peak_enabled": true,
    "crest_factor_enabled": true,
    "kurtosis_enabled": true
  }
}
```

### 频域特征配置
```json
{
  "frequency_domain": {
    "spectrum_enabled": true,
    "harmonic_analysis": true,
    "sideband_analysis": true,
    "envelope_analysis": true
  }
}
```

### 交互式训练参数
```json
{
  "interactive_features": {
    "frequency_filtering": true,
    "amplitude_thresholds": true,
    "data_selection": true,
    "real_time_adjustment": true,
    "feature_weights": true
  }
}
```

## 测试

### 运行GPU资源管理集成测试
```bash
python test_gpu_resource_integration.py
```

### 运行振动算法测试
```bash
cd backend
python test_vibration_interactive.py
```

### 运行交互式调试测试
```bash
cd backend
python test_interactive_debug.py
```

## 部署

### Docker部署
```bash
docker-compose up -d
```

### Kubernetes部署
```bash
kubectl apply -f k8s/
```

## 监控和日志

- **Prometheus**: 性能指标监控
- **Grafana**: 可视化仪表板
- **ELK Stack**: 日志收集和分析

## 贡献指南

1. Fork 项目
2. 创建功能分支
3. 提交更改
4. 推送到分支
5. 创建 Pull Request

## 许可证

MIT License

## 联系方式

- 项目维护者: [Your Name]
- 邮箱: [your.email@example.com]
- 项目地址: [GitHub Repository URL]

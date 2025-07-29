"""
Epoch轮次训练器实现
支持TensorFlow/PyTorch的完整epoch训练流程
"""
import asyncio
import logging
import json
import os
import time
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum

from .gpu_resource_integration import (
    get_gpu_resource_manager,
    get_tensorflow_gpu_integration,
    get_pytorch_gpu_integration,
    TrainingGPUConfig
)

logger = logging.getLogger(__name__)


class TrainingState(str, Enum):
    """训练状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class EpochMetrics:
    """Epoch级别指标"""
    epoch: int
    train_loss: float
    train_accuracy: float
    val_loss: Optional[float] = None
    val_accuracy: Optional[float] = None
    learning_rate: float = 0.001
    time_per_epoch: float = 0.0
    memory_usage: Optional[float] = None
    gpu_utilization: Optional[float] = None


class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001, monitor: str = 'val_loss'):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.best_metric = None
        self.counter = 0
        self.best_epoch = 0
    
    def should_stop(self, current_metric: float, epoch: int) -> bool:
        """判断是否应该早停"""
        if self.best_metric is None:
            self.best_metric = current_metric
            self.best_epoch = epoch
            return False
        
        # 对于损失函数，越小越好；对于准确率，越大越好
        is_better = (current_metric < self.best_metric - self.min_delta) if 'loss' in self.monitor else (current_metric > self.best_metric + self.min_delta)
        
        if is_better:
            self.best_metric = current_metric
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            
        return self.counter >= self.patience
    
    def get_best_epoch(self) -> int:
        """获取最佳epoch"""
        return self.best_epoch


class LearningRateScheduler:
    """学习率调度器"""
    
    def __init__(self, initial_lr: float, scheduler_type: str = 'step'):
        self.initial_lr = initial_lr
        self.current_lr = initial_lr
        self.scheduler_type = scheduler_type
        self.best_val_loss = float('inf')
        self.step_size = 30
        self.gamma = 0.1
    
    def step(self, epoch: int, metrics: Dict[str, float]) -> float:
        """更新学习率"""
        if self.scheduler_type == 'step':
            if epoch % self.step_size == 0 and epoch > 0:
                self.current_lr *= self.gamma
        elif self.scheduler_type == 'plateau':
            if 'val_loss' in metrics:
                if metrics['val_loss'] > self.best_val_loss:
                    self.current_lr *= 0.5
                    self.best_val_loss = metrics['val_loss']
        elif self.scheduler_type == 'cosine':
            # 余弦退火
            self.current_lr = self.initial_lr * (1 + np.cos(np.pi * epoch / 100)) / 2
        
        return self.current_lr
    
    def get_current_lr(self) -> float:
        """获取当前学习率"""
        return self.current_lr


class TrainingProgressManager:
    """训练进度管理器"""
    
    def __init__(self):
        self.websocket_connections = {}
        self.progress_callbacks = {}
        self.training_states = {}
    
    def register_progress_callback(self, task_id: str, callback: Callable):
        """注册进度回调"""
        self.progress_callbacks[task_id] = callback
    
    def broadcast_epoch_progress(self, task_id: str, epoch: int, metrics: Dict[str, Any]):
        """广播epoch进度"""
        progress_data = {
            'task_id': task_id,
            'epoch': epoch,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        # 调用回调函数
        if task_id in self.progress_callbacks:
            try:
                self.progress_callbacks[task_id](epoch, metrics)
            except Exception as e:
                logger.error(f"进度回调执行失败: {str(e)}")
        
        # 发送到WebSocket（如果有连接）
        if task_id in self.websocket_connections:
            for connection in self.websocket_connections[task_id]:
                try:
                    connection.send_json(progress_data)
                except Exception as e:
                    logger.error(f"WebSocket发送失败: {str(e)}")
    
    def update_training_state(self, task_id: str, state: TrainingState):
        """更新训练状态"""
        self.training_states[task_id] = state
    
    def get_training_state(self, task_id: str) -> Optional[TrainingState]:
        """获取训练状态"""
        return self.training_states.get(task_id)


class EpochDeepLearningTrainer:
    """支持完整epoch训练的深度学习训练器"""
    
    def __init__(self):
        self.epoch_history = []
        self.current_epoch = 0
        self.best_metrics = {}
        self.early_stopping = None
        self.learning_rate_scheduler = None
        self.progress_manager = TrainingProgressManager()
        self.training_state = TrainingState.PENDING
        self.model = None
        self.data = None
        self.config = None
        
        # TensorFlow/PyTorch模型
        self.tf_model = None
        self.pytorch_model = None
        
        # 初始化GPU资源管理器
        self.gpu_manager = get_gpu_resource_manager()
        self.tensorflow_gpu = get_tensorflow_gpu_integration()
        self.pytorch_gpu = get_pytorch_gpu_integration()
        
    def _initialize_training(self, config: Dict[str, Any], data: Dict[str, Any]):
        """初始化训练"""
        self.config = config
        self.data = data
        self.current_epoch = 0
        self.epoch_history = []
        
        # 初始化早停机制
        patience = config.get('early_stopping_patience', 10)
        self.early_stopping = EarlyStopping(patience=patience)
        
        # 初始化学习率调度器
        initial_lr = config.get('learning_rate', 0.001)
        scheduler_type = config.get('learning_rate_scheduler', 'step')
        self.learning_rate_scheduler = LearningRateScheduler(initial_lr, scheduler_type)
        
        # 创建模型
        self._create_model()
        
        logger.info(f"训练初始化完成，总epoch数: {config.get('epochs', 100)}")
    
    def _create_model(self):
        """创建深度学习模型"""
        model_type = self.config.get('model_type', 'mlp')
        
        try:
            # 尝试创建TensorFlow模型
            self.tf_model = self._create_tensorflow_model(model_type)
            if self.tf_model:
                logger.info("使用TensorFlow模型")
                return
        except Exception as e:
            logger.warning(f"TensorFlow模型创建失败: {str(e)}")
        
        try:
            # 尝试创建PyTorch模型
            self.pytorch_model = self._create_pytorch_model(model_type)
            if self.pytorch_model:
                logger.info("使用PyTorch模型")
                return
        except Exception as e:
            logger.warning(f"PyTorch模型创建失败: {str(e)}")
        
        raise ValueError("无法创建深度学习模型，请检查TensorFlow或PyTorch安装")
    
    def _create_tensorflow_model(self, model_type: str):
        """创建TensorFlow模型"""
        try:
            import tensorflow as tf
            from tensorflow import keras
            
            input_dim = len(self.config.get('feature_columns', []))
            hidden_units = self.config.get('hidden_units', [64, 32])
            dropout_rate = self.config.get('dropout_rate', 0.2)
            
            if model_type == 'mlp':
                model = keras.Sequential()
                
                # 输入层
                model.add(keras.layers.Dense(hidden_units[0], activation='relu', input_shape=(input_dim,)))
                model.add(keras.layers.Dropout(dropout_rate))
                
                # 隐藏层
                for units in hidden_units[1:]:
                    model.add(keras.layers.Dense(units, activation='relu'))
                    model.add(keras.layers.Dropout(dropout_rate))
                
                # 输出层
                model.add(keras.layers.Dense(1, activation='sigmoid'))
                
                # 编译模型
                optimizer = keras.optimizers.Adam(learning_rate=self.config.get('learning_rate', 0.001))
                model.compile(
                    optimizer=optimizer,
                    loss='binary_crossentropy',
                    metrics=['accuracy']
                )
                
                return model
            else:
                logger.warning(f"不支持的TensorFlow模型类型: {model_type}")
                return None
                
        except ImportError:
            logger.warning("TensorFlow未安装")
            return None
    
    def _create_pytorch_model(self, model_type: str):
        """创建PyTorch模型"""
        try:
            import torch
            import torch.nn as nn
            
            input_dim = len(self.config.get('feature_columns', []))
            hidden_units = self.config.get('hidden_units', [64, 32])
            dropout_rate = self.config.get('dropout_rate', 0.2)
            
            if model_type == 'mlp':
                class MLP(nn.Module):
                    def __init__(self, input_dim, hidden_units, dropout_rate):
                        super(MLP, self).__init__()
                        layers = []
                        
                        # 输入层
                        layers.append(nn.Linear(input_dim, hidden_units[0]))
                        layers.append(nn.ReLU())
                        layers.append(nn.Dropout(dropout_rate))
                        
                        # 隐藏层
                        for i in range(len(hidden_units) - 1):
                            layers.append(nn.Linear(hidden_units[i], hidden_units[i + 1]))
                            layers.append(nn.ReLU())
                            layers.append(nn.Dropout(dropout_rate))
                        
                        # 输出层
                        layers.append(nn.Linear(hidden_units[-1], 1))
                        layers.append(nn.Sigmoid())
                        
                        self.model = nn.Sequential(*layers)
                    
                    def forward(self, x):
                        return self.model(x)
                
                model = MLP(input_dim, hidden_units, dropout_rate)
                return model
            else:
                logger.warning(f"不支持的PyTorch模型类型: {model_type}")
                return None
                
        except ImportError:
            logger.warning("PyTorch未安装")
            return None
    
    async def train_with_epochs(self, config: Dict[str, Any], data: Dict[str, Any], task_id: str = None) -> Dict[str, Any]:
        """执行完整的epoch训练"""
        start_time = datetime.now()
        
        try:
            # 初始化训练
            self._initialize_training(config, data)
            
            # 配置GPU资源
            gpu_config = self._configure_gpu_resources(config)
            allocated_node = None
            
            if gpu_config and self.gpu_manager.initialized:
                # 验证GPU需求
                if not self.gpu_manager.validate_gpu_requirements(gpu_config):
                    logger.warning("GPU资源需求不满足，将使用CPU训练")
                else:
                    # 分配GPU资源
                    allocated_node = self.gpu_manager.allocate_gpu_resources(gpu_config)
                    if allocated_node:
                        logger.info(f"GPU资源分配成功: {allocated_node}")
                        # 设置GPU环境
                        self._setup_gpu_environment(gpu_config)
                    else:
                        logger.warning("GPU资源分配失败，将使用CPU训练")
            
            self.training_state = TrainingState.RUNNING
            
            if task_id:
                self.progress_manager.update_training_state(task_id, self.training_state)
            
            epochs = config.get('epochs', 100)
            batch_size = config.get('batch_size', 32)
            
            # 准备数据
            X_train, y_train, X_val, y_val = self._prepare_data()
            
            # 开始epoch训练
            for epoch in range(epochs):
                if self.training_state == TrainingState.PAUSED:
                    logger.info("训练已暂停")
                    break
                
                epoch_start_time = time.time()
                
                try:
                    # 训练一个epoch
                    train_metrics = await self._train_single_epoch(epoch, X_train, y_train, batch_size)
                    
                    # 验证
                    val_metrics = await self._validate_epoch(epoch, X_val, y_val, batch_size)
                    
                    # 获取GPU监控数据
                    gpu_metrics = self._get_gpu_metrics()
                    
                    # 合并指标
                    epoch_metrics = EpochMetrics(
                        epoch=epoch,
                        train_loss=train_metrics['loss'],
                        train_accuracy=train_metrics['accuracy'],
                        val_loss=val_metrics.get('loss'),
                        val_accuracy=val_metrics.get('accuracy'),
                        learning_rate=self.learning_rate_scheduler.get_current_lr(),
                        time_per_epoch=time.time() - epoch_start_time,
                        gpu_utilization=gpu_metrics.get('gpu_utilization')
                    )
                    
                    # 记录指标
                    self._log_epoch_metrics(epoch_metrics)
                    
                    # 广播进度
                    if task_id:
                        self.progress_manager.broadcast_epoch_progress(
                            task_id, epoch, epoch_metrics.__dict__
                        )
                    
                    # 更新学习率
                    self.learning_rate_scheduler.step(epoch, val_metrics)
                    
                    # 检查早停
                    if self.early_stopping.should_stop(val_metrics['loss'], epoch):
                        logger.info(f"早停触发，在第{epoch}轮停止训练")
                        break
                    
                    # 保存最佳模型
                    if self._is_best_model(val_metrics):
                        self._save_best_model(config)
                    
                except Exception as e:
                    logger.error(f"第{epoch}轮训练失败: {str(e)}")
                    self.training_state = TrainingState.FAILED
                    raise
            
            # 训练完成
            self.training_state = TrainingState.COMPLETED
            if task_id:
                self.progress_manager.update_training_state(task_id, self.training_state)
            
            # 清理GPU资源
            if allocated_node and gpu_config:
                self.gpu_manager.cleanup_gpu_resources(allocated_node, gpu_config.gpu_count)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            return {
                'success': True,
                'task_id': task_id,
                'total_epochs': len(self.epoch_history),
                'best_epoch': self.early_stopping.get_best_epoch(),
                'final_metrics': self.epoch_history[-1].__dict__ if self.epoch_history else {},
                'training_history': [metrics.__dict__ for metrics in self.epoch_history],
                'duration': duration,
                'gpu_node': allocated_node,
                'model_path': self._get_model_path(config)
            }
            
        except Exception as e:
            self.training_state = TrainingState.FAILED
            if task_id:
                self.progress_manager.update_training_state(task_id, self.training_state)
            
            logger.error(f"训练失败: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'task_id': task_id
            }
    
    def _prepare_data(self):
        """准备训练数据"""
        # 这里应该根据实际数据格式进行数据准备
        # 简化示例
        X = np.random.randn(1000, len(self.config.get('feature_columns', [])))
        y = np.random.randint(0, 2, 1000)
        
        # 分割训练和验证数据
        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        return X_train, y_train, X_val, y_val
    
    async def _train_single_epoch(self, epoch: int, X_train, y_train, batch_size: int) -> Dict[str, float]:
        """训练单个epoch"""
        if self.tf_model:
            return await self._train_tensorflow_epoch(epoch, X_train, y_train, batch_size)
        elif self.pytorch_model:
            return await self._train_pytorch_epoch(epoch, X_train, y_train, batch_size)
        else:
            raise ValueError("没有可用的模型")
    
    async def _train_tensorflow_epoch(self, epoch: int, X_train, y_train, batch_size: int) -> Dict[str, float]:
        """TensorFlow epoch训练"""
        try:
            import tensorflow as tf
            
            # 训练一个epoch
            history = self.tf_model.fit(
                X_train, y_train,
                batch_size=batch_size,
                epochs=1,
                verbose=0
            )
            
            return {
                'loss': history.history['loss'][0],
                'accuracy': history.history['accuracy'][0]
            }
        except Exception as e:
            logger.error(f"TensorFlow训练失败: {str(e)}")
            raise
    
    async def _train_pytorch_epoch(self, epoch: int, X_train, y_train, batch_size: int) -> Dict[str, float]:
        """PyTorch epoch训练"""
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            
            # 转换为PyTorch张量
            X_train_tensor = torch.FloatTensor(X_train)
            y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
            
            # 设置优化器
            optimizer = optim.Adam(self.pytorch_model.parameters(), lr=self.learning_rate_scheduler.get_current_lr())
            criterion = nn.BCELoss()
            
            # 训练模式
            self.pytorch_model.train()
            
            total_loss = 0
            total_correct = 0
            total_samples = 0
            
            # 批次训练
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train_tensor[i:i+batch_size]
                batch_y = y_train_tensor[i:i+batch_size]
                
                # 前向传播
                optimizer.zero_grad()
                outputs = self.pytorch_model(batch_X)
                loss = criterion(outputs, batch_y)
                
                # 反向传播
                loss.backward()
                optimizer.step()
                
                # 计算指标
                total_loss += loss.item()
                predicted = (outputs > 0.5).float()
                total_correct += (predicted == batch_y).sum().item()
                total_samples += batch_y.size(0)
            
            return {
                'loss': total_loss / (len(X_train) // batch_size),
                'accuracy': total_correct / total_samples
            }
        except Exception as e:
            logger.error(f"PyTorch训练失败: {str(e)}")
            raise
    
    async def _validate_epoch(self, epoch: int, X_val, y_val, batch_size: int) -> Dict[str, float]:
        """验证单个epoch"""
        if self.tf_model:
            return await self._validate_tensorflow_epoch(epoch, X_val, y_val, batch_size)
        elif self.pytorch_model:
            return await self._validate_pytorch_epoch(epoch, X_val, y_val, batch_size)
        else:
            raise ValueError("没有可用的模型")
    
    async def _validate_tensorflow_epoch(self, epoch: int, X_val, y_val, batch_size: int) -> Dict[str, float]:
        """TensorFlow epoch验证"""
        try:
            # 验证
            val_loss, val_accuracy = self.tf_model.evaluate(X_val, y_val, batch_size=batch_size, verbose=0)
            
            return {
                'loss': val_loss,
                'accuracy': val_accuracy
            }
        except Exception as e:
            logger.error(f"TensorFlow验证失败: {str(e)}")
            raise
    
    async def _validate_pytorch_epoch(self, epoch: int, X_val, y_val, batch_size: int) -> Dict[str, float]:
        """PyTorch epoch验证"""
        try:
            import torch
            import torch.nn as nn
            
            # 转换为PyTorch张量
            X_val_tensor = torch.FloatTensor(X_val)
            y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1)
            
            criterion = nn.BCELoss()
            
            # 评估模式
            self.pytorch_model.eval()
            
            total_loss = 0
            total_correct = 0
            total_samples = 0
            
            with torch.no_grad():
                for i in range(0, len(X_val), batch_size):
                    batch_X = X_val_tensor[i:i+batch_size]
                    batch_y = y_val_tensor[i:i+batch_size]
                    
                    # 前向传播
                    outputs = self.pytorch_model(batch_X)
                    loss = criterion(outputs, batch_y)
                    
                    # 计算指标
                    total_loss += loss.item()
                    predicted = (outputs > 0.5).float()
                    total_correct += (predicted == batch_y).sum().item()
                    total_samples += batch_y.size(0)
            
            return {
                'loss': total_loss / (len(X_val) // batch_size),
                'accuracy': total_correct / total_samples
            }
        except Exception as e:
            logger.error(f"PyTorch验证失败: {str(e)}")
            raise
    
    def _log_epoch_metrics(self, metrics: EpochMetrics):
        """记录epoch指标"""
        self.epoch_history.append(metrics)
        logger.info(f"Epoch {metrics.epoch}: "
                   f"Train Loss={metrics.train_loss:.4f}, "
                   f"Train Acc={metrics.train_accuracy:.4f}, "
                   f"Val Loss={metrics.val_loss:.4f}, "
                   f"Val Acc={metrics.val_accuracy:.4f}, "
                   f"LR={metrics.learning_rate:.6f}")
    
    def _is_best_model(self, val_metrics: Dict[str, float]) -> bool:
        """判断是否为最佳模型"""
        if not self.best_metrics:
            self.best_metrics = val_metrics
            return True
        
        # 比较验证损失
        if val_metrics['loss'] < self.best_metrics['loss']:
            self.best_metrics = val_metrics
            return True
        
        return False
    
    def _save_best_model(self, config: Dict[str, Any]):
        """保存最佳模型"""
        output_path = config.get('output_path', './models')
        os.makedirs(output_path, exist_ok=True)
        
        if self.tf_model:
            self.tf_model.save(f"{output_path}/best_model.h5")
        elif self.pytorch_model:
            import torch
            torch.save(self.pytorch_model.state_dict(), f"{output_path}/best_model.pth")
        
        logger.info(f"最佳模型已保存到: {output_path}")
    
    def _get_model_path(self, config: Dict[str, Any]) -> str:
        """获取模型路径"""
        output_path = config.get('output_path', './models')
        if self.tf_model:
            return f"{output_path}/best_model.h5"
        elif self.pytorch_model:
            return f"{output_path}/best_model.pth"
        return ""
    
    def pause_training(self):
        """暂停训练"""
        self.training_state = TrainingState.PAUSED
        logger.info("训练已暂停")
    
    def resume_training(self):
        """恢复训练"""
        self.training_state = TrainingState.RUNNING
        logger.info("训练已恢复")
    
    def get_training_progress(self) -> Dict[str, Any]:
        """获取训练进度"""
        return {
            'current_epoch': self.current_epoch,
            'total_epochs': len(self.epoch_history),
            'training_state': self.training_state.value,
            'latest_metrics': self.epoch_history[-1].__dict__ if self.epoch_history else {},
            'best_metrics': self.best_metrics
        }
    
    def _configure_gpu_resources(self, config: Dict[str, Any]) -> Optional[TrainingGPUConfig]:
        """配置GPU资源"""
        try:
            # 从配置中提取GPU设置
            gpu_settings = config.get('gpu_config', {})
            
            if not gpu_settings.get('enabled', True):
                return None
            
            gpu_config = TrainingGPUConfig(
                gpu_count=gpu_settings.get('gpu_count', 1),
                gpu_type=gpu_settings.get('gpu_type', 'V100'),
                memory_gb=gpu_settings.get('memory_gb', 32.0),
                compute_ratio=gpu_settings.get('compute_ratio', 1.0),
                distributed_training=gpu_settings.get('distributed_training', False),
                mixed_precision=gpu_settings.get('mixed_precision', True),
                gpu_memory_fraction=gpu_settings.get('gpu_memory_fraction', 0.9)
            )
            
            return gpu_config
            
        except Exception as e:
            logger.error(f"配置GPU资源失败: {e}")
            return None
    
    def _setup_gpu_environment(self, gpu_config: TrainingGPUConfig):
        """设置GPU环境"""
        try:
            if self.tf_model:
                # 设置TensorFlow GPU环境
                tf_config = self.tensorflow_gpu.setup_tensorflow_gpu(gpu_config)
                logger.info(f"TensorFlow GPU环境设置完成: {tf_config}")
            elif self.pytorch_model:
                # 设置PyTorch GPU环境
                pytorch_config = self.pytorch_gpu.setup_pytorch_gpu(gpu_config)
                logger.info(f"PyTorch GPU环境设置完成: {pytorch_config}")
        except Exception as e:
            logger.error(f"设置GPU环境失败: {e}")
    
    def _get_gpu_metrics(self) -> Dict[str, Any]:
        """获取GPU监控指标"""
        try:
            if self.gpu_manager.initialized:
                monitoring_data = self.gpu_manager.get_gpu_monitoring_data()
                return {
                    'gpu_utilization': monitoring_data.get('utilization', {}).get('gpu_utilization', 0.0),
                    'gpu_memory_usage': monitoring_data.get('utilization', {}).get('gpu_memory_usage', 0.0),
                    'gpu_temperature': monitoring_data.get('utilization', {}).get('gpu_temperature', 0.0)
                }
            else:
                return {'gpu_utilization': 0.0, 'gpu_memory_usage': 0.0, 'gpu_temperature': 0.0}
        except Exception as e:
            logger.error(f"获取GPU指标失败: {e}")
            return {'gpu_utilization': 0.0, 'gpu_memory_usage': 0.0, 'gpu_temperature': 0.0}


# 全局训练管理器
epoch_training_manager = EpochDeepLearningTrainer() 
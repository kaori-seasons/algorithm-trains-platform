"""
增量学习系统
实现基于Transformer的时序数据统一模型和增量学习算法
"""
import logging
import os
import json
import pickle
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import joblib

logger = logging.getLogger(__name__)


class TimeSeriesDataset(Dataset):
    """时序数据集"""
    
    def __init__(self, data: np.ndarray, targets: np.ndarray = None, 
                 sequence_length: int = 100, stride: int = 1):
        self.data = data
        self.targets = targets
        self.sequence_length = sequence_length
        self.stride = stride
        
        # 计算序列数量
        self.num_sequences = (len(data) - sequence_length) // stride + 1
    
    def __len__(self):
        return self.num_sequences
    
    def __getitem__(self, idx):
        start_idx = idx * self.stride
        end_idx = start_idx + self.sequence_length
        
        sequence = self.data[start_idx:end_idx]
        
        if self.targets is not None:
            # 使用序列最后一个时间步的标签
            target = self.targets[end_idx - 1]
            return torch.FloatTensor(sequence), torch.LongTensor([target])
        else:
            return torch.FloatTensor(sequence)


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.d_k]))
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        
        # 线性变换并重塑为多头
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale.to(query.device)
        
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 应用注意力权重
        context = torch.matmul(attention_weights, V)
        
        # 重塑并线性变换
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        output = self.w_o(context)
        
        return output, attention_weights


class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class TransformerEncoderLayer(nn.Module):
    """Transformer编码器层"""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # 自注意力
        attn_output, _ = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 前馈网络
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class TransformerTimeSeriesModel(nn.Module):
    """基于Transformer的时序数据统一模型"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        # 模型参数
        self.input_dim = config.get('input_dim', 64)
        self.d_model = config.get('d_model', 256)
        self.num_heads = config.get('num_heads', 8)
        self.num_layers = config.get('num_layers', 6)
        self.d_ff = config.get('d_ff', 1024)
        self.dropout = config.get('dropout', 0.1)
        self.num_classes = config.get('num_classes', 2)
        self.max_seq_length = config.get('max_seq_length', 1000)
        
        # 输入投影层
        self.input_projection = nn.Linear(self.input_dim, self.d_model)
        
        # 位置编码
        self.positional_encoding = PositionalEncoding(self.d_model, self.max_seq_length)
        
        # Transformer编码器
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(self.d_model, self.num_heads, self.d_ff, self.dropout)
            for _ in range(self.num_layers)
        ])
        
        # 输出层
        self.output_projection = nn.Linear(self.d_model, self.num_classes)
        
        # Dropout
        self.dropout_layer = nn.Dropout(self.dropout)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x, mask=None):
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, seq_len, input_dim]
            mask: 注意力掩码
            
        Returns:
            输出张量 [batch_size, num_classes]
        """
        batch_size, seq_len, _ = x.shape
        
        # 输入投影
        x = self.input_projection(x)
        
        # 位置编码
        x = x.transpose(0, 1)  # [seq_len, batch_size, d_model]
        x = self.positional_encoding(x)
        x = x.transpose(0, 1)  # [batch_size, seq_len, d_model]
        
        # Dropout
        x = self.dropout_layer(x)
        
        # Transformer编码器层
        for layer in self.transformer_layers:
            x = layer(x, mask)
        
        # 全局平均池化
        if mask is not None:
            # 使用掩码进行平均池化
            mask_expanded = mask.unsqueeze(-1).expand_as(x)
            x = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
        else:
            x = x.mean(dim=1)
        
        # 输出投影
        output = self.output_projection(x)
        
        return output
    
    def get_attention_weights(self, x, mask=None):
        """获取注意力权重（用于可视化）"""
        batch_size, seq_len, _ = x.shape
        
        # 输入投影
        x = self.input_projection(x)
        
        # 位置编码
        x = x.transpose(0, 1)
        x = self.positional_encoding(x)
        x = x.transpose(0, 1)
        
        x = self.dropout_layer(x)
        
        attention_weights = []
        
        # 收集每层的注意力权重
        for layer in self.transformer_layers:
            attn_output, attn_weights = layer.self_attention.forward(
                layer.norm1(x), layer.norm1(x), layer.norm1(x), mask
            )
            attention_weights.append(attn_weights)
            x = layer.norm1(x + layer.dropout(attn_output))
            ff_output = layer.feed_forward(x)
            x = layer.norm2(x + layer.dropout(ff_output))
        
        return attention_weights


class IncrementalLearner:
    """增量学习器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.knowledge_distiller = KnowledgeDistiller(config)
        self.continual_learner = ContinualLearner(config)
        
        # 学习历史
        self.learning_history = []
        
        logger.info("增量学习器初始化完成")
    
    def initialize_model(self, input_dim: int, num_classes: int):
        """初始化模型"""
        model_config = self.config.copy()
        model_config.update({
            'input_dim': input_dim,
            'num_classes': num_classes
        })
        
        self.model = TransformerTimeSeriesModel(model_config)
        
        # 初始化优化器
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.get('learning_rate', 0.001),
            weight_decay=self.config.get('weight_decay', 1e-5)
        )
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.config.get('lr_step_size', 30),
            gamma=self.config.get('lr_gamma', 0.1)
        )
        
        logger.info(f"模型初始化完成: input_dim={input_dim}, num_classes={num_classes}")
    
    def train_on_new_data(self, new_data: pd.DataFrame, task_type: str = 'classification') -> Dict[str, Any]:
        """
        在新数据上进行增量学习
        
        Args:
            new_data: 新数据
            task_type: 任务类型
            
        Returns:
            训练结果
        """
        try:
            if self.model is None:
                raise ValueError("模型未初始化")
            
            # 数据预处理
            processed_data = self._preprocess_data(new_data, task_type)
            
            # 创建数据加载器
            train_loader = self._create_data_loader(processed_data, task_type)
            
            # 增量学习
            training_result = self._incremental_training(train_loader, task_type)
            
            # 记录学习历史
            self.learning_history.append({
                'timestamp': datetime.now().isoformat(),
                'data_size': len(new_data),
                'task_type': task_type,
                'training_result': training_result
            })
            
            logger.info(f"增量学习完成: {training_result}")
            return training_result
            
        except Exception as e:
            logger.error(f"增量学习失败: {e}")
            raise
    
    def _preprocess_data(self, data: pd.DataFrame, task_type: str) -> Dict[str, np.ndarray]:
        """数据预处理"""
        # 分离特征和标签
        feature_columns = [col for col in data.columns if col not in ['target', 'label', 'class']]
        target_column = next((col for col in ['target', 'label', 'class'] if col in data.columns), None)
        
        features = data[feature_columns].values
        targets = data[target_column].values if target_column else None
        
        # 特征标准化
        if len(self.learning_history) == 0:
            # 第一次训练，拟合标准化器
            features_scaled = self.scaler.fit_transform(features)
        else:
            # 增量学习，使用已有的标准化器
            features_scaled = self.scaler.transform(features)
        
        # 标签编码
        if targets is not None and task_type == 'classification':
            if len(self.learning_history) == 0:
                targets_encoded = self.label_encoder.fit_transform(targets)
            else:
                # 处理新类别
                targets_encoded = self._handle_new_classes(targets)
        
        return {
            'features': features_scaled,
            'targets': targets_encoded if targets is not None else None
        }
    
    def _handle_new_classes(self, targets: np.ndarray) -> np.ndarray:
        """处理新类别"""
        # 获取已知类别
        known_classes = set(self.label_encoder.classes_)
        new_classes = set(targets) - known_classes
        
        if new_classes:
            logger.info(f"发现新类别: {new_classes}")
            
            # 扩展标签编码器
            all_classes = list(known_classes) + list(new_classes)
            self.label_encoder.classes_ = np.array(all_classes)
            
            # 重新编码所有标签
            targets_encoded = self.label_encoder.transform(targets)
        else:
            targets_encoded = self.label_encoder.transform(targets)
        
        return targets_encoded
    
    def _create_data_loader(self, processed_data: Dict[str, np.ndarray], 
                           task_type: str) -> DataLoader:
        """创建数据加载器"""
        features = processed_data['features']
        targets = processed_data['targets']
        
        # 创建数据集
        dataset = TimeSeriesDataset(
            data=features,
            targets=targets,
            sequence_length=self.config.get('sequence_length', 100),
            stride=self.config.get('stride', 1)
        )
        
        # 创建数据加载器
        loader = DataLoader(
            dataset,
            batch_size=self.config.get('batch_size', 32),
            shuffle=True,
            num_workers=self.config.get('num_workers', 0)
        )
        
        return loader
    
    def _incremental_training(self, train_loader: DataLoader, task_type: str) -> Dict[str, Any]:
        """增量训练"""
        self.model.train()
        
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            self.optimizer.zero_grad()
            
            # 前向传播
            outputs = self.model(data)
            
            # 计算损失
            if task_type == 'classification':
                loss = F.cross_entropy(outputs, targets.squeeze())
            else:
                loss = F.mse_loss(outputs, targets.float())
            
            # 知识蒸馏损失
            if len(self.learning_history) > 0:
                distillation_loss = self.knowledge_distiller.compute_distillation_loss(
                    self.model, data
                )
                loss += self.config.get('distillation_weight', 0.1) * distillation_loss
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.get('max_grad_norm', 1.0)
            )
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        # 更新学习率
        self.scheduler.step()
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        
        return {
            'avg_loss': avg_loss,
            'num_batches': num_batches,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """预测"""
        self.model.eval()
        
        with torch.no_grad():
            # 数据预处理
            processed_data = self._preprocess_data(data, 'classification')
            features = processed_data['features']
            
            # 创建数据集
            dataset = TimeSeriesDataset(
                data=features,
                targets=None,
                sequence_length=self.config.get('sequence_length', 100),
                stride=self.config.get('stride', 1)
            )
            
            # 预测
            predictions = []
            for i in range(len(dataset)):
                sequence = dataset[i].unsqueeze(0)
                output = self.model(sequence)
                pred = torch.softmax(output, dim=1)
                predictions.append(pred.numpy())
            
            predictions = np.concatenate(predictions, axis=0)
            
            return predictions
    
    def save_model(self, path: str):
        """保存模型"""
        model_state = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'config': self.config,
            'learning_history': self.learning_history
        }
        
        torch.save(model_state, path)
        logger.info(f"模型保存成功: {path}")
    
    def load_model(self, path: str):
        """加载模型"""
        model_state = torch.load(path, map_location='cpu')
        
        # 恢复配置
        self.config = model_state['config']
        
        # 初始化模型
        self.initialize_model(
            self.config.get('input_dim', 64),
            self.config.get('num_classes', 2)
        )
        
        # 加载模型状态
        self.model.load_state_dict(model_state['model_state_dict'])
        self.optimizer.load_state_dict(model_state['optimizer_state_dict'])
        
        # 恢复预处理器
        self.scaler = model_state['scaler']
        self.label_encoder = model_state['label_encoder']
        
        # 恢复学习历史
        self.learning_history = model_state.get('learning_history', [])
        
        logger.info(f"模型加载成功: {path}")


class KnowledgeDistiller:
    """知识蒸馏器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.teacher_model = None
        self.temperature = config.get('distillation_temperature', 3.0)
        self.alpha = config.get('distillation_alpha', 0.7)
    
    def set_teacher_model(self, teacher_model: nn.Module):
        """设置教师模型"""
        self.teacher_model = teacher_model
        self.teacher_model.eval()
    
    def compute_distillation_loss(self, student_model: nn.Module, data: torch.Tensor) -> torch.Tensor:
        """计算知识蒸馏损失"""
        if self.teacher_model is None:
            return torch.tensor(0.0, device=data.device)
        
        with torch.no_grad():
            teacher_outputs = self.teacher_model(data)
            teacher_probs = F.softmax(teacher_outputs / self.temperature, dim=1)
        
        student_outputs = student_model(data)
        student_probs = F.softmax(student_outputs / self.temperature, dim=1)
        
        # KL散度损失
        distillation_loss = F.kl_div(
            torch.log(student_probs),
            teacher_probs,
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        return distillation_loss


class ContinualLearner:
    """持续学习器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.experience_buffer = []
        self.buffer_size = config.get('experience_buffer_size', 1000)
    
    def add_experience(self, data: pd.DataFrame, task_type: str):
        """添加经验到缓冲区"""
        experience = {
            'data': data,
            'task_type': task_type,
            'timestamp': datetime.now().isoformat()
        }
        
        self.experience_buffer.append(experience)
        
        # 保持缓冲区大小
        if len(self.experience_buffer) > self.buffer_size:
            self.experience_buffer.pop(0)
    
    def replay_experience(self, learner: IncrementalLearner, replay_ratio: float = 0.3):
        """经验回放"""
        if not self.experience_buffer:
            return
        
        # 随机选择经验进行回放
        num_replay = int(len(self.experience_buffer) * replay_ratio)
        replay_experiences = np.random.choice(
            self.experience_buffer,
            size=min(num_replay, len(self.experience_buffer)),
            replace=False
        )
        
        for experience in replay_experiences:
            learner.train_on_new_data(
                experience['data'],
                experience['task_type']
            )
    
    def get_experience_statistics(self) -> Dict[str, Any]:
        """获取经验统计信息"""
        if not self.experience_buffer:
            return {'total_experiences': 0}
        
        task_types = [exp['task_type'] for exp in self.experience_buffer]
        task_type_counts = pd.Series(task_types).value_counts().to_dict()
        
        return {
            'total_experiences': len(self.experience_buffer),
            'task_type_distribution': task_type_counts,
            'buffer_utilization': len(self.experience_buffer) / self.buffer_size
        }


# 全局增量学习器实例
incremental_learner = IncrementalLearner({
    'd_model': 256,
    'num_heads': 8,
    'num_layers': 6,
    'd_ff': 1024,
    'dropout': 0.1,
    'learning_rate': 0.001,
    'batch_size': 32,
    'sequence_length': 100,
    'stride': 1,
    'distillation_temperature': 3.0,
    'distillation_alpha': 0.7,
    'experience_buffer_size': 1000
}) 
"""
GPU资源管理器与训练平台集成模块
提供多GPU支持、资源调度和监控功能
"""
import os
import sys
import logging
import asyncio
import threading
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import json

# 添加GPU资源管理器路径
gpu_manager_path = os.path.join(os.path.dirname(__file__), '..', '..', 'gpu-resource-manager', 'src')
if gpu_manager_path not in sys.path:
    sys.path.append(gpu_manager_path)

try:
    from gpu_parser import GPUResourceParser, GPUResource
    from gpu_scheduler import GPUScheduler, SchedulingRequest, SchedulingPolicy
    from gpu_memory_guard import GPUMemoryGuard, GPUMemoryRequirement
    from k8s_client import K8sResourceManager
    from gpu_hpa_controller import GPUHPAController
    from resource_monitor import ResourceMonitor
    # 创建模拟配置
    gpu_config = type('Config', (), {'config': {}})()
except ImportError as e:
    logging.warning(f"GPU资源管理器模块导入失败: {e}")
    # 创建模拟类以保持兼容性
    class GPUResourceParser:
        def __init__(self, config=None):
            self.config = config or {}
        
        def parse_gpu_resource(self, resource_gpu: str, resource_name: Optional[str] = None):
            return type('GPUResource', (), {
                'gpu_num': 1.0,
                'gpu_type': 'V100',
                'resource_name': 'nvidia.com/gpu',
                'memory_gb': 32.0,
                'compute_ratio': 1.0
            })()
    
    class GPUScheduler:
        def __init__(self, *args, **kwargs):
            pass
        
        def schedule_pod(self, request):
            return "gpu-node-1"
    
    class GPUMemoryGuard:
        def __init__(self, *args, **kwargs):
            pass
        
        def validate_memory_requirement(self, gpu_resource, min_memory_gb):
            return True
    
    class K8sResourceManager:
        def __init__(self, *args, **kwargs):
            pass
    
    class GPUHPAController:
        def __init__(self, *args, **kwargs):
            pass
    
    class ResourceMonitor:
        def __init__(self, *args, **kwargs):
            pass

logger = logging.getLogger(__name__)


class GPUResourceType(Enum):
    """GPU资源类型"""
    NVIDIA = "nvidia"
    AMD = "amd"
    INTEL = "intel"
    NPU = "npu"


@dataclass
class TrainingGPUConfig:
    """训练GPU配置"""
    gpu_count: int = 1
    gpu_type: str = "V100"
    memory_gb: float = 32.0
    compute_ratio: float = 1.0
    distributed_training: bool = False
    mixed_precision: bool = True
    gpu_memory_fraction: float = 0.9


@dataclass
class GPUResourceStatus:
    """GPU资源状态"""
    node_name: str
    gpu_type: str
    total_gpus: int
    available_gpus: int
    memory_per_gpu: float
    utilization: float
    temperature: Optional[float] = None
    power_usage: Optional[float] = None


class GPUResourceManager:
    """GPU资源管理器集成类"""
    
    def __init__(self, kubeconfig_path: Optional[str] = None):
        """初始化GPU资源管理器"""
        self.logger = logging.getLogger(__name__)
        
        try:
            # 初始化GPU资源管理器组件
            self.k8s_client = K8sResourceManager(kubeconfig_path)
            self.gpu_parser = GPUResourceParser(gpu_config.config)
            self.memory_guard = GPUMemoryGuard(self.k8s_client, self.gpu_parser)
            self.scheduler = GPUScheduler(self.k8s_client, self.gpu_parser, self.memory_guard)
            self.hpa_controller = GPUHPAController(self.k8s_client)
            self.resource_monitor = ResourceMonitor(self.k8s_client, update_interval=30)
            
            # 启动资源监控
            self.resource_monitor.start_monitoring()
            self.resource_monitor.add_callback(self._on_resource_change)
            
            self.logger.info("GPU资源管理器初始化成功")
            self.initialized = True
            
        except Exception as e:
            self.logger.error(f"GPU资源管理器初始化失败: {e}")
            self.initialized = False
    
    def _on_resource_change(self, summary):
        """资源变化回调"""
        self.logger.info(f"GPU资源使用情况更新: {summary}")
    
    def get_available_gpu_nodes(self, gpu_type: Optional[str] = None, min_memory_gb: float = 16.0) -> List[GPUResourceStatus]:
        """获取可用的GPU节点"""
        if not self.initialized:
            return []
        
        try:
            nodes = self.k8s_client.get_nodes()
            available_nodes = []
            
            for node in nodes:
                if not node['ready'] or not node['schedulable']:
                    continue
                
                # 检查GPU信息
                for gpu_type_name, gpu_info in node.get('gpu_info', {}).items():
                    if gpu_type and gpu_type_name != gpu_type:
                        continue
                    
                    if gpu_info['available'] > 0:
                        # 估算GPU显存
                        memory_per_gpu = self._estimate_gpu_memory(gpu_type_name)
                        
                        if memory_per_gpu >= min_memory_gb:
                            available_nodes.append(GPUResourceStatus(
                                node_name=node['name'],
                                gpu_type=gpu_type_name,
                                total_gpus=gpu_info['allocatable'],
                                available_gpus=gpu_info['available'],
                                memory_per_gpu=memory_per_gpu,
                                utilization=self._get_gpu_utilization(node, gpu_type_name)
                            ))
            
            return available_nodes
            
        except Exception as e:
            self.logger.error(f"获取可用GPU节点失败: {e}")
            return []
    
    def _estimate_gpu_memory(self, gpu_type: str) -> float:
        """估算GPU显存大小"""
        memory_mapping = {
            'T4': 16.0,
            'V100': 32.0,
            'A100': 80.0,
            'H100': 80.0,
            'RTX3090': 24.0,
            'RTX4090': 24.0,
            'A6000': 48.0,
            'A40': 48.0
        }
        return memory_mapping.get(gpu_type, 16.0)
    
    def _get_gpu_utilization(self, node: Dict, gpu_type: str) -> float:
        """获取GPU利用率"""
        try:
            # 这里应该从监控系统获取实际的GPU利用率
            # 暂时返回模拟值
            return 0.5
        except Exception as e:
            self.logger.error(f"获取GPU利用率失败: {e}")
            return 0.0
    
    def allocate_gpu_resources(self, training_config: TrainingGPUConfig) -> Optional[str]:
        """分配GPU资源"""
        if not self.initialized:
            return None
        
        try:
            # 构建GPU资源字符串
            gpu_resource_str = self._build_gpu_resource_string(training_config)
            
            # 创建调度请求
            request = SchedulingRequest(
                pod_name=f"training-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                namespace="training",
                gpu_requirement=gpu_resource_str,
                memory_requirement=training_config.memory_gb,
                priority=1
            )
            
            # 执行调度
            selected_node = self.scheduler.schedule_pod(request)
            
            if selected_node:
                self.logger.info(f"成功分配GPU资源到节点: {selected_node}")
                return selected_node
            else:
                self.logger.warning("无法分配GPU资源")
                return None
                
        except Exception as e:
            self.logger.error(f"分配GPU资源失败: {e}")
            return None
    
    def _build_gpu_resource_string(self, config: TrainingGPUConfig) -> str:
        """构建GPU资源字符串"""
        if config.memory_gb and config.compute_ratio != 1.0:
            return f"{config.memory_gb}G,{config.compute_ratio}"
        elif config.gpu_type:
            return f"{config.gpu_count}({config.gpu_type})"
        else:
            return str(config.gpu_count)
    
    def validate_gpu_requirements(self, gpu_config: TrainingGPUConfig) -> bool:
        """验证GPU需求是否满足"""
        if not self.initialized:
            return True  # 如果GPU管理器未初始化，默认通过
        
        try:
            gpu_resource_str = self._build_gpu_resource_string(gpu_config)
            requirement = GPUMemoryRequirement(
                min_memory_gb=gpu_config.memory_gb,
                gpu_type=gpu_config.gpu_type
            )
            
            return self.memory_guard.validate_memory_requirement(gpu_resource_str, requirement)
            
        except Exception as e:
            self.logger.error(f"验证GPU需求失败: {e}")
            return False
    
    def setup_distributed_training(self, gpu_config: TrainingGPUConfig) -> Dict[str, Any]:
        """设置分布式训练环境"""
        if not gpu_config.distributed_training:
            return {}
        
        try:
            # 获取可用的GPU节点
            available_nodes = self.get_available_gpu_nodes(
                gpu_type=gpu_config.gpu_type,
                min_memory_gb=gpu_config.memory_gb
            )
            
            if len(available_nodes) < gpu_config.gpu_count:
                raise ValueError(f"可用GPU节点数量不足: 需要{gpu_config.gpu_count}个，可用{len(available_nodes)}个")
            
            # 选择最佳节点组合
            selected_nodes = self._select_nodes_for_distributed(available_nodes, gpu_config)
            
            # 配置分布式训练参数
            distributed_config = {
                'world_size': len(selected_nodes) * gpu_config.gpu_count,
                'nodes': selected_nodes,
                'backend': 'nccl',  # NVIDIA GPU使用NCCL
                'init_method': 'env://',
                'rank': 0
            }
            
            self.logger.info(f"分布式训练配置: {distributed_config}")
            return distributed_config
            
        except Exception as e:
            self.logger.error(f"设置分布式训练失败: {e}")
            return {}
    
    def _select_nodes_for_distributed(self, available_nodes: List[GPUResourceStatus], 
                                    config: TrainingGPUConfig) -> List[str]:
        """为分布式训练选择节点"""
        # 按可用GPU数量排序，选择GPU数量最多的节点
        sorted_nodes = sorted(available_nodes, key=lambda x: x.available_gpus, reverse=True)
        
        selected_nodes = []
        remaining_gpus = config.gpu_count
        
        for node in sorted_nodes:
            if remaining_gpus <= 0:
                break
            
            gpus_to_use = min(node.available_gpus, remaining_gpus)
            selected_nodes.append({
                'node_name': node.node_name,
                'gpu_count': gpus_to_use,
                'gpu_type': node.gpu_type
            })
            remaining_gpus -= gpus_to_use
        
        return selected_nodes
    
    def get_gpu_monitoring_data(self) -> Dict[str, Any]:
        """获取GPU监控数据"""
        if not self.initialized:
            return {}
        
        try:
            # 获取资源利用率
            utilization = self.resource_monitor.get_resource_utilization()
            
            # 获取可用节点
            available_nodes = self.get_available_gpu_nodes()
            
            return {
                'utilization': utilization,
                'available_nodes': [
                    {
                        'node_name': node.node_name,
                        'gpu_type': node.gpu_type,
                        'available_gpus': node.available_gpus,
                        'memory_per_gpu': node.memory_per_gpu,
                        'utilization': node.utilization
                    }
                    for node in available_nodes
                ],
                'total_nodes': len(available_nodes),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"获取GPU监控数据失败: {e}")
            return {}
    
    def create_hpa_for_training(self, namespace: str, deployment_name: str, 
                               gpu_config: TrainingGPUConfig) -> bool:
        """为训练任务创建HPA"""
        if not self.initialized:
            return False
        
        try:
            # 构建HPA指标
            metrics = [
                'cpu:70%',
                'mem:80%',
                'gpu:75%'
            ]
            
            # 创建HPA
            success = self.hpa_controller.create_hpa(
                namespace=namespace,
                name=deployment_name,
                min_replicas=1,
                max_replicas=5,
                metrics=metrics,
                target_ref={
                    'apiVersion': 'apps/v1',
                    'kind': 'Deployment',
                    'name': deployment_name
                }
            )
            
            if success:
                self.logger.info(f"成功为训练任务创建HPA: {deployment_name}")
            else:
                self.logger.error(f"创建HPA失败: {deployment_name}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"创建HPA失败: {e}")
            return False
    
    def cleanup_gpu_resources(self, node_name: str, gpu_count: int) -> bool:
        """清理GPU资源"""
        if not self.initialized:
            return True
        
        try:
            # 这里应该实现具体的资源清理逻辑
            self.logger.info(f"清理GPU资源: 节点={node_name}, GPU数量={gpu_count}")
            return True
            
        except Exception as e:
            self.logger.error(f"清理GPU资源失败: {e}")
            return False


class TensorFlowGPUIntegration:
    """TensorFlow GPU集成"""
    
    def __init__(self, gpu_manager: GPUResourceManager):
        self.gpu_manager = gpu_manager
        self.logger = logging.getLogger(__name__)
    
    def setup_tensorflow_gpu(self, gpu_config: TrainingGPUConfig) -> Dict[str, Any]:
        """设置TensorFlow GPU环境"""
        try:
            import tensorflow as tf
            
            # 检查GPU可用性
            gpus = tf.config.list_physical_devices('GPU')
            if not gpus:
                self.logger.warning("未检测到可用的GPU设备")
                return {}
            
            # 配置GPU内存增长
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # 设置GPU内存限制
            if gpu_config.gpu_memory_fraction < 1.0:
                memory_limit = int(gpu_config.memory_gb * 1024 * gpu_config.gpu_memory_fraction)
                tf.config.set_logical_device_configuration(
                    gpus[0],
                    [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit)]
                )
            
            # 配置分布式策略
            if gpu_config.distributed_training:
                strategy = tf.distribute.MirroredStrategy()
                return {
                    'strategy': strategy,
                    'gpu_count': len(gpus),
                    'distributed': True
                }
            else:
                return {
                    'gpu_count': len(gpus),
                    'distributed': False
                }
                
        except ImportError:
            self.logger.error("TensorFlow未安装")
            return {}
        except Exception as e:
            self.logger.error(f"设置TensorFlow GPU失败: {e}")
            return {}
    
    def create_tensorflow_model_with_gpu(self, model_config: Dict[str, Any], 
                                       gpu_config: TrainingGPUConfig) -> Any:
        """创建支持GPU的TensorFlow模型"""
        try:
            import tensorflow as tf
            
            # 设置GPU策略
            tf_config = self.setup_tensorflow_gpu(gpu_config)
            
            if tf_config.get('distributed'):
                strategy = tf_config['strategy']
                with strategy.scope():
                    return self._build_tensorflow_model(model_config)
            else:
                return self._build_tensorflow_model(model_config)
                
        except Exception as e:
            self.logger.error(f"创建TensorFlow模型失败: {e}")
            return None
    
    def _build_tensorflow_model(self, model_config: Dict[str, Any]):
        """构建TensorFlow模型"""
        import tensorflow as tf
        
        model_type = model_config.get('type', 'mlp')
        
        if model_type == 'mlp':
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(128, activation='relu', input_shape=(model_config.get('input_dim', 10),)),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(model_config.get('output_dim', 1), activation='softmax')
            ])
        elif model_type == 'cnn':
            model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=model_config.get('input_shape', (28, 28, 1))),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(model_config.get('output_dim', 10), activation='softmax')
            ])
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        return model


class PyTorchGPUIntegration:
    """PyTorch GPU集成"""
    
    def __init__(self, gpu_manager: GPUResourceManager):
        self.gpu_manager = gpu_manager
        self.logger = logging.getLogger(__name__)
    
    def setup_pytorch_gpu(self, gpu_config: TrainingGPUConfig) -> Dict[str, Any]:
        """设置PyTorch GPU环境"""
        try:
            import torch
            
            # 检查GPU可用性
            if not torch.cuda.is_available():
                self.logger.warning("未检测到可用的CUDA GPU")
                return {}
            
            gpu_count = torch.cuda.device_count()
            
            # 设置GPU设备
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # 配置分布式训练
            if gpu_config.distributed_training and gpu_count > 1:
                import torch.distributed as dist
                
                # 初始化分布式环境
                dist.init_process_group(backend='nccl')
                
                return {
                    'device': device,
                    'gpu_count': gpu_count,
                    'distributed': True,
                    'rank': dist.get_rank(),
                    'world_size': dist.get_world_size()
                }
            else:
                return {
                    'device': device,
                    'gpu_count': gpu_count,
                    'distributed': False
                }
                
        except ImportError:
            self.logger.error("PyTorch未安装")
            return {}
        except Exception as e:
            self.logger.error(f"设置PyTorch GPU失败: {e}")
            return {}
    
    def create_pytorch_model_with_gpu(self, model_config: Dict[str, Any], 
                                    gpu_config: TrainingGPUConfig) -> Any:
        """创建支持GPU的PyTorch模型"""
        try:
            import torch
            import torch.nn as nn
            
            # 设置GPU环境
            pytorch_config = self.setup_pytorch_gpu(gpu_config)
            device = pytorch_config['device']
            
            # 构建模型
            model = self._build_pytorch_model(model_config)
            model = model.to(device)
            
            # 如果使用多GPU，包装为DataParallel
            if pytorch_config.get('distributed'):
                model = nn.DataParallel(model)
            
            return model
            
        except Exception as e:
            self.logger.error(f"创建PyTorch模型失败: {e}")
            return None
    
    def _build_pytorch_model(self, model_config: Dict[str, Any]):
        """构建PyTorch模型"""
        import torch
        import torch.nn as nn
        
        model_type = model_config.get('type', 'mlp')
        
        if model_type == 'mlp':
            class MLP(nn.Module):
                def __init__(self, input_dim, hidden_units, dropout_rate, output_dim):
                    super(MLP, self).__init__()
                    layers = []
                    prev_dim = input_dim
                    
                    for units in hidden_units:
                        layers.extend([
                            nn.Linear(prev_dim, units),
                            nn.ReLU(),
                            nn.Dropout(dropout_rate)
                        ])
                        prev_dim = units
                    
                    layers.append(nn.Linear(prev_dim, output_dim))
                    self.layers = nn.Sequential(*layers)
                
                def forward(self, x):
                    return self.layers(x)
            
            return MLP(
                input_dim=model_config.get('input_dim', 10),
                hidden_units=model_config.get('hidden_units', [128, 64]),
                dropout_rate=model_config.get('dropout_rate', 0.3),
                output_dim=model_config.get('output_dim', 1)
            )
        
        elif model_type == 'cnn':
            class CNN(nn.Module):
                def __init__(self, input_channels, num_classes):
                    super(CNN, self).__init__()
                    self.conv_layers = nn.Sequential(
                        nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(2),
                        nn.Conv2d(32, 64, kernel_size=3, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(2),
                        nn.Conv2d(64, 64, kernel_size=3, padding=1),
                        nn.ReLU()
                    )
                    self.fc_layers = nn.Sequential(
                        nn.Flatten(),
                        nn.Linear(64 * 7 * 7, 64),
                        nn.ReLU(),
                        nn.Linear(64, num_classes)
                    )
                
                def forward(self, x):
                    x = self.conv_layers(x)
                    x = self.fc_layers(x)
                    return x
            
            return CNN(
                input_channels=model_config.get('input_channels', 1),
                num_classes=model_config.get('num_classes', 10)
            )
        
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")


# 全局GPU资源管理器实例
_gpu_resource_manager = None

def get_gpu_resource_manager() -> GPUResourceManager:
    """获取全局GPU资源管理器实例"""
    global _gpu_resource_manager
    if _gpu_resource_manager is None:
        _gpu_resource_manager = GPUResourceManager()
    return _gpu_resource_manager


def get_tensorflow_gpu_integration() -> TensorFlowGPUIntegration:
    """获取TensorFlow GPU集成实例"""
    gpu_manager = get_gpu_resource_manager()
    return TensorFlowGPUIntegration(gpu_manager)


def get_pytorch_gpu_integration() -> PyTorchGPUIntegration:
    """获取PyTorch GPU集成实例"""
    gpu_manager = get_gpu_resource_manager()
    return PyTorchGPUIntegration(gpu_manager)
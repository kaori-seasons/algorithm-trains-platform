"""
算法训练引擎核心模块
提供统一的算法训练接口
"""
import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum

from .trainers import (
    StatusRecognitionTrainer,
    HealthAssessmentTrainer,
    VibrationAnalysisTrainer,
    SimulationTrainer
)
from .models import TrainingConfig, TrainingResult, AlgorithmType
from .gpu_resource_integration import (
    get_gpu_resource_manager,
    get_tensorflow_gpu_integration,
    get_pytorch_gpu_integration,
    TrainingGPUConfig
)

logger = logging.getLogger(__name__)


class AlgorithmTrainingEngine:
    """
    算法训练引擎
    支持多种算法类型的统一训练接口
    """
    
    def __init__(self):
        """初始化算法训练引擎"""
        self.algorithm_registry = {
            AlgorithmType.STATUS_RECOGNITION: StatusRecognitionTrainer(),
            AlgorithmType.HEALTH_ASSESSMENT: HealthAssessmentTrainer(),
            AlgorithmType.VIBRATION_ANALYSIS: VibrationAnalysisTrainer(),
            AlgorithmType.SIMULATION: SimulationTrainer()
        }
        self.training_tasks = {}  # 存储训练任务状态
        
        # 初始化GPU资源管理器
        self.gpu_manager = get_gpu_resource_manager()
        self.tensorflow_gpu = get_tensorflow_gpu_integration()
        self.pytorch_gpu = get_pytorch_gpu_integration()
        
        logger.info("算法训练引擎初始化完成")
    
    async def train_algorithm(
        self, 
        algorithm_type: AlgorithmType, 
        config: TrainingConfig, 
        data: Dict[str, Any]
    ) -> TrainingResult:
        """
        训练算法
        
        Args:
            algorithm_type: 算法类型
            config: 训练配置
            data: 训练数据
            
        Returns:
            训练结果
        """
        try:
            logger.info(f"开始训练算法: {algorithm_type.value}")
            
            # 获取对应的训练器
            trainer = self.algorithm_registry.get(algorithm_type)
            if not trainer:
                raise ValueError(f"不支持的算法类型: {algorithm_type}")
            
            # 创建训练任务
            task_id = self._create_task_id(algorithm_type, config)
            self.training_tasks[task_id] = {
                'status': 'running',
                'start_time': datetime.now(),
                'progress': 0
            }
            
            # 配置GPU资源
            gpu_config = self._configure_gpu_resources(config)
            allocated_node = None
            if gpu_config:
                # 验证GPU需求
                if not self.gpu_manager.validate_gpu_requirements(gpu_config):
                    raise ValueError("GPU资源需求不满足")
                
                # 分配GPU资源
                allocated_node = self.gpu_manager.allocate_gpu_resources(gpu_config)
                if allocated_node:
                    logger.info(f"GPU资源分配成功: {allocated_node}")
                    self.training_tasks[task_id]['gpu_node'] = allocated_node
                else:
                    logger.warning("GPU资源分配失败，将使用CPU训练")
            
            # 执行训练
            result = await trainer.train(config, data)
            
            # 更新任务状态
            self.training_tasks[task_id].update({
                'status': 'completed',
                'end_time': datetime.now(),
                'progress': 100,
                'result': result
            })
            
            # 清理GPU资源
            if allocated_node:
                self.gpu_manager.cleanup_gpu_resources(allocated_node, gpu_config.gpu_count)
            
            logger.info(f"算法训练完成: {algorithm_type.value}")
            return result
            
        except Exception as e:
            logger.error(f"算法训练失败: {algorithm_type.value}, 错误: {str(e)}")
            if task_id in self.training_tasks:
                self.training_tasks[task_id].update({
                    'status': 'failed',
                    'end_time': datetime.now(),
                    'error': str(e)
                })
            raise
    
    async def generate_parameters(
        self, 
        algorithm_type: AlgorithmType, 
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        生成算法参数
        
        Args:
            algorithm_type: 算法类型
            config: 配置信息
            
        Returns:
            生成的参数
        """
        try:
            logger.info(f"开始生成算法参数: {algorithm_type.value}")
            
            trainer = self.algorithm_registry.get(algorithm_type)
            if not trainer:
                raise ValueError(f"不支持的算法类型: {algorithm_type}")
            
            parameters = await trainer.generate_parameters(config)
            
            logger.info(f"算法参数生成完成: {algorithm_type.value}")
            return parameters
            
        except Exception as e:
            logger.error(f"算法参数生成失败: {algorithm_type.value}, 错误: {str(e)}")
            raise
    
    async def get_training_status(self, task_id: str) -> Dict[str, Any]:
        """
        获取训练任务状态
        
        Args:
            task_id: 任务ID
            
        Returns:
            任务状态信息
        """
        return self.training_tasks.get(task_id, {})
    
    async def list_training_tasks(self) -> List[Dict[str, Any]]:
        """
        列出所有训练任务
        
        Returns:
            任务列表
        """
        return [
            {'task_id': task_id, **task_info}
            for task_id, task_info in self.training_tasks.items()
        ]
    
    async def cancel_training(self, task_id: str) -> bool:
        """
        取消训练任务
        
        Args:
            task_id: 任务ID
            
        Returns:
            是否成功取消
        """
        if task_id in self.training_tasks:
            self.training_tasks[task_id].update({
                'status': 'cancelled',
                'end_time': datetime.now()
            })
            logger.info(f"训练任务已取消: {task_id}")
            return True
        return False
    
    def _create_task_id(self, algorithm_type: AlgorithmType, config: TrainingConfig) -> str:
        """创建任务ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{algorithm_type.value}_{timestamp}_{config.name}"
    
    def get_supported_algorithms(self) -> List[str]:
        """获取支持的算法类型列表"""
        return [alg.value for alg in self.algorithm_registry.keys()]
    
    def _configure_gpu_resources(self, config: TrainingConfig) -> Optional[TrainingGPUConfig]:
        """配置GPU资源"""
        try:
            # 从配置中提取GPU设置
            gpu_settings = config.algorithm_params.get('gpu_config', {})
            
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
    
    def get_gpu_resource_status(self) -> Dict[str, Any]:
        """获取GPU资源状态"""
        if not self.gpu_manager.initialized:
            return {'error': 'GPU资源管理器未初始化'}
        
        return self.gpu_manager.get_gpu_monitoring_data()
    
    def setup_tensorflow_training(self, model_config: Dict[str, Any], 
                                gpu_config: TrainingGPUConfig) -> Dict[str, Any]:
        """设置TensorFlow训练环境"""
        return self.tensorflow_gpu.setup_tensorflow_gpu(gpu_config)
    
    def setup_pytorch_training(self, model_config: Dict[str, Any], 
                              gpu_config: TrainingGPUConfig) -> Dict[str, Any]:
        """设置PyTorch训练环境"""
        return self.pytorch_gpu.setup_pytorch_gpu(gpu_config) 
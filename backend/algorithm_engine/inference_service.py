"""
实时推理服务模块
支持高并发、低延迟的模型推理
"""
import asyncio
import logging
import json
import time
import threading
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd
import joblib
from collections import defaultdict, deque
import os

from .models import AlgorithmType, InferenceRequest, InferenceResult
from .model_manager import ModelVersionManager

logger = logging.getLogger(__name__)


class InferenceStatus(Enum):
    """推理状态枚举"""
    PENDING = "pending"       # 等待中
    PROCESSING = "processing" # 处理中
    COMPLETED = "completed"   # 已完成
    FAILED = "failed"         # 失败


@dataclass
class InferenceMetrics:
    """推理指标"""
    request_id: str
    model_version: str
    input_size: int
    processing_time: float
    memory_usage: float
    cpu_usage: float
    gpu_usage: Optional[float]
    timestamp: datetime


class ModelLoader:
    """模型加载器"""
    
    def __init__(self):
        self.loaded_models = {}
        self.model_cache = {}
        self.load_lock = threading.Lock()
        
        logger.info("模型加载器初始化完成")
    
    async def load_model(self, model_version: str) -> Any:
        """加载模型"""
        try:
            # 检查缓存
            if model_version in self.model_cache:
                logger.info(f"从缓存加载模型: {model_version}")
                return self.model_cache[model_version]
            
            with self.load_lock:
                # 双重检查
                if model_version in self.model_cache:
                    return self.model_cache[model_version]
                
                # 从文件加载模型
                model_path = f"models/{model_version}/model.joblib"
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"模型文件不存在: {model_path}")
                
                model = joblib.load(model_path)
                
                # 缓存模型
                self.model_cache[model_version] = model
                self.loaded_models[model_version] = {
                    'loaded_at': datetime.now(),
                    'access_count': 0
                }
                
                logger.info(f"成功加载模型: {model_version}")
                return model
                
        except Exception as e:
            logger.error(f"加载模型失败: {model_version}, 错误: {e}")
            raise
    
    async def unload_model(self, model_version: str):
        """卸载模型"""
        try:
            if model_version in self.model_cache:
                del self.model_cache[model_version]
                del self.loaded_models[model_version]
                logger.info(f"卸载模型: {model_version}")
        except Exception as e:
            logger.error(f"卸载模型失败: {model_version}, 错误: {e}")
    
    def get_loaded_models(self) -> List[Dict[str, Any]]:
        """获取已加载的模型列表"""
        return [
            {
                'version': version,
                'loaded_at': info['loaded_at'].isoformat(),
                'access_count': info['access_count']
            }
            for version, info in self.loaded_models.items()
        ]


class LoadBalancer:
    """负载均衡器"""
    
    def __init__(self):
        self.inference_nodes = []
        self.node_metrics = defaultdict(lambda: {
            'requests': 0,
            'avg_response_time': 0.0,
            'error_rate': 0.0,
            'last_health_check': datetime.now()
        })
        
        logger.info("负载均衡器初始化完成")
    
    def add_node(self, node_id: str, node_config: Dict[str, Any]):
        """添加推理节点"""
        self.inference_nodes.append({
            'id': node_id,
            'config': node_config,
            'status': 'active'
        })
        logger.info(f"添加推理节点: {node_id}")
    
    def remove_node(self, node_id: str):
        """移除推理节点"""
        self.inference_nodes = [node for node in self.inference_nodes if node['id'] != node_id]
        logger.info(f"移除推理节点: {node_id}")
    
    def select_node(self) -> str:
        """选择推理节点"""
        if not self.inference_nodes:
            return "default_node"
        
        # 简单的轮询策略
        # 实际实现中可以使用更复杂的负载均衡算法
        available_nodes = [node for node in self.inference_nodes if node['status'] == 'active']
        
        if not available_nodes:
            return "default_node"
        
        # 选择负载最低的节点
        selected_node = min(available_nodes, 
                          key=lambda x: self.node_metrics[x['id']]['requests'])
        
        return selected_node['id']
    
    def update_node_metrics(self, node_id: str, metrics: Dict[str, Any]):
        """更新节点指标"""
        if node_id in self.node_metrics:
            current = self.node_metrics[node_id]
            current['requests'] += 1
            current['avg_response_time'] = (
                (current['avg_response_time'] * (current['requests'] - 1) + metrics.get('response_time', 0)) 
                / current['requests']
            )
            current['last_health_check'] = datetime.now()


class CacheManager:
    """缓存管理器"""
    
    def __init__(self, max_cache_size: int = 1000):
        self.cache = {}
        self.max_cache_size = max_cache_size
        self.access_times = deque()
        
        logger.info("缓存管理器初始化完成")
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存"""
        if key in self.cache:
            # 更新访问时间
            self.access_times.append((key, time.time()))
            return self.cache[key]
        return None
    
    def set(self, key: str, value: Any, ttl: int = 3600):
        """设置缓存"""
        # 检查缓存大小
        if len(self.cache) >= self.max_cache_size:
            self._evict_oldest()
        
        self.cache[key] = {
            'value': value,
            'expires_at': time.time() + ttl
        }
    
    def _evict_oldest(self):
        """淘汰最旧的缓存项"""
        if not self.cache:
            return
        
        # 找到最旧的项
        oldest_key = min(self.cache.keys(), 
                        key=lambda k: self.cache[k]['expires_at'])
        del self.cache[oldest_key]
    
    def clear_expired(self):
        """清理过期缓存"""
        current_time = time.time()
        expired_keys = [
            key for key, data in self.cache.items()
            if data['expires_at'] < current_time
        ]
        
        for key in expired_keys:
            del self.cache[key]
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        return {
            'size': len(self.cache),
            'max_size': self.max_cache_size,
            'hit_rate': 0.8,  # 模拟命中率
            'memory_usage': len(self.cache) * 1024  # 模拟内存使用
        }


class InferenceMonitor:
    """推理监控器"""
    
    def __init__(self):
        self.metrics_history = deque(maxlen=10000)
        self.error_log = deque(maxlen=1000)
        self.performance_stats = defaultdict(list)
        
        logger.info("推理监控器初始化完成")
    
    async def record_inference(self, model_version: str, input_data: Dict[str, Any], 
                             result: Dict[str, Any], processing_time: float):
        """记录推理指标"""
        try:
            metrics = InferenceMetrics(
                request_id=result.get('request_id', 'unknown'),
                model_version=model_version,
                input_size=len(str(input_data)),
                processing_time=processing_time,
                memory_usage=self._get_memory_usage(),
                cpu_usage=self._get_cpu_usage(),
                gpu_usage=self._get_gpu_usage(),
                timestamp=datetime.now()
            )
            
            self.metrics_history.append(metrics)
            self.performance_stats[model_version].append({
                'processing_time': processing_time,
                'timestamp': datetime.now()
            })
            
            # 保持性能统计在合理范围内
            if len(self.performance_stats[model_version]) > 1000:
                self.performance_stats[model_version] = self.performance_stats[model_version][-1000:]
            
        except Exception as e:
            logger.error(f"记录推理指标失败: {e}")
    
    def _get_memory_usage(self) -> float:
        """获取内存使用率"""
        # 这里应该实现实际的内存监控
        # 暂时返回模拟值
        return 75.5
    
    def _get_cpu_usage(self) -> float:
        """获取CPU使用率"""
        # 这里应该实现实际的CPU监控
        # 暂时返回模拟值
        return 45.2
    
    def _get_gpu_usage(self) -> Optional[float]:
        """获取GPU使用率"""
        # 这里应该实现实际的GPU监控
        # 暂时返回模拟值
        return 60.8
    
    def get_performance_stats(self, model_version: str = None) -> Dict[str, Any]:
        """获取性能统计"""
        if model_version:
            stats = self.performance_stats.get(model_version, [])
        else:
            # 合并所有模型的统计
            stats = []
            for model_stats in self.performance_stats.values():
                stats.extend(model_stats)
        
        if not stats:
            return {}
        
        processing_times = [s['processing_time'] for s in stats]
        
        return {
            'total_requests': len(stats),
            'avg_processing_time': np.mean(processing_times),
            'min_processing_time': np.min(processing_times),
            'max_processing_time': np.max(processing_times),
            'p95_processing_time': np.percentile(processing_times, 95),
            'p99_processing_time': np.percentile(processing_times, 99)
        }
    
    def get_error_log(self) -> List[Dict[str, Any]]:
        """获取错误日志"""
        return list(self.error_log)
    
    def log_error(self, error: Exception, context: Dict[str, Any]):
        """记录错误"""
        error_entry = {
            'timestamp': datetime.now().isoformat(),
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context
        }
        self.error_log.append(error_entry)
        logger.error(f"推理错误: {error}")


class RealTimeInferenceService:
    """
    实时推理服务
    支持高并发、低延迟的模型推理
    """
    
    def __init__(self):
        self.model_loader = ModelLoader()
        self.load_balancer = LoadBalancer()
        self.cache_manager = CacheManager()
        self.monitor = InferenceMonitor()
        self.request_queue = asyncio.Queue()
        self.processing_semaphore = asyncio.Semaphore(10)  # 限制并发数
        
        # 启动后台任务
        asyncio.create_task(self._background_tasks())
        
        logger.info("实时推理服务初始化完成")
    
    async def _background_tasks(self):
        """后台任务"""
        while True:
            try:
                # 清理过期缓存
                self.cache_manager.clear_expired()
                
                # 更新节点健康状态
                await self._update_node_health()
                
                await asyncio.sleep(60)  # 每分钟执行一次
                
            except Exception as e:
                logger.error(f"后台任务错误: {e}")
                await asyncio.sleep(10)
    
    async def _update_node_health(self):
        """更新节点健康状态"""
        # 这里应该实现实际的健康检查
        # 暂时跳过
        pass
    
    async def inference(self, model_version: str, input_data: Dict[str, Any], 
                       request_id: str = None) -> InferenceResult:
        """执行推理"""
        start_time = time.time()
        
        try:
            # 生成请求ID
            if not request_id:
                request_id = f"req_{int(time.time() * 1000)}"
            
            # 检查缓存
            cache_key = f"{model_version}_{hash(str(input_data))}"
            cached_result = self.cache_manager.get(cache_key)
            if cached_result:
                processing_time = time.time() - start_time
                result = InferenceResult(
                    model_id=model_version,
                    prediction=cached_result['value'],
                    confidence=0.95,  # 缓存结果的置信度
                    processing_time=processing_time,
                    metadata={'cached': True}
                )
                
                # 记录指标
                await self.monitor.record_inference(
                    model_version, input_data, 
                    {'request_id': request_id, 'cached': True}, 
                    processing_time
                )
                
                return result
            
            # 限制并发数
            async with self.processing_semaphore:
                # 选择推理节点
                node_id = self.load_balancer.select_node()
                
                # 加载模型
                model = await self.model_loader.load_model(model_version)
                
                # 执行推理
                prediction = await self._execute_inference(model, input_data, model_version)
                
                processing_time = time.time() - start_time
                
                # 缓存结果
                self.cache_manager.set(cache_key, prediction)
                
                # 更新负载均衡器指标
                self.load_balancer.update_node_metrics(node_id, {
                    'response_time': processing_time
                })
                
                # 记录指标
                await self.monitor.record_inference(
                    model_version, input_data, 
                    {'request_id': request_id}, 
                    processing_time
                )
                
                # 计算置信度
                confidence = self._calculate_confidence(prediction, model_version)
                
                result = InferenceResult(
                    model_id=model_version,
                    prediction=prediction,
                    confidence=confidence,
                    processing_time=processing_time,
                    metadata={
                        'node_id': node_id,
                        'request_id': request_id,
                        'cached': False
                    }
                )
                
                return result
                
        except Exception as e:
            processing_time = time.time() - start_time
            
            # 记录错误
            self.monitor.log_error(e, {
                'model_version': model_version,
                'request_id': request_id,
                'processing_time': processing_time
            })
            
            # 返回错误结果
            return InferenceResult(
                model_id=model_version,
                prediction=None,
                confidence=0.0,
                processing_time=processing_time,
                metadata={
                    'error': str(e),
                    'request_id': request_id
                }
            )
    
    async def _execute_inference(self, model, input_data: Dict[str, Any], 
                               model_version: str) -> Any:
        """执行推理"""
        try:
            # 预处理输入数据
            processed_input = self._preprocess_input(input_data, model_version)
            
            # 执行模型推理
            if hasattr(model, 'predict'):
                prediction = model.predict(processed_input)
            elif hasattr(model, 'predict_proba'):
                prediction = model.predict_proba(processed_input)
            else:
                # 对于自定义模型，尝试直接调用
                prediction = model(processed_input)
            
            # 后处理输出
            processed_output = self._postprocess_output(prediction, model_version)
            
            return processed_output
            
        except Exception as e:
            logger.error(f"推理执行失败: {e}")
            raise
    
    def _preprocess_input(self, input_data: Dict[str, Any], model_version: str) -> np.ndarray:
        """预处理输入数据"""
        try:
            # 根据模型版本选择不同的预处理策略
            if 'status_recognition' in model_version:
                return self._preprocess_status_recognition(input_data)
            elif 'health_assessment' in model_version:
                return self._preprocess_health_assessment(input_data)
            elif 'vibration_analysis' in model_version:
                return self._preprocess_vibration_analysis(input_data)
            else:
                return self._preprocess_generic(input_data)
                
        except Exception as e:
            logger.error(f"输入预处理失败: {e}")
            raise
    
    def _preprocess_status_recognition(self, input_data: Dict[str, Any]) -> np.ndarray:
        """状态识别数据预处理"""
        # 提取数值特征
        features = []
        for key, value in input_data.items():
            if isinstance(value, (int, float)):
                features.append(float(value))
            elif isinstance(value, str):
                # 字符串特征编码
                features.append(hash(value) % 1000)
        
        return np.array(features).reshape(1, -1)
    
    def _preprocess_health_assessment(self, input_data: Dict[str, Any]) -> np.ndarray:
        """健康度评估数据预处理"""
        # 类似状态识别，但可能有不同的特征处理
        return self._preprocess_status_recognition(input_data)
    
    def _preprocess_vibration_analysis(self, input_data: Dict[str, Any]) -> np.ndarray:
        """振动分析数据预处理"""
        # 处理振动信号数据
        if 'signal' in input_data:
            signal = np.array(input_data['signal'])
            # 简单的特征提取
            features = [
                np.mean(signal),
                np.std(signal),
                np.max(signal),
                np.min(signal),
                len(signal)
            ]
            return np.array(features).reshape(1, -1)
        else:
            return self._preprocess_generic(input_data)
    
    def _preprocess_generic(self, input_data: Dict[str, Any]) -> np.ndarray:
        """通用数据预处理"""
        # 简单的数值提取
        features = []
        for value in input_data.values():
            if isinstance(value, (int, float)):
                features.append(float(value))
        
        return np.array(features).reshape(1, -1)
    
    def _postprocess_output(self, prediction: Any, model_version: str) -> Any:
        """后处理输出"""
        try:
            if isinstance(prediction, np.ndarray):
                if prediction.ndim == 1:
                    return prediction.tolist()
                elif prediction.ndim == 2:
                    return prediction.tolist()
                else:
                    return prediction.tolist()
            else:
                return prediction
                
        except Exception as e:
            logger.error(f"输出后处理失败: {e}")
            return prediction
    
    def _calculate_confidence(self, prediction: Any, model_version: str) -> float:
        """计算预测置信度"""
        try:
            if isinstance(prediction, (list, np.ndarray)):
                if len(prediction) > 0:
                    # 对于概率输出，取最大值作为置信度
                    if isinstance(prediction[0], (list, np.ndarray)):
                        # 二维数组，取每行最大值
                        max_probs = [max(p) for p in prediction]
                        return np.mean(max_probs)
                    else:
                        # 一维数组，直接取最大值
                        return max(prediction)
                else:
                    return 0.0
            else:
                return 0.8  # 默认置信度
                
        except Exception as e:
            logger.error(f"置信度计算失败: {e}")
            return 0.5
    
    async def batch_inference(self, model_version: str, 
                            input_batch: List[Dict[str, Any]]) -> List[InferenceResult]:
        """批量推理"""
        results = []
        
        # 创建任务
        tasks = [
            self.inference(model_version, input_data, f"batch_{i}")
            for i, input_data in enumerate(input_batch)
        ]
        
        # 并发执行
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, result in enumerate(batch_results):
            if isinstance(result, Exception):
                # 处理异常
                results.append(InferenceResult(
                    model_id=model_version,
                    prediction=None,
                    confidence=0.0,
                    processing_time=0.0,
                    metadata={'error': str(result)}
                ))
            else:
                results.append(result)
        
        return results
    
    def get_service_stats(self) -> Dict[str, Any]:
        """获取服务统计"""
        return {
            'loaded_models': self.model_loader.get_loaded_models(),
            'cache_stats': self.cache_manager.get_cache_stats(),
            'performance_stats': self.monitor.get_performance_stats(),
            'queue_size': self.request_queue.qsize(),
            'active_connections': self.processing_semaphore._value
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            # 检查各个组件状态
            cache_stats = self.cache_manager.get_cache_stats()
            performance_stats = self.monitor.get_performance_stats()
            
            health_status = {
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'components': {
                    'model_loader': 'healthy',
                    'cache_manager': 'healthy' if cache_stats['size'] < cache_stats['max_size'] else 'warning',
                    'monitor': 'healthy'
                },
                'metrics': {
                    'cache_hit_rate': cache_stats['hit_rate'],
                    'avg_processing_time': performance_stats.get('avg_processing_time', 0),
                    'error_rate': 0.01  # 模拟错误率
                }
            }
            
            return health_status
            
        except Exception as e:
            logger.error(f"健康检查失败: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            } 
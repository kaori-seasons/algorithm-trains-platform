import time
import threading
from typing import Dict, List, Optional
from prometheus_client import start_http_server, Gauge, Counter, Histogram
from dataclasses import dataclass
import logging
from datetime import datetime

@dataclass
class GPUMetric:
    """GPU指标数据"""
    node_name: str
    gpu_id: str
    utilization: float
    memory_used: float
    memory_total: float
    temperature: float
    power_usage: float

class GPUExporter:
    """GPU指标导出器"""
    
    def __init__(self, k8s_client, port: int = 9090):
        self.k8s_client = k8s_client
        self.port = port
        self.logger = logging.getLogger(__name__)
        
        # 定义Prometheus指标
        self.gpu_utilization = Gauge(
            'gpu_utilization_percent',
            'GPU utilization percentage',
            ['node', 'gpu_id', 'gpu_type']
        )
        
        self.gpu_memory_used = Gauge(
            'gpu_memory_used_bytes',
            'GPU memory used in bytes',
            ['node', 'gpu_id', 'gpu_type']
        )
        
        self.gpu_memory_total = Gauge(
            'gpu_memory_total_bytes',
            'GPU memory total in bytes',
            ['node', 'gpu_id', 'gpu_type']
        )
        
        self.gpu_temperature = Gauge(
            'gpu_temperature_celsius',
            'GPU temperature in celsius',
            ['node', 'gpu_id', 'gpu_type']
        )
        
        self.gpu_power_usage = Gauge(
            'gpu_power_usage_watts',
            'GPU power usage in watts',
            ['node', 'gpu_id', 'gpu_type']
        )
        
        # 集群级别指标
        self.cluster_gpu_total = Gauge(
            'cluster_gpu_total',
            'Total GPUs in cluster',
            ['cluster', 'gpu_type']
        )
        
        self.cluster_gpu_allocated = Gauge(
            'cluster_gpu_allocated',
            'Allocated GPUs in cluster',
            ['cluster', 'gpu_type']
        )
        
        # HPA相关指标
        self.hpa_scaling_events = Counter(
            'hpa_scaling_events_total',
            'Total HPA scaling events',
            ['namespace', 'hpa_name', 'direction']
        )
        
        # 调度器指标
        self.scheduler_queue_size = Gauge(
            'gpu_scheduler_queue_size',
            'GPU scheduler queue size',
            ['priority']
        )
        
        self.scheduling_latency = Histogram(
            'gpu_scheduling_latency_seconds',
            'GPU scheduling latency in seconds'
        )
        
        self.running = False
        self.collection_thread = None
    
    def start(self):
        """启动指标导出器"""
        if self.running:
            return
        
        self.running = True
        
        # 启动HTTP服务器
        start_http_server(self.port)
        self.logger.info(f"GPU Exporter started on port {self.port}")
        
        # 启动指标收集线程
        self.collection_thread = threading.Thread(target=self._collect_metrics_loop, daemon=True)
        self.collection_thread.start()
    
    def stop(self):
        """停止指标导出器"""
        self.running = False
        if self.collection_thread:
            self.collection_thread.join()
        self.logger.info("GPU Exporter stopped")
    
    def _collect_metrics_loop(self):
        """指标收集循环"""
        while self.running:
            try:
                self._collect_gpu_metrics()
                self._collect_cluster_metrics()
                time.sleep(30)  # 每30秒收集一次
            except Exception as e:
                self.logger.error(f"指标收集错误: {e}")
                time.sleep(60)
    
    def _collect_gpu_metrics(self):
        """收集GPU指标"""
        try:
            nodes = self.k8s_client.get_nodes()
            
            for node in nodes:
                if not node.get('gpu_info'):
                    continue
                
                node_name = node['name']
                
                # 模拟从DCGM获取GPU指标
                for gpu_type, gpu_info in node['gpu_info'].items():
                    for gpu_id in range(gpu_info.get('allocatable', 0)):
                        # 这里应该从实际的DCGM exporter获取数据
                        # 为了演示，我们使用模拟数据
                        gpu_metric = self._get_gpu_metric_from_dcgm(node_name, str(gpu_id), gpu_type)
                        
                        if gpu_metric:
                            self.gpu_utilization.labels(
                                node=node_name,
                                gpu_id=str(gpu_id),
                                gpu_type=gpu_type
                            ).set(gpu_metric.utilization)
                            
                            self.gpu_memory_used.labels(
                                node=node_name,
                                gpu_id=str(gpu_id),
                                gpu_type=gpu_type
                            ).set(gpu_metric.memory_used)
                            
                            self.gpu_memory_total.labels(
                                node=node_name,
                                gpu_id=str(gpu_id),
                                gpu_type=gpu_type
                            ).set(gpu_metric.memory_total)
                            
                            self.gpu_temperature.labels(
                                node=node_name,
                                gpu_id=str(gpu_id),
                                gpu_type=gpu_type
                            ).set(gpu_metric.temperature)
                            
                            self.gpu_power_usage.labels(
                                node=node_name,
                                gpu_id=str(gpu_id),
                                gpu_type=gpu_type
                            ).set(gpu_metric.power_usage)
        
        except Exception as e:
            self.logger.error(f"收集GPU指标失败: {e}")
    
    def _collect_cluster_metrics(self):
        """收集集群级别指标"""
        try:
            nodes = self.k8s_client.get_nodes()
            pods = self.k8s_client.get_pods()
            
            # 统计总GPU数量
            gpu_totals = {}
            gpu_allocated = {}
            
            for node in nodes:
                for gpu_type, gpu_info in node.get('gpu_info', {}).items():
                    gpu_totals[gpu_type] = gpu_totals.get(gpu_type, 0) + gpu_info.get('allocatable', 0)
            
            # 统计已分配GPU数量
            for pod in pods:
                for resource_name, count in pod.get('gpu_resources', {}).items():
                    gpu_type = self._get_gpu_type_by_resource(resource_name)
                    gpu_allocated[gpu_type] = gpu_allocated.get(gpu_type, 0) + int(count)
            
            # 更新指标
            for gpu_type, total in gpu_totals.items():
                self.cluster_gpu_total.labels(
                    cluster='default',
                    gpu_type=gpu_type
                ).set(total)
                
                allocated = gpu_allocated.get(gpu_type, 0)
                self.cluster_gpu_allocated.labels(
                    cluster='default',
                    gpu_type=gpu_type
                ).set(allocated)
        
        except Exception as e:
            self.logger.error(f"收集集群指标失败: {e}")
    
    def _get_gpu_metric_from_dcgm(self, node_name: str, gpu_id: str, gpu_type: str) -> Optional[GPUMetric]:
        """从DCGM获取GPU指标（模拟实现）"""
        # 在实际实现中，这里应该调用DCGM API或解析DCGM exporter的数据
        # 这里使用模拟数据
        import random
        
        return GPUMetric(
            node_name=node_name,
            gpu_id=gpu_id,
            utilization=random.uniform(0, 100),
            memory_used=random.uniform(0, 32) * 1024 * 1024 * 1024,  # 转换为字节
            memory_total=32 * 1024 * 1024 * 1024,  # 32GB
            temperature=random.uniform(30, 80),
            power_usage=random.uniform(100, 300)
        )
    
    def _get_gpu_type_by_resource(self, resource_name: str) -> str:
        """根据资源名称获取GPU类型"""
        mapping = {
            'nvidia.com/gpu': 'nvidia',
            'amd.com/gpu': 'amd',
            'intel.com/gpu': 'intel'
        }
        return mapping.get(resource_name, 'unknown')
    
    def record_hpa_scaling_event(self, namespace: str, hpa_name: str, direction: str):
        """记录HPA伸缩事件"""
        self.hpa_scaling_events.labels(
            namespace=namespace,
            hpa_name=hpa_name,
            direction=direction
        ).inc()
    
    def record_scheduling_latency(self, latency_seconds: float):
        """记录调度延迟"""
        self.scheduling_latency.observe(latency_seconds)
    
    def update_scheduler_queue_size(self, priority: str, size: int):
        """更新调度器队列大小"""
        self.scheduler_queue_size.labels(priority=priority).set(size) 
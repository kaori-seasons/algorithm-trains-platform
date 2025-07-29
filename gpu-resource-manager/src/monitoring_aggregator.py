import time
import threading
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

@dataclass
class AggregatedMetrics:
    """聚合指标数据"""
    timestamp: datetime
    cluster_name: str
    total_gpus: int
    allocated_gpus: int
    avg_gpu_utilization: float
    avg_gpu_memory_usage: float
    avg_gpu_temperature: float
    total_power_usage: float
    active_pods: int
    pending_pods: int

class MonitoringAggregator:
    """监控数据聚合器"""
    
    def __init__(self, prometheus_client, k8s_client):
        self.prometheus_client = prometheus_client
        self.k8s_client = k8s_client
        self.logger = logging.getLogger(__name__)
        
        # 聚合配置
        self.aggregation_interval = 60  # 1分钟聚合一次
        self.retention_days = 30  # 保留30天数据
        
        # 数据存储
        self.aggregated_data: List[AggregatedMetrics] = []
        self.running = False
        self.aggregation_thread = None
    
    def start_aggregation(self):
        """启动数据聚合"""
        if self.running:
            return
        
        self.running = True
        self.aggregation_thread = threading.Thread(target=self._aggregation_loop, daemon=True)
        self.aggregation_thread.start()
        self.logger.info("监控数据聚合器已启动")
    
    def stop_aggregation(self):
        """停止数据聚合"""
        self.running = False
        if self.aggregation_thread:
            self.aggregation_thread.join()
        self.logger.info("监控数据聚合器已停止")
    
    def _aggregation_loop(self):
        """聚合循环"""
        while self.running:
            try:
                # 聚合当前数据
                aggregated = self._aggregate_current_metrics()
                if aggregated:
                    self.aggregated_data.append(aggregated)
                
                # 清理过期数据
                self._cleanup_old_data()
                
                time.sleep(self.aggregation_interval)
            except Exception as e:
                self.logger.error(f"数据聚合错误: {e}")
                time.sleep(60)
    
    def _aggregate_current_metrics(self) -> Optional[AggregatedMetrics]:
        """聚合当前指标"""
        try:
            # 获取GPU利用率数据
            gpu_util_metrics = self.prometheus_client.query_metric("gpu_utilization_percent")
            gpu_memory_metrics = self.prometheus_client.query_metric("gpu_memory_used_bytes / gpu_memory_total_bytes * 100")
            gpu_temp_metrics = self.prometheus_client.query_metric("gpu_temperature_celsius")
            gpu_power_metrics = self.prometheus_client.query_metric("gpu_power_usage_watts")
            
            # 获取集群GPU总数和分配数
            total_gpu_metrics = self.prometheus_client.query_metric("cluster_gpu_total")
            allocated_gpu_metrics = self.prometheus_client.query_metric("cluster_gpu_allocated")
            
            # 获取Pod数据
            pods = self.k8s_client.get_pods()
            active_pods = len([p for p in pods if p['status'] == 'Running'])
            pending_pods = len([p for p in pods if p['status'] == 'Pending'])
            
            # 计算聚合值
            total_gpus = sum(metric.value for metric in total_gpu_metrics)
            allocated_gpus = sum(metric.value for metric in allocated_gpu_metrics)
            
            avg_utilization = sum(metric.value for metric in gpu_util_metrics) / max(len(gpu_util_metrics), 1)
            avg_memory_usage = sum(metric.value for metric in gpu_memory_metrics) / max(len(gpu_memory_metrics), 1)
            avg_temperature = sum(metric.value for metric in gpu_temp_metrics) / max(len(gpu_temp_metrics), 1)
            total_power = sum(metric.value for metric in gpu_power_metrics)
            
            return AggregatedMetrics(
                timestamp=datetime.now(),
                cluster_name="default",
                total_gpus=int(total_gpus),
                allocated_gpus=int(allocated_gpus),
                avg_gpu_utilization=round(avg_utilization, 2),
                avg_gpu_memory_usage=round(avg_memory_usage, 2),
                avg_gpu_temperature=round(avg_temperature, 2),
                total_power_usage=round(total_power, 2),
                active_pods=active_pods,
                pending_pods=pending_pods
            )
        
        except Exception as e:
            self.logger.error(f"聚合当前指标失败: {e}")
            return None
    
    def _cleanup_old_data(self):
        """清理过期数据"""
        cutoff_time = datetime.now() - timedelta(days=self.retention_days)
        self.aggregated_data = [
            data for data in self.aggregated_data 
            if data.timestamp > cutoff_time
        ]
    
    def get_historical_data(self, hours: int = 24) -> List[AggregatedMetrics]:
        """获取历史数据"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            data for data in self.aggregated_data 
            if data.timestamp > cutoff_time
        ]
    
    def get_resource_trends(self) -> Dict[str, List[float]]:
        """获取资源趋势数据"""
        recent_data = self.get_historical_data(24)  # 最近24小时
        
        return {
            'timestamps': [data.timestamp.isoformat() for data in recent_data],
            'gpu_utilization': [data.avg_gpu_utilization for data in recent_data],
            'gpu_memory_usage': [data.avg_gpu_memory_usage for data in recent_data],
            'gpu_allocation_rate': [
                data.allocated_gpus / max(data.total_gpus, 1) * 100 
                for data in recent_data
            ],
            'active_pods': [data.active_pods for data in recent_data],
            'pending_pods': [data.pending_pods for data in recent_data]
        }
    
    def get_cluster_summary(self) -> Dict[str, Any]:
        """获取集群摘要信息"""
        try:
            # 获取最新的聚合数据
            if not self.aggregated_data:
                return {}
            
            latest_data = self.aggregated_data[-1]
            
            # 计算24小时趋势
            day_ago_data = None
            for data in reversed(self.aggregated_data):
                if data.timestamp < datetime.now() - timedelta(hours=24):
                    day_ago_data = data
                    break
            
            # 计算变化率
            utilization_change = 0
            allocation_change = 0
            
            if day_ago_data:
                utilization_change = latest_data.avg_gpu_utilization - day_ago_data.avg_gpu_utilization
                allocation_change = (latest_data.allocated_gpus / max(latest_data.total_gpus, 1) * 100) - \
                                 (day_ago_data.allocated_gpus / max(day_ago_data.total_gpus, 1) * 100)
            
            return {
                'current': {
                    'total_gpus': latest_data.total_gpus,
                    'allocated_gpus': latest_data.allocated_gpus,
                    'allocation_rate': round(latest_data.allocated_gpus / max(latest_data.total_gpus, 1) * 100, 2),
                    'avg_utilization': latest_data.avg_gpu_utilization,
                    'avg_memory_usage': latest_data.avg_gpu_memory_usage,
                    'avg_temperature': latest_data.avg_gpu_temperature,
                    'total_power': latest_data.total_power_usage,
                    'active_pods': latest_data.active_pods,
                    'pending_pods': latest_data.pending_pods
                },
                'trends': {
                    'utilization_change': round(utilization_change, 2),
                    'allocation_change': round(allocation_change, 2)
                },
                'timestamp': latest_data.timestamp.isoformat()
            }
        
        except Exception as e:
            self.logger.error(f"获取集群摘要失败: {e}")
            return {}
    
    def get_node_performance_ranking(self) -> List[Dict[str, Any]]:
        """获取节点性能排名"""
        try:
            nodes = self.k8s_client.get_nodes()
            node_performance = []
            
            for node in nodes:
                if not node.get('gpu_info'):
                    continue
                
                # 计算节点GPU利用率
                gpu_count = sum(info.get('allocatable', 0) for info in node['gpu_info'].values())
                if gpu_count == 0:
                    continue
                
                # 模拟获取节点GPU指标
                node_utilization = self._get_node_gpu_utilization(node['name'])
                node_memory_usage = self._get_node_gpu_memory_usage(node['name'])
                
                performance_score = (node_utilization * 0.6 + node_memory_usage * 0.4)
                
                node_performance.append({
                    'node_name': node['name'],
                    'gpu_count': gpu_count,
                    'utilization': round(node_utilization, 2),
                    'memory_usage': round(node_memory_usage, 2),
                    'performance_score': round(performance_score, 2)
                })
            
            # 按性能分数排序
            node_performance.sort(key=lambda x: x['performance_score'], reverse=True)
            
            return node_performance
        
        except Exception as e:
            self.logger.error(f"获取节点性能排名失败: {e}")
            return []
    
    def _get_node_gpu_utilization(self, node_name: str) -> float:
        """获取节点GPU利用率（模拟实现）"""
        import random
        return random.uniform(0, 100)
    
    def _get_node_gpu_memory_usage(self, node_name: str) -> float:
        """获取节点GPU显存使用率（模拟实现）"""
        import random
        return random.uniform(0, 100) 
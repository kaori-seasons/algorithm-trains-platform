import json
import re
import logging
from typing import Dict, List, Optional, Any
from kubernetes import client
from kubernetes.client.rest import ApiException
from dataclasses import dataclass
from enum import Enum

class MetricType(Enum):
    """指标类型枚举"""
    RESOURCE = "Resource"
    PODS = "Pods"
    OBJECT = "Object"
    EXTERNAL = "External"

@dataclass
class HPAMetric:
    """HPA指标配置"""
    metric_type: MetricType
    name: str
    target_value: float
    target_type: str = "Utilization"  # Utilization, AverageValue, Value

class GPUHPAController:
    """GPU感知的HPA控制器"""
    
    def __init__(self, k8s_client):
        self.k8s_client = k8s_client
        self.autoscaling_v2 = client.AutoscalingV2Api()
        self.logger = logging.getLogger(__name__)
        
        # GPU指标配置
        self.gpu_metrics_mapping = {
            'nvidia.com/gpu': 'container_gpu_usage',
            'amd.com/gpu': 'container_amd_gpu_usage',
            'intel.com/gpu': 'container_intel_gpu_usage'
        }
    
    def create_hpa(self, namespace: str, name: str, min_replicas: int, 
                   max_replicas: int, metrics: List[str], 
                   target_ref: Dict[str, str] = None) -> bool:
        """创建HPA"""
        try:
            # 删除已存在的HPA
            self.delete_hpa(namespace, name)
            
            # 解析指标配置
            parsed_metrics = self._parse_metrics(metrics)
            
            # 构建HPA规格
            hpa_spec = self._build_hpa_spec(
                name, namespace, min_replicas, max_replicas, 
                parsed_metrics, target_ref
            )
            
            # 创建HPA
            self.autoscaling_v2.create_namespaced_horizontal_pod_autoscaler(
                namespace=namespace, body=hpa_spec
            )
            
            self.logger.info(f"成功创建HPA: {namespace}/{name}")
            return True
            
        except Exception as e:
            self.logger.error(f"创建HPA失败: {e}")
            return False
    
    def delete_hpa(self, namespace: str, name: str) -> bool:
        """删除HPA"""
        try:
            # 尝试删除v2版本
            try:
                self.autoscaling_v2.delete_namespaced_horizontal_pod_autoscaler(
                    name=name, namespace=namespace, grace_period_seconds=0
                )
            except ApiException as e:
                if e.status != 404:
                    self.logger.warning(f"删除v2 HPA失败: {e}")
            
            # 尝试删除v1版本
            try:
                client.AutoscalingV1Api().delete_namespaced_horizontal_pod_autoscaler(
                    name=name, namespace=namespace, grace_period_seconds=0
                )
            except ApiException as e:
                if e.status != 404:
                    self.logger.warning(f"删除v1 HPA失败: {e}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"删除HPA失败: {e}")
            return False
    
    def _parse_metrics(self, metrics: List[str]) -> List[HPAMetric]:
        """解析指标配置"""
        parsed_metrics = []
        
        for metric in metrics:
            metric = metric.strip()
            if not metric:
                continue
            
            # 解析指标格式: cpu:50%, mem:80%, gpu:70%
            if ':' in metric or '=' in metric:
                parts = re.split(':|=', metric)
                if len(parts) >= 2:
                    metric_name = parts[0].strip().lower()
                    target_value = float(parts[1].replace('%', '').strip())
                    
                    if metric_name == 'cpu':
                        parsed_metrics.append(HPAMetric(
                            metric_type=MetricType.RESOURCE,
                            name='cpu',
                            target_value=target_value,
                            target_type='Utilization'
                        ))
                    elif metric_name in ['mem', 'memory']:
                        parsed_metrics.append(HPAMetric(
                            metric_type=MetricType.RESOURCE,
                            name='memory',
                            target_value=target_value,
                            target_type='Utilization'
                        ))
                    elif metric_name == 'gpu':
                        parsed_metrics.append(HPAMetric(
                            metric_type=MetricType.PODS,
                            name='container_gpu_usage',
                            target_value=target_value / 100,
                            target_type='AverageValue'
                        ))
        
        return parsed_metrics
    
    def _build_hpa_spec(self, name: str, namespace: str, min_replicas: int,
                       max_replicas: int, metrics: List[HPAMetric],
                       target_ref: Dict[str, str] = None) -> Dict[str, Any]:
        """构建HPA规格"""
        # 默认目标引用
        if not target_ref:
            target_ref = {
                'apiVersion': 'apps/v1',
                'kind': 'Deployment',
                'name': name
            }
        
        hpa_spec = {
            'apiVersion': 'autoscaling/v2',
            'kind': 'HorizontalPodAutoscaler',
            'metadata': {
                'name': name,
                'namespace': namespace,
                'labels': {
                    'app': name,
                    'managed-by': 'gpu-resource-manager'
                }
            },
            'spec': {
                'scaleTargetRef': target_ref,
                'minReplicas': min_replicas,
                'maxReplicas': max_replicas,
                'metrics': []
            }
        }
        
        # 添加指标配置
        for metric in metrics:
            metric_config = self._build_metric_config(metric)
            if metric_config:
                hpa_spec['spec']['metrics'].append(metric_config)
        
        return hpa_spec
    
    def _build_metric_config(self, metric: HPAMetric) -> Optional[Dict[str, Any]]:
        """构建单个指标配置"""
        if metric.metric_type == MetricType.RESOURCE:
            return {
                'type': 'Resource',
                'resource': {
                    'name': metric.name,
                    'target': {
                        'type': metric.target_type,
                        'averageUtilization': int(metric.target_value)
                    }
                }
            }
        elif metric.metric_type == MetricType.PODS:
            return {
                'type': 'Pods',
                'pods': {
                    'metric': {
                        'name': metric.name
                    },
                    'target': {
                        'type': 'AverageValue',
                        'averageValue': str(metric.target_value)
                    }
                }
            }
        
        return None
    
    def get_hpa_status(self, namespace: str, name: str) -> Optional[Dict[str, Any]]:
        """获取HPA状态"""
        try:
            hpa = self.autoscaling_v2.read_namespaced_horizontal_pod_autoscaler(
                name=name, namespace=namespace
            )
            
            return {
                'name': hpa.metadata.name,
                'namespace': hpa.metadata.namespace,
                'min_replicas': hpa.spec.min_replicas,
                'max_replicas': hpa.spec.max_replicas,
                'current_replicas': hpa.status.current_replicas,
                'desired_replicas': hpa.status.desired_replicas,
                'last_scale_time': hpa.status.last_scale_time,
                'conditions': [
                    {
                        'type': condition.type,
                        'status': condition.status,
                        'reason': condition.reason,
                        'message': condition.message
                    } for condition in (hpa.status.conditions or [])
                ]
            }
        except Exception as e:
            self.logger.error(f"获取HPA状态失败: {e}")
            return None 
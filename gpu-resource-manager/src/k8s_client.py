import os
import yaml
import json
from typing import Dict, List, Optional, Tuple
from kubernetes import client, config
from kubernetes.client.rest import ApiException
import logging

class K8sResourceManager:
    """Kubernetes资源管理器"""
    
    def __init__(self, kubeconfig_path: Optional[str] = None):
        """初始化K8s客户端"""
        try:
            if kubeconfig_path and os.path.exists(kubeconfig_path):
                config.load_kube_config(config_file=kubeconfig_path)
            else:
                config.load_incluster_config()
        except Exception:
            # 如果都失败，尝试默认配置
            config.load_kube_config()
        
        self.v1 = client.CoreV1Api()
        self.apps_v1 = client.AppsV1Api()
        self.custom_objects = client.CustomObjectsApi()
        self.autoscaling_v2 = client.AutoscalingV2Api()
        
        # GPU资源配置
        self.gpu_resource_mapping = {
            "gpu": "nvidia.com/gpu",
            "nvidia": "nvidia.com/gpu",
            "amd": "amd.com/gpu",
            "intel": "intel.com/gpu"
        }
        
        # 初始化GPU解析器
        from .gpu_parser import GPUResourceParser
        self.gpu_parser = GPUResourceParser({
            'GPU_RESOURCE': self.gpu_resource_mapping,
            'DEFAULT_GPU_RESOURCE_NAME': 'nvidia.com/gpu'
        })
    
    def create_pod_with_gpu(self, namespace: str, pod_spec: Dict, gpu_resource: str) -> bool:
        """创建带GPU资源的Pod"""
        try:
            # 解析GPU资源
            gpu_info = self.gpu_parser.parse_gpu_resource(gpu_resource)
            
            # 修改Pod规格添加GPU资源
            self._add_gpu_resources_to_pod(pod_spec, gpu_info)
            
            # 创建Pod
            pod = client.V1Pod(**pod_spec)
            self.v1.create_namespaced_pod(namespace=namespace, body=pod)
            return True
        except Exception as e:
            logging.error(f"创建GPU Pod失败: {e}")
            return False
    
    def _add_gpu_resources_to_pod(self, pod_spec: Dict, gpu_info) -> None:
        """为Pod添加GPU资源配置"""
        if not pod_spec.get('spec', {}).get('containers'):
            return
        
        for container in pod_spec['spec']['containers']:
            if not container.get('resources'):
                container['resources'] = {'requests': {}, 'limits': {}}
            
            # 添加GPU资源请求和限制
            if gpu_info.gpu_num > 0:
                gpu_count = str(int(gpu_info.gpu_num)) if gpu_info.gpu_num >= 1 else str(gpu_info.gpu_num)
                container['resources']['requests'][gpu_info.resource_name] = gpu_count
                container['resources']['limits'][gpu_info.resource_name] = gpu_count
            
            # 设置GPU类型节点选择器
            if gpu_info.gpu_type:
                if not pod_spec['spec'].get('nodeSelector'):
                    pod_spec['spec']['nodeSelector'] = {}
                pod_spec['spec']['nodeSelector']['gpu-type'] = gpu_info.gpu_type
    
    def get_pods(self, namespace: str = None, labels: Dict[str, str] = None) -> List[Dict]:
        """获取Pod列表"""
        try:
            if namespace:
                if labels:
                    label_selector = ','.join([f"{k}={v}" for k, v in labels.items()])
                    pods = self.v1.list_namespaced_pod(namespace=namespace, label_selector=label_selector)
                else:
                    pods = self.v1.list_namespaced_pod(namespace=namespace)
            else:
                pods = self.v1.list_pod_for_all_namespaces()
            
            return [self._pod_to_dict(pod) for pod in pods.items]
        except Exception as e:
            logging.error(f"获取Pod列表失败: {e}")
            return []
    
    def _pod_to_dict(self, pod) -> Dict:
        """将Pod对象转换为字典格式"""
        
        # 基于cube-studio的pod_model2dict实现
        metadata = pod.metadata
        status = pod.status.phase if pod and hasattr(pod, 'status') and hasattr(pod.status, 'phase') else ''
        
        # 处理运行状态
        if status.lower() == 'running':
            status = 'Running' if [x.status for x in pod.status.conditions if
                                   x.type == 'Ready' and x.status == 'True'] else 'CrashLoopBackOff'
        
        containers = pod.spec.containers
        memory = [self._to_memory_gb(container.resources.requests.get('memory', '0G')) for container in containers if
                  container.resources and container.resources.requests]
        cpu = [self._to_cpu(container.resources.requests.get('cpu', '0')) for container in containers if
               container.resources and container.resources.requests]
        
        # 获取GPU资源占用
        gpu_resources = {}
        for container in containers:
            if container.resources and container.resources.requests:
                for resource_name in self.gpu_resource_mapping.values():
                    gpu_count = container.resources.requests.get(resource_name, '0')
                    if gpu_count != '0':
                        gpu_resources[resource_name] = gpu_resources.get(resource_name, 0) + float(gpu_count)
        
        return {
            'name': metadata.name,
            'namespace': metadata.namespace,
            'host_ip': pod.status.host_ip if pod.status.host_ip else '',
            'pod_ip': pod.status.pod_ip,
            'status': status,
            'node_name': pod.spec.node_name,
            'labels': metadata.labels if metadata.labels else {},
            'annotations': metadata.annotations if metadata.annotations else {},
            'memory': sum(memory),
            'cpu': sum(cpu),
            'gpu_resources': gpu_resources,
            'start_time': metadata.creation_timestamp,
        }
    
    def _to_memory_gb(self, memory_str: str) -> float:
        """转换内存字符串为GB"""
        if not memory_str or memory_str == '0':
            return 0.0
        
        memory_str = memory_str.upper()
        if 'GI' in memory_str:
            return float(memory_str.replace('GI', '')) * 1.073741824
        elif 'G' in memory_str:
            return float(memory_str.replace('G', ''))
        elif 'MI' in memory_str:
            return float(memory_str.replace('MI', '')) * 1.073741824 / 1024
        elif 'M' in memory_str:
            return float(memory_str.replace('M', '')) / 1024
        else:
            return float(memory_str) / (1024**3)
    
    def _to_cpu(self, cpu_str: str) -> float:
        """转换CPU字符串为核数"""
        if not cpu_str or cpu_str == '0':
            return 0.0
        
        if 'm' in cpu_str:
            return float(cpu_str.replace('m', '')) / 1000
        else:
            return float(cpu_str)
    
    def get_nodes(self) -> List[Dict]:
        """获取节点列表"""
        try:
            nodes = self.v1.list_node()
            return [self._node_to_dict(node) for node in nodes.items]
        except Exception as e:
            logging.error(f"获取节点列表失败: {e}")
            return []
    
    def _node_to_dict(self, node) -> Dict:
        """将节点对象转换为字典格式"""
        allocatable = node.status.allocatable
        capacity = node.status.capacity
        
        # 获取GPU信息
        gpu_info = {}
        for gpu_type, resource_name in self.gpu_resource_mapping.items():
            if resource_name in allocatable:
                gpu_info[gpu_type] = {
                    'allocatable': int(allocatable[resource_name]),
                    'capacity': int(capacity.get(resource_name, 0))
                }
        
        return {
            'name': node.metadata.name,
            'labels': node.metadata.labels or {},
            'annotations': node.metadata.annotations or {},
            'cpu_allocatable': self._to_cpu(allocatable.get('cpu', '0')),
            'memory_allocatable': self._to_memory_gb(allocatable.get('memory', '0')),
            'cpu_capacity': self._to_cpu(capacity.get('cpu', '0')),
            'memory_capacity': self._to_memory_gb(capacity.get('memory', '0')),
            'gpu_info': gpu_info,
            'ready': self._is_node_ready(node),
            'schedulable': not node.spec.unschedulable if node.spec.unschedulable is not None else True
        }
    
    def _is_node_ready(self, node) -> bool:
        """检查节点是否就绪"""
        if not node.status.conditions:
            return False
        
        for condition in node.status.conditions:
            if condition.type == 'Ready':
                return condition.status == 'True'
        return False
    
    def create_deployment(self, namespace: str, deployment_spec: Dict) -> bool:
        """创建Deployment"""
        try:
            deployment = client.V1Deployment(**deployment_spec)
            self.apps_v1.create_namespaced_deployment(namespace=namespace, body=deployment)
            return True
        except Exception as e:
            logging.error(f"创建Deployment失败: {e}")
            return False
    
    def delete_pod(self, namespace: str, pod_name: str) -> bool:
        """删除Pod"""
        try:
            self.v1.delete_namespaced_pod(name=pod_name, namespace=namespace, grace_period_seconds=0)
            return True
        except Exception as e:
            logging.error(f"删除Pod失败: {e}")
            return False
    
    def label_node(self, node_names: List[str], labels: Dict[str, str]) -> bool:
        """为节点添加标签"""
        try:
            for node_name in node_names:
                body = {"metadata": {"labels": labels}}
                self.v1.patch_node(name=node_name, body=body)
            return True
        except Exception as e:
            logging.error(f"节点标签更新失败: {e}")
            return False 
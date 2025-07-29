import logging
import threading
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from kubernetes import client
from kubernetes.client.rest import ApiException

class SchedulingPolicy(Enum):
    """调度策略枚举"""
    SPREAD = "spread"  # 分散调度
    PACK = "pack"      # 紧凑调度
    BALANCED = "balanced"  # 均衡调度

@dataclass
class SchedulingRequest:
    """调度请求"""
    pod_name: str
    namespace: str
    gpu_requirement: str
    memory_requirement: float
    priority: int = 0
    node_selector: Dict[str, str] = None
    
class GPUScheduler:
    """GPU感知的资源调度器"""
    
    def __init__(self, k8s_client, gpu_parser, memory_guard):
        self.k8s_client = k8s_client
        self.gpu_parser = gpu_parser
        self.memory_guard = memory_guard
        self.logger = logging.getLogger(__name__)
        
        # 调度配置
        self.scheduling_policy = SchedulingPolicy.BALANCED
        self.gpu_fragmentation_threshold = 0.3  # GPU碎片化阈值
        
        # 调度队列
        self.pending_requests: List[SchedulingRequest] = []
        self.scheduling_lock = threading.Lock()
        
    def schedule_pod(self, request: SchedulingRequest) -> Optional[str]:
        """调度Pod到合适的节点"""
        try:
            # 解析GPU需求
            gpu_info = self.gpu_parser.parse_gpu_resource(request.gpu_requirement)
            
            # 验证显存需求
            if not self.memory_guard.validate_memory_requirement(
                request.gpu_requirement, 
                request.memory_requirement
            ):
                self.logger.warning(f"Pod {request.pod_name} GPU显存需求不满足")
                return None
            
            # 查找合适的节点
            suitable_nodes = self._find_suitable_nodes(request, gpu_info)
            
            if not suitable_nodes:
                self.logger.warning(f"未找到满足需求的节点: {request.pod_name}")
                return None
            
            # 根据调度策略选择最佳节点
            selected_node = self._select_best_node(suitable_nodes, gpu_info)
            
            self.logger.info(f"为Pod {request.pod_name} 选择节点: {selected_node}")
            return selected_node
            
        except Exception as e:
            self.logger.error(f"调度Pod失败: {e}")
            return None
    
    def _find_suitable_nodes(self, request: SchedulingRequest, gpu_info) -> List[Dict]:
        """查找满足需求的节点"""
        suitable_nodes = []
        nodes = self.k8s_client.get_nodes()
        
        for node in nodes:
            if not node['ready'] or not node['schedulable']:
                continue
            
            # 检查节点选择器
            if request.node_selector:
                if not all(
                    node['labels'].get(k) == v 
                    for k, v in request.node_selector.items()
                ):
                    continue
            
            # 检查GPU类型匹配
            if gpu_info.gpu_type:
                node_gpu_type = node['labels'].get('gpu-type')
                if node_gpu_type != gpu_info.gpu_type:
                    continue
            
            # 检查GPU资源可用性
            if gpu_info.gpu_num > 0:
                available_gpus = self._get_available_gpus(node)
                if available_gpus < gpu_info.gpu_num:
                    continue
            
            # 计算节点评分
            score = self._calculate_node_score(node, gpu_info)
            suitable_nodes.append({
                'node': node,
                'score': score,
                'available_gpus': self._get_available_gpus(node)
            })
        
        return suitable_nodes
    
    def _get_available_gpus(self, node: Dict) -> float:
        """获取节点可用GPU数量"""
        total_gpus = 0
        used_gpus = 0
        
        # 统计总GPU数量
        for gpu_type, gpu_info in node.get('gpu_info', {}).items():
            total_gpus += gpu_info.get('allocatable', 0)
        
        # 统计已使用GPU数量
        pods = self.k8s_client.get_pods()
        for pod in pods:
            if pod.get('node_name') == node['name']:
                for resource_name, count in pod.get('gpu_resources', {}).items():
                    used_gpus += float(count)
        
        return total_gpus - used_gpus
    
    def _calculate_node_score(self, node: Dict, gpu_info) -> float:
        """计算节点评分"""
        score = 0.0
        
        # GPU利用率评分
        gpu_utilization = self._get_gpu_utilization(node)
        if self.scheduling_policy == SchedulingPolicy.SPREAD:
            score += (1.0 - gpu_utilization) * 40  # 偏好低利用率节点
        elif self.scheduling_policy == SchedulingPolicy.PACK:
            score += gpu_utilization * 40  # 偏好高利用率节点
        else:  # BALANCED
            score += (0.5 - abs(gpu_utilization - 0.5)) * 40
        
        # 内存利用率评分
        memory_utilization = node.get('used_memory', 0) / node.get('memory_allocatable', 1)
        score += (1.0 - memory_utilization) * 30
        
        # CPU利用率评分
        cpu_utilization = node.get('used_cpu', 0) / node.get('cpu_allocatable', 1)
        score += (1.0 - cpu_utilization) * 20
        
        # GPU碎片化惩罚
        fragmentation_penalty = self._calculate_fragmentation_penalty(node, gpu_info)
        score -= fragmentation_penalty * 10
        
        return score
    
    def _get_gpu_utilization(self, node: Dict) -> float:
        """获取节点GPU利用率"""
        total_gpus = 0
        used_gpus = 0
        
        for gpu_type, gpu_info in node.get('gpu_info', {}).items():
            total_gpus += gpu_info.get('allocatable', 0)
            used_gpus += gpu_info.get('allocatable', 0) - gpu_info.get('available', 0)
        
        return used_gpus / max(total_gpus, 1)
    
    def _calculate_fragmentation_penalty(self, node: Dict, gpu_info) -> float:
        """计算GPU碎片化惩罚"""
        if gpu_info.gpu_num < 1:
            return 0.0  # 虚拟GPU不考虑碎片化
        
        available_gpus = self._get_available_gpus(node)
        required_gpus = gpu_info.gpu_num
        
        # 如果剩余GPU数量小于阈值，增加惩罚
        remaining_after_allocation = available_gpus - required_gpus
        if 0 < remaining_after_allocation < self.gpu_fragmentation_threshold:
            return 1.0
        
        return 0.0
    
    def _select_best_node(self, suitable_nodes: List[Dict], gpu_info) -> str:
        """选择最佳节点"""
        if not suitable_nodes:
            return None
        
        # 按评分排序
        suitable_nodes.sort(key=lambda x: x['score'], reverse=True)
        
        # 返回评分最高的节点
        return suitable_nodes[0]['node']['name']
    
    def add_scheduling_request(self, request: SchedulingRequest):
        """添加调度请求到队列"""
        with self.scheduling_lock:
            self.pending_requests.append(request)
            # 按优先级排序
            self.pending_requests.sort(key=lambda x: x.priority, reverse=True)
    
    def process_pending_requests(self):
        """处理待调度请求"""
        with self.scheduling_lock:
            processed_requests = []
            
            for request in self.pending_requests[:]:
                selected_node = self.schedule_pod(request)
                if selected_node:
                    self.logger.info(f"成功调度Pod {request.pod_name} 到节点 {selected_node}")
                    processed_requests.append(request)
                else:
                    self.logger.warning(f"暂时无法调度Pod {request.pod_name}")
            
            # 移除已处理的请求
            for request in processed_requests:
                self.pending_requests.remove(request) 
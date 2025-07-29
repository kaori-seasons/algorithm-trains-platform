from typing import Dict, List, Optional
from dataclasses import dataclass
import logging

@dataclass
class GPUMemoryRequirement:
    """GPU显存需求"""
    min_memory_gb: float
    preferred_memory_gb: Optional[float] = None
    gpu_type: Optional[str] = None

class GPUMemoryGuard:
    """GPU显存保障器"""
    
    def __init__(self, k8s_client, gpu_parser):
        self.k8s_client = k8s_client
        self.gpu_parser = gpu_parser
        
        # 预定义的GPU显存配置
        self.gpu_memory_specs = {
            'T4': 16.0,
            'V100': 32.0,
            'A100': 80.0,
            'H100': 80.0,
            'RTX3090': 24.0,
            'RTX4090': 24.0,
            'A6000': 48.0,
            'A40': 48.0
        }
    
    def validate_memory_requirement(self, gpu_resource: str, memory_requirement: float) -> bool:
        """验证GPU资源是否满足显存需求"""
        gpu_info = self.gpu_parser.parse_gpu_resource(gpu_resource)
        
        # 如果直接指定了显存大小
        if gpu_info.memory_gb:
            return gpu_info.memory_gb >= memory_requirement
        
        # 根据GPU型号和数量计算总显存
        if gpu_info.gpu_type and gpu_info.gpu_type in self.gpu_memory_specs:
            total_memory = self.gpu_memory_specs[gpu_info.gpu_type] * gpu_info.gpu_num
            return total_memory >= memory_requirement
        
        # 如果无法确定，记录警告但允许通过
        logging.warning(f"无法验证GPU资源 {gpu_resource} 的显存需求")
        return True
    
    def find_suitable_nodes(self, memory_requirement: float) -> List[Dict]:
        """查找满足显存需求的节点"""
        suitable_nodes = []
        nodes = self.k8s_client.get_nodes()
        
        for node in nodes:
            if not node['ready'] or not node['schedulable']:
                continue
            
            # 检查节点GPU信息
            for gpu_type, gpu_info in node.get('gpu_info', {}).items():
                if gpu_info['allocatable'] > 0:
                    gpu_memory = self.gpu_memory_specs.get(gpu_type, 0)
                    
                    # 检查是否满足显存需求
                    if gpu_memory >= memory_requirement:
                        suitable_nodes.append({
                            'node_name': node['name'],
                            'gpu_type': gpu_type,
                            'available_gpus': gpu_info['allocatable'],
                            'memory_per_gpu': gpu_memory,
                            'total_available_memory': gpu_memory * gpu_info['allocatable']
                        })
        
        # 按可用显存排序
        suitable_nodes.sort(key=lambda x: x['total_available_memory'], reverse=True)
        return suitable_nodes
    
    def reserve_gpu_memory(self, node_name: str, gpu_count: int, memory_gb: float) -> bool:
        """预留GPU显存"""
        try:
            # 这里可以实现具体的显存预留逻辑
            # 例如通过节点标签或自定义资源来标记预留的显存
            labels = {
                f"reserved-gpu-memory-{int(memory_gb)}gb": str(gpu_count)
            }
            return self.k8s_client.label_node([node_name], labels)
        except Exception as e:
            logging.error(f"预留GPU显存失败: {e}")
            return False 
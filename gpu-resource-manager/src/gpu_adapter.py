from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class NodeGPUInfo:
    """节点GPU信息"""
    node_name: str
    gpu_type: str
    total_gpus: int
    available_gpus: int
    memory_per_gpu: float
    driver_version: str

class GPUAdapter(ABC):
    """GPU适配器基类"""
    
    @abstractmethod
    def get_gpu_nodes(self) -> List[NodeGPUInfo]:
        """获取GPU节点信息"""
        pass
    
    @abstractmethod
    def allocate_gpu(self, node_name: str, gpu_count: float) -> bool:
        """分配GPU资源"""
        pass
    
    @abstractmethod
    def release_gpu(self, node_name: str, gpu_count: float) -> bool:
        """释放GPU资源"""
        pass

class NvidiaGPUAdapter(GPUAdapter):
    """NVIDIA GPU适配器"""
    
    def __init__(self, k8s_client):
        self.k8s_client = k8s_client
        self.resource_name = "nvidia.com/gpu"
    
    def get_gpu_nodes(self) -> List[NodeGPUInfo]:
        """获取NVIDIA GPU节点信息"""
        nodes = []
        try:
            # 获取所有节点
            node_list = self.k8s_client.v1.list_node()
            
            for node in node_list.items:
                # 检查节点是否有GPU
                if self.resource_name in node.status.allocatable:
                    total_gpus = int(node.status.allocatable[self.resource_name])
                    
                    # 计算已使用的GPU
                    used_gpus = self._get_used_gpus_on_node(node.metadata.name)
                    available_gpus = total_gpus - used_gpus
                    
                    # 获取GPU型号
                    gpu_type = node.metadata.labels.get('gpu-type', 'Unknown')
                    
                    nodes.append(NodeGPUInfo(
                        node_name=node.metadata.name,
                        gpu_type=gpu_type,
                        total_gpus=total_gpus,
                        available_gpus=available_gpus,
                        memory_per_gpu=self._get_gpu_memory(gpu_type),
                        driver_version=node.metadata.labels.get('nvidia-driver-version', 'Unknown')
                    ))
        except Exception as e:
            print(f"获取NVIDIA GPU节点信息失败: {e}")
        
        return nodes
    
    def _get_used_gpus_on_node(self, node_name: str) -> int:
        """获取节点上已使用的GPU数量"""
        used_gpus = 0
        try:
            # 获取节点上的所有Pod
            pods = self.k8s_client.v1.list_pod_for_all_namespaces(
                field_selector=f"spec.nodeName={node_name}"
            )
            
            for pod in pods.items:
                if pod.status.phase in ['Running', 'Pending']:
                    for container in pod.spec.containers:
                        if container.resources and container.resources.requests:
                            gpu_request = container.resources.requests.get(self.resource_name, '0')
                            used_gpus += int(gpu_request)
        except Exception as e:
            print(f"获取节点{node_name}已使用GPU失败: {e}")
        
        return used_gpus
    
    def _get_gpu_memory(self, gpu_type: str) -> float:
        """根据GPU型号获取显存大小"""
        memory_mapping = {
            'T4': 16.0,
            'V100': 32.0,
            'A100': 80.0,
            'H100': 80.0,
            'RTX3090': 24.0,
            'RTX4090': 24.0
        }
        return memory_mapping.get(gpu_type, 16.0)  # 默认16GB
    
    def allocate_gpu(self, node_name: str, gpu_count: float) -> bool:
        """分配GPU资源"""
        # 这里实现GPU分配逻辑
        # 在实际实现中，这通常通过Kubernetes调度器完成
        return True
    
    def release_gpu(self, node_name: str, gpu_count: float) -> bool:
        """释放GPU资源"""
        # 这里实现GPU释放逻辑
        return True

class GPUAdapterFactory:
    """GPU适配器工厂"""
    
    @staticmethod
    def create_adapter(gpu_vendor: str, k8s_client) -> GPUAdapter:
        """创建GPU适配器"""
        if gpu_vendor.lower() == 'nvidia':
            return NvidiaGPUAdapter(k8s_client)
        else:
            raise ValueError(f"不支持的GPU厂商: {gpu_vendor}") 
import re
import math
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum

class GPUType(Enum):
    """GPU类型枚举"""
    NVIDIA = "nvidia"
    AMD = "amd"
    INTEL = "intel"
    NPU = "npu"

@dataclass
class GPUResource:
    """GPU资源配置"""
    gpu_num: float
    gpu_type: Optional[str]
    resource_name: str
    memory_gb: Optional[float] = None
    compute_ratio: Optional[float] = None

class GPUResourceParser:
    """GPU资源解析器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.gpu_resource_mapping = self.config.get('GPU_RESOURCE', {
            "gpu": "nvidia.com/gpu",
            "nvidia": "nvidia.com/gpu",
            "amd": "amd.com/gpu",
            "intel": "intel.com/gpu",
            "npu": "huawei.com/npu"
        })
        self.default_gpu_resource_name = self.config.get('DEFAULT_GPU_RESOURCE_NAME', 'nvidia.com/gpu')
        
        # GPU显存配置 (GB)
        self.gpu_memory_mapping = {
            'T4': 16,
            'V100': 32,
            'A100': 80,
            'H100': 80,
            'RTX3090': 24,
            'RTX4090': 24
        }
    
    def parse_gpu_resource(self, resource_gpu: str, resource_name: Optional[str] = None) -> GPUResource:
        """
        解析GPU资源字符串
        支持格式：
        - "2" -> 2个GPU
        - "0.5" -> 0.5个GPU (虚拟化)
        - "2(V100)" -> 2个V100 GPU
        - "1(nvidia,V100)" -> 1个NVIDIA V100
        - "8G,0.5" -> 8GB显存，0.5算力比例
        """
        if not resource_gpu:
            return GPUResource(0, None, resource_name or self.default_gpu_resource_name)
        
        gpu_num = 0
        gpu_type = None
        memory_gb = None
        compute_ratio = None
        
        try:
            # 提取括号内容 (支持中英文括号)
            gpu_type = self._extract_gpu_type(resource_gpu)
            
            # 处理GPU厂商和型号
            if gpu_type:
                resource_name, gpu_type = self._process_gpu_vendor_and_type(gpu_type, resource_name)
            
            # 移除括号内容
            resource_gpu = self._remove_brackets(resource_gpu)
            
            # 解析显存和算力比例
            if ',' in resource_gpu:
                memory_str, compute_str = resource_gpu.split(',', 1)
                memory_gb = self._parse_memory(memory_str.strip())
                compute_ratio = float(compute_str.strip())
                gpu_num = compute_ratio
            else:
                gpu_num = float(resource_gpu)
                
        except Exception as e:
            print(f"GPU资源解析错误: {e}")
            gpu_num = 0
        
        # 标准化GPU类型
        if gpu_type:
            gpu_type = gpu_type.upper()
        
        # 整数化处理
        if isinstance(gpu_num, float) and gpu_num == int(gpu_num):
            gpu_num = int(gpu_num)
        
        return GPUResource(
            gpu_num=gpu_num,
            gpu_type=gpu_type,
            resource_name=resource_name or self.default_gpu_resource_name,
            memory_gb=memory_gb,
            compute_ratio=compute_ratio
        )
    
    def _extract_gpu_type(self, resource_gpu: str) -> Optional[str]:
        """提取GPU类型信息"""
        # 英文括号
        if '(' in resource_gpu:
            match = re.findall(r"\((.+?)\)", resource_gpu)
            return match[0] if match else None
        # 中文括号
        if '（' in resource_gpu:
            match = re.findall(r"（(.+?)）", resource_gpu)
            return match[0] if match else None
        return None
    
    def _process_gpu_vendor_and_type(self, gpu_type: str, resource_name: Optional[str]) -> Tuple[str, str]:
        """处理GPU厂商和型号"""
        # 检查是否是厂商类型
        if gpu_type.lower() in self.gpu_resource_mapping:
            gpu_vendor = gpu_type.lower()
            resource_name = self.gpu_resource_mapping.get(gpu_vendor, resource_name)
            return resource_name, None
        
        # 处理 "厂商,型号" 格式
        if ',' in gpu_type:
            vendor, model = gpu_type.split(',', 1)
            vendor = vendor.strip().lower()
            model = model.strip().upper()
            
            if vendor in self.gpu_resource_mapping:
                resource_name = self.gpu_resource_mapping.get(vendor, resource_name)
            
            return resource_name, model
        
        return resource_name, gpu_type
    
    def _remove_brackets(self, resource_gpu: str) -> str:
        """移除括号内容"""
        if '(' in resource_gpu:
            resource_gpu = resource_gpu[:resource_gpu.index('(')]
        if '（' in resource_gpu:
            resource_gpu = resource_gpu[:resource_gpu.index('（')]
        return resource_gpu
    
    def _parse_memory(self, memory_str: str) -> Optional[float]:
        """解析显存大小"""
        if not memory_str:
            return None
        
        memory_str = memory_str.upper()
        if 'G' in memory_str:
            return float(memory_str.replace('G', ''))
        elif 'M' in memory_str:
            return float(memory_str.replace('M', '')) / 1024
        else:
            return float(memory_str)
    
    def validate_gpu_memory(self, gpu_resource: GPUResource, min_memory_gb: float) -> bool:
        """验证GPU显存是否满足最小要求"""
        if gpu_resource.memory_gb:
            return gpu_resource.memory_gb >= min_memory_gb
        
        # 根据GPU型号估算显存
        if gpu_resource.gpu_type and gpu_resource.gpu_type in self.gpu_memory_mapping:
            estimated_memory = self.gpu_memory_mapping[gpu_resource.gpu_type] * gpu_resource.gpu_num
            return estimated_memory >= min_memory_gb
        
        return True  # 无法验证时默认通过 
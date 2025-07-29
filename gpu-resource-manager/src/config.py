import os
from typing import Dict, Any

class ResourceConfig:
    """资源配置管理"""
    
    def __init__(self):
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置"""
        return {
            # GPU资源配置
            'GPU_RESOURCE': {
                "gpu": "nvidia.com/gpu",
                "nvidia": "nvidia.com/gpu",
                "amd": "amd.com/gpu",
                "intel": "intel.com/gpu",
                "npu": "huawei.com/npu"
            },
            
            # 默认GPU资源名称
            'DEFAULT_GPU_RESOURCE_NAME': 'nvidia.com/gpu',
            
            # GPU显存配置 (GB)
            'GPU_MEMORY_SPECS': {
                'T4': 16.0,
                'V100': 32.0,
                'A100': 80.0,
                'H100': 80.0,
                'RTX3090': 24.0,
                'RTX4090': 24.0,
                'A6000': 48.0,
                'A40': 48.0
            },
            
            # 监控配置
            'MONITOR_UPDATE_INTERVAL': int(os.getenv('MONITOR_UPDATE_INTERVAL', '30')),
            'RESOURCE_CHANGE_THRESHOLD': float(os.getenv('RESOURCE_CHANGE_THRESHOLD', '0.05')),
            
            # Kubernetes配置
            'KUBECONFIG_PATH': os.getenv('KUBECONFIG', ''),
            
            # 日志配置
            'LOG_LEVEL': os.getenv('LOG_LEVEL', 'INFO'),
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        return self.config.get(key, default)
    
    def update(self, updates: Dict[str, Any]):
        """更新配置"""
        self.config.update(updates)

# 全局配置实例
config = ResourceConfig() 
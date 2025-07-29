class ResourceManagerException(Exception):
    """资源管理器基础异常"""
    pass

class GPUResourceException(ResourceManagerException):
    """GPU资源相关异常"""
    pass

class K8sResourceException(ResourceManagerException):
    """Kubernetes资源相关异常"""
    pass

class MonitoringException(ResourceManagerException):
    """监控相关异常"""
    pass

class MemoryGuardException(ResourceManagerException):
    """显存保障相关异常"""
    pass 
"""
GPU资源管理API接口
提供GPU资源查询、分配和监控功能
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from datetime import datetime

from ..algorithm_engine.gpu_resource_integration import (
    get_gpu_resource_manager,
    TrainingGPUConfig
)

router = APIRouter(prefix="/api/v1/gpu", tags=["GPU资源管理"])


class GPUResourceRequest(BaseModel):
    """GPU资源请求"""
    gpu_count: int = 1
    gpu_type: str = "V100"
    memory_gb: float = 32.0
    compute_ratio: float = 1.0
    distributed_training: bool = False
    mixed_precision: bool = True
    gpu_memory_fraction: float = 0.9


class GPUAllocationResponse(BaseModel):
    """GPU分配响应"""
    success: bool
    node_name: Optional[str] = None
    gpu_config: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


class GPUResourceStatus(BaseModel):
    """GPU资源状态"""
    node_name: str
    gpu_type: str
    total_gpus: int
    available_gpus: int
    memory_per_gpu: float
    utilization: float
    temperature: Optional[float] = None
    power_usage: Optional[float] = None


class GPUMonitoringData(BaseModel):
    """GPU监控数据"""
    utilization: Dict[str, float]
    available_nodes: List[Dict[str, Any]]
    total_nodes: int
    timestamp: str


@router.get("/status", response_model=GPUMonitoringData)
async def get_gpu_resource_status():
    """获取GPU资源状态"""
    try:
        gpu_manager = get_gpu_resource_manager()
        
        if not gpu_manager.initialized:
            raise HTTPException(status_code=503, detail="GPU资源管理器未初始化")
        
        monitoring_data = gpu_manager.get_gpu_monitoring_data()
        
        return GPUMonitoringData(
            utilization=monitoring_data.get('utilization', {}),
            available_nodes=monitoring_data.get('available_nodes', []),
            total_nodes=monitoring_data.get('total_nodes', 0),
            timestamp=monitoring_data.get('timestamp', datetime.now().isoformat())
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取GPU资源状态失败: {str(e)}")


@router.get("/nodes", response_model=List[GPUResourceStatus])
async def get_available_gpu_nodes(
    gpu_type: Optional[str] = None,
    min_memory_gb: float = 16.0
):
    """获取可用的GPU节点"""
    try:
        gpu_manager = get_gpu_resource_manager()
        
        if not gpu_manager.initialized:
            raise HTTPException(status_code=503, detail="GPU资源管理器未初始化")
        
        available_nodes = gpu_manager.get_available_gpu_nodes(gpu_type, min_memory_gb)
        
        return [
            GPUResourceStatus(
                node_name=node.node_name,
                gpu_type=node.gpu_type,
                total_gpus=node.total_gpus,
                available_gpus=node.available_gpus,
                memory_per_gpu=node.memory_per_gpu,
                utilization=node.utilization,
                temperature=node.temperature,
                power_usage=node.power_usage
            )
            for node in available_nodes
        ]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取GPU节点失败: {str(e)}")


@router.post("/allocate", response_model=GPUAllocationResponse)
async def allocate_gpu_resources(request: GPUResourceRequest):
    """分配GPU资源"""
    try:
        gpu_manager = get_gpu_resource_manager()
        
        if not gpu_manager.initialized:
            raise HTTPException(status_code=503, detail="GPU资源管理器未初始化")
        
        # 创建GPU配置
        gpu_config = TrainingGPUConfig(
            gpu_count=request.gpu_count,
            gpu_type=request.gpu_type,
            memory_gb=request.memory_gb,
            compute_ratio=request.compute_ratio,
            distributed_training=request.distributed_training,
            mixed_precision=request.mixed_precision,
            gpu_memory_fraction=request.gpu_memory_fraction
        )
        
        # 验证GPU需求
        if not gpu_manager.validate_gpu_requirements(gpu_config):
            return GPUAllocationResponse(
                success=False,
                error_message="GPU资源需求不满足"
            )
        
        # 分配GPU资源
        allocated_node = gpu_manager.allocate_gpu_resources(gpu_config)
        
        if allocated_node:
            return GPUAllocationResponse(
                success=True,
                node_name=allocated_node,
                gpu_config=gpu_config.__dict__
            )
        else:
            return GPUAllocationResponse(
                success=False,
                error_message="无法分配GPU资源"
            )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"分配GPU资源失败: {str(e)}")


@router.post("/validate")
async def validate_gpu_requirements(request: GPUResourceRequest):
    """验证GPU需求是否满足"""
    try:
        gpu_manager = get_gpu_resource_manager()
        
        if not gpu_manager.initialized:
            raise HTTPException(status_code=503, detail="GPU资源管理器未初始化")
        
        # 创建GPU配置
        gpu_config = TrainingGPUConfig(
            gpu_count=request.gpu_count,
            gpu_type=request.gpu_type,
            memory_gb=request.memory_gb,
            compute_ratio=request.compute_ratio,
            distributed_training=request.distributed_training,
            mixed_precision=request.mixed_precision,
            gpu_memory_fraction=request.gpu_memory_fraction
        )
        
        # 验证需求
        is_valid = gpu_manager.validate_gpu_requirements(gpu_config)
        
        return {
            "valid": is_valid,
            "gpu_config": gpu_config.__dict__,
            "available_nodes": gpu_manager.get_available_gpu_nodes(
                gpu_type=request.gpu_type,
                min_memory_gb=request.memory_gb
            )
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"验证GPU需求失败: {str(e)}")


@router.delete("/cleanup/{node_name}")
async def cleanup_gpu_resources(node_name: str, gpu_count: int = 1):
    """清理GPU资源"""
    try:
        gpu_manager = get_gpu_resource_manager()
        
        if not gpu_manager.initialized:
            raise HTTPException(status_code=503, detail="GPU资源管理器未初始化")
        
        success = gpu_manager.cleanup_gpu_resources(node_name, gpu_count)
        
        return {
            "success": success,
            "node_name": node_name,
            "gpu_count": gpu_count
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"清理GPU资源失败: {str(e)}")


@router.get("/monitoring")
async def get_gpu_monitoring_data():
    """获取GPU监控数据"""
    try:
        gpu_manager = get_gpu_resource_manager()
        
        if not gpu_manager.initialized:
            raise HTTPException(status_code=503, detail="GPU资源管理器未初始化")
        
        monitoring_data = gpu_manager.get_gpu_monitoring_data()
        
        return {
            "status": "success",
            "data": monitoring_data,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取GPU监控数据失败: {str(e)}")


@router.post("/setup-tensorflow")
async def setup_tensorflow_gpu(request: GPUResourceRequest):
    """设置TensorFlow GPU环境"""
    try:
        gpu_manager = get_gpu_resource_manager()
        
        if not gpu_manager.initialized:
            raise HTTPException(status_code=503, detail="GPU资源管理器未初始化")
        
        # 创建GPU配置
        gpu_config = TrainingGPUConfig(
            gpu_count=request.gpu_count,
            gpu_type=request.gpu_type,
            memory_gb=request.memory_gb,
            compute_ratio=request.compute_ratio,
            distributed_training=request.distributed_training,
            mixed_precision=request.mixed_precision,
            gpu_memory_fraction=request.gpu_memory_fraction
        )
        
        # 设置TensorFlow GPU环境
        tensorflow_gpu = gpu_manager.tensorflow_gpu
        tf_config = tensorflow_gpu.setup_tensorflow_gpu(gpu_config)
        
        return {
            "success": True,
            "tensorflow_config": tf_config,
            "gpu_config": gpu_config.__dict__
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"设置TensorFlow GPU失败: {str(e)}")


@router.post("/setup-pytorch")
async def setup_pytorch_gpu(request: GPUResourceRequest):
    """设置PyTorch GPU环境"""
    try:
        gpu_manager = get_gpu_resource_manager()
        
        if not gpu_manager.initialized:
            raise HTTPException(status_code=503, detail="GPU资源管理器未初始化")
        
        # 创建GPU配置
        gpu_config = TrainingGPUConfig(
            gpu_count=request.gpu_count,
            gpu_type=request.gpu_type,
            memory_gb=request.memory_gb,
            compute_ratio=request.compute_ratio,
            distributed_training=request.distributed_training,
            mixed_precision=request.mixed_precision,
            gpu_memory_fraction=request.gpu_memory_fraction
        )
        
        # 设置PyTorch GPU环境
        pytorch_gpu = gpu_manager.pytorch_gpu
        pytorch_config = pytorch_gpu.setup_pytorch_gpu(gpu_config)
        
        return {
            "success": True,
            "pytorch_config": pytorch_config,
            "gpu_config": gpu_config.__dict__
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"设置PyTorch GPU失败: {str(e)}")


@router.get("/health")
async def gpu_resource_health_check():
    """GPU资源管理器健康检查"""
    try:
        gpu_manager = get_gpu_resource_manager()
        
        return {
            "status": "healthy" if gpu_manager.initialized else "unhealthy",
            "initialized": gpu_manager.initialized,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        } 
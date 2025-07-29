"""
Epoch训练API接口
提供TensorFlow/PyTorch epoch轮次训练的REST API
"""
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from typing import Dict, Any, Optional
import json
import asyncio
from datetime import datetime

from ..algorithm_engine.epoch_trainers import (
    EpochDeepLearningTrainer, TrainingState, epoch_training_manager
)
from ..shared.models import User
from ..auth.auth import get_current_user

router = APIRouter(prefix="/api/v1/epoch-training", tags=["Epoch训练"])


@router.post("/start")
async def start_epoch_training(
    config: Dict[str, Any],
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """启动epoch训练"""
    try:
        # 验证配置
        required_fields = ['name', 'algorithm_type', 'epochs', 'batch_size', 'learning_rate']
        for field in required_fields:
            if field not in config:
                raise HTTPException(status_code=400, detail=f"缺少必需字段: {field}")
        
        # 生成任务ID
        task_id = f"epoch_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 在后台启动训练
        background_tasks.add_task(
            run_epoch_training,
            task_id=task_id,
            config=config,
            user_id=current_user.id
        )
        
        return {
            "success": True,
            "task_id": task_id,
            "message": "Epoch训练已启动",
            "config": config
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"启动训练失败: {str(e)}")


@router.get("/progress/{task_id}")
async def get_training_progress(task_id: str):
    """获取训练进度"""
    try:
        progress = epoch_training_manager.get_training_progress()
        
        return {
            "success": True,
            "task_id": task_id,
            "progress": progress
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取进度失败: {str(e)}")


@router.post("/pause/{task_id}")
async def pause_training(task_id: str):
    """暂停训练"""
    try:
        epoch_training_manager.pause_training()
        
        return {
            "success": True,
            "task_id": task_id,
            "message": "训练已暂停"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"暂停训练失败: {str(e)}")


@router.post("/resume/{task_id}")
async def resume_training(task_id: str):
    """恢复训练"""
    try:
        epoch_training_manager.resume_training()
        
        return {
            "success": True,
            "task_id": task_id,
            "message": "训练已恢复"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"恢复训练失败: {str(e)}")


@router.get("/history/{task_id}")
async def get_training_history(task_id: str):
    """获取训练历史"""
    try:
        history = epoch_training_manager.epoch_history
        
        return {
            "success": True,
            "task_id": task_id,
            "history": [metrics.__dict__ for metrics in history]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取训练历史失败: {str(e)}")


@router.get("/stream-progress/{task_id}")
async def stream_training_progress(task_id: str):
    """流式获取训练进度（SSE）"""
    
    async def generate_progress():
        """生成进度流"""
        while True:
            try:
                progress = epoch_training_manager.get_training_progress()
                
                # 检查训练是否完成
                if progress['training_state'] in ['completed', 'failed']:
                    yield f"data: {json.dumps({'type': 'complete', 'progress': progress})}\n\n"
                    break
                
                # 发送进度数据
                yield f"data: {json.dumps({'type': 'progress', 'progress': progress})}\n\n"
                
                # 等待1秒
                await asyncio.sleep(1)
                
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
                break
    
    return StreamingResponse(
        generate_progress(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream"
        }
    )


@router.post("/config")
async def generate_training_config(
    model_type: str = "mlp",
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    current_user: User = Depends(get_current_user)
):
    """生成训练配置"""
    try:
        config = {
            "name": f"深度学习模型训练_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "algorithm_type": "deep_learning",
            "model_type": model_type,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "hidden_units": [128, 64, 32],
            "dropout_rate": 0.2,
            "early_stopping_patience": 10,
            "learning_rate_scheduler": "step",
            "feature_columns": ["feature1", "feature2", "feature3"],
            "target_column": "target",
            "output_path": f"/models/deep_learning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        }
        
        return {
            "success": True,
            "config": config
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成配置失败: {str(e)}")


@router.get("/supported-models")
async def get_supported_models():
    """获取支持的模型类型"""
    try:
        models = {
            "tensorflow": {
                "mlp": "多层感知机",
                "cnn": "卷积神经网络",
                "lstm": "长短期记忆网络",
                "gru": "门控循环单元"
            },
            "pytorch": {
                "mlp": "多层感知机",
                "cnn": "卷积神经网络",
                "lstm": "长短期记忆网络",
                "gru": "门控循环单元",
                "transformer": "Transformer模型"
            }
        }
        
        return {
            "success": True,
            "models": models
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取模型列表失败: {str(e)}")


@router.get("/supported-schedulers")
async def get_supported_schedulers():
    """获取支持的学习率调度器"""
    try:
        schedulers = {
            "step": {
                "name": "步进调度",
                "description": "每隔固定epoch降低学习率",
                "parameters": {
                    "step_size": "降低学习率的间隔",
                    "gamma": "学习率衰减因子"
                }
            },
            "plateau": {
                "name": "平台调度",
                "description": "当验证指标不再改善时降低学习率",
                "parameters": {
                    "patience": "等待改善的epoch数",
                    "factor": "学习率衰减因子"
                }
            },
            "cosine": {
                "name": "余弦退火",
                "description": "使用余弦函数平滑降低学习率",
                "parameters": {
                    "T_max": "最大epoch数",
                    "eta_min": "最小学习率"
                }
            }
        }
        
        return {
            "success": True,
            "schedulers": schedulers
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取调度器列表失败: {str(e)}")


async def run_epoch_training(task_id: str, config: Dict[str, Any], user_id: int):
    """在后台运行epoch训练"""
    try:
        # 模拟数据（实际应用中应该从数据库或文件加载）
        data = {
            "train_data": np.random.randn(1000, len(config.get('feature_columns', []))),
            "train_labels": np.random.randint(0, 2, 1000),
            "val_data": np.random.randn(200, len(config.get('feature_columns', []))),
            "val_labels": np.random.randint(0, 2, 200)
        }
        
        # 执行训练
        result = await epoch_training_manager.train_with_epochs(config, data, task_id)
        
        logger.info(f"训练完成 - 任务ID: {task_id}, 结果: {result}")
        
    except Exception as e:
        logger.error(f"训练失败 - 任务ID: {task_id}, 错误: {str(e)}")
        epoch_training_manager.training_state = TrainingState.FAILED


# 示例配置
EXAMPLE_CONFIG = {
    "name": "深度学习模型训练",
    "algorithm_type": "deep_learning",
    "model_type": "mlp",
    "epochs": 100,
    "batch_size": 32,
    "learning_rate": 0.001,
    "hidden_units": [128, 64, 32],
    "dropout_rate": 0.2,
    "early_stopping_patience": 10,
    "learning_rate_scheduler": "step",
    "feature_columns": ["feature1", "feature2", "feature3"],
    "target_column": "target",
    "output_path": "/models/deep_learning_model"
} 
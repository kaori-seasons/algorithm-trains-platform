"""
算法引擎API接口
提供算法训练、交互式调试、模型管理等功能
"""
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from typing import Dict, Any, List, Optional
import os
import json
from datetime import datetime

from ..algorithm_engine.trainers import (
    TrainerFactory, AsyncTrainingManager, ParameterGenerator, 
    ModelManager, ModelEvaluator
)
from ..algorithm_engine.models import (
    TrainingConfig, TrainingResult, AlgorithmType, TrainingStatus,
    ModelFormat
)
from ..shared.database import get_db
from ..shared.models import User
from ..auth.auth import get_current_user

router = APIRouter(prefix="/api/v1/algorithm", tags=["算法引擎"])

# 全局训练管理器
training_manager = AsyncTrainingManager()


@router.post("/train")
async def start_training(
    config: TrainingConfig,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """启动算法训练"""
    try:
        # 验证算法类型
        if config.algorithm_type not in AlgorithmType:
            raise HTTPException(status_code=400, detail="不支持的算法类型")
        
        # 创建训练任务
        task_id = await training_manager.start_training(config, {})
        
        return {
            "success": True,
            "task_id": task_id,
            "message": "训练任务已启动"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"启动训练失败: {str(e)}")


@router.get("/train/status/{task_id}")
async def get_training_status(task_id: str):
    """获取训练任务状态"""
    try:
        result = await training_manager.get_task_status(task_id)
        if result is None:
            raise HTTPException(status_code=404, detail="任务不存在")
        
        return {
            "success": True,
            "task_id": task_id,
            "status": result.status.value,
            "result": result.dict() if result.status == TrainingStatus.COMPLETED else None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取任务状态失败: {str(e)}")


@router.get("/train/tasks")
async def get_all_training_tasks():
    """获取所有训练任务"""
    try:
        tasks = await training_manager.get_all_tasks()
        return {
            "success": True,
            "tasks": {task_id: task.dict() for task_id, task in tasks.items()}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取任务列表失败: {str(e)}")


@router.delete("/train/cancel/{task_id}")
async def cancel_training_task(task_id: str):
    """取消训练任务"""
    try:
        success = await training_manager.cancel_task(task_id)
        if not success:
            raise HTTPException(status_code=404, detail="任务不存在或已完成")
        
        return {
            "success": True,
            "message": "任务已取消"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"取消任务失败: {str(e)}")


@router.post("/interactive-debug")
async def interactive_debug(
    task_id: str,
    debug_params: Dict[str, Any],
    current_user: User = Depends(get_current_user)
):
    """交互式调试 - 不依赖模型训练"""
    try:
        # 获取原始训练结果
        original_result = await training_manager.get_task_status(task_id)
        if original_result is None:
            raise HTTPException(status_code=404, detail="原始训练任务不存在")
        
        # 获取对应的训练器
        trainer = TrainerFactory.get_trainer(original_result.algorithm_type)
        
        # 执行交互式调试
        if hasattr(trainer, 'interactive_debug'):
            result = await trainer.interactive_debug(debug_params)
            
            return {
                "success": True,
                "task_id": task_id,
                "result": result,
                "message": "交互式调试完成"
            }
        else:
            raise HTTPException(status_code=400, detail="该算法类型不支持交互式调试")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"交互式调试失败: {str(e)}")


@router.get("/data-analysis/{task_id}")
async def get_data_analysis(task_id: str):
    """获取数据分析结果"""
    try:
        result = await training_manager.get_task_status(task_id)
        if result is None:
            raise HTTPException(status_code=404, detail="任务不存在")
        
        if result.status != TrainingStatus.COMPLETED:
            raise HTTPException(status_code=400, detail="训练任务尚未完成")
        
        # 获取训练器
        trainer = TrainerFactory.get_trainer(result.algorithm_type)
        
        if hasattr(trainer, 'get_data_analysis'):
            data_analysis = await trainer.get_data_analysis()
            
            return {
                "success": True,
                "task_id": task_id,
                "data_analysis": data_analysis
            }
        else:
            raise HTTPException(status_code=400, detail="该算法类型不支持数据分析")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取数据分析失败: {str(e)}")


@router.get("/debug-suggestions/{task_id}")
async def get_debug_suggestions(task_id: str):
    """获取调试建议"""
    try:
        result = await training_manager.get_task_status(task_id)
        if result is None:
            raise HTTPException(status_code=404, detail="任务不存在")
        
        if result.status != TrainingStatus.COMPLETED:
            raise HTTPException(status_code=400, detail="训练任务尚未完成")
        
        # 获取训练器
        trainer = TrainerFactory.get_trainer(result.algorithm_type)
        
        if hasattr(trainer, 'get_debug_suggestions'):
            suggestions = await trainer.get_debug_suggestions()
            
            return {
                "success": True,
                "task_id": task_id,
                "suggestions": suggestions
            }
        else:
            raise HTTPException(status_code=400, detail="该算法类型不支持调试建议")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取调试建议失败: {str(e)}")


@router.post("/apply-debug-params")
async def apply_debug_parameters(
    task_id: str,
    debug_params: Dict[str, Any],
    current_user: User = Depends(get_current_user)
):
    """应用调试参数并重新训练"""
    try:
        # 获取原始训练结果
        original_result = await training_manager.get_task_status(task_id)
        if original_result is None:
            raise HTTPException(status_code=404, detail="原始训练任务不存在")
        
        # 创建新的训练配置，应用调试参数
        config = TrainingConfig(
            algorithm_type=original_result.algorithm_type,
            train_data_path=original_result.metadata.get('train_data_path', ''),
            feature_columns=original_result.metadata.get('feature_columns', []),
            target_column=original_result.metadata.get('target_column', ''),
            algorithm_params=original_result.metadata.get('algorithm_params', {}),
            output_path=original_result.metadata.get('output_path', ''),
            save_model=True,
            model_format=ModelFormat.M
        )
        
        # 获取对应的训练器
        trainer = TrainerFactory.get_trainer(config.algorithm_type)
        
        # 应用调试参数到训练器
        if hasattr(trainer, 'interactive_params'):
            trainer.interactive_params.update(debug_params)
        
        # 重新训练
        result = await trainer.train(config, {})
        
        return {
            "success": True,
            "task_id": result.task_id,
            "result": result.dict(),
            "message": "调试参数应用完成，模型已重新训练"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"应用调试参数失败: {str(e)}")


@router.get("/visualization/{task_id}")
async def get_visualization_data(task_id: str):
    """获取可视化数据"""
    try:
        result = await training_manager.get_task_status(task_id)
        if result is None:
            raise HTTPException(status_code=404, detail="任务不存在")
        
        if result.status != TrainingStatus.COMPLETED:
            raise HTTPException(status_code=400, detail="训练任务尚未完成")
        
        visualization_data = result.metadata.get('visualization_data', {})
        
        return {
            "success": True,
            "task_id": task_id,
            "visualization_data": visualization_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取可视化数据失败: {str(e)}")


@router.get("/optimal-parameters/{task_id}")
async def get_optimal_parameters(task_id: str):
    """获取最优参数建议"""
    try:
        result = await training_manager.get_task_status(task_id)
        if result is None:
            raise HTTPException(status_code=404, detail="任务不存在")
        
        if result.status != TrainingStatus.COMPLETED:
            raise HTTPException(status_code=400, detail="训练任务尚未完成")
        
        # 获取训练器
        trainer = TrainerFactory.get_trainer(result.algorithm_type)
        
        if hasattr(trainer, 'get_optimal_parameters'):
            optimal_params = await trainer.get_optimal_parameters()
            
            return {
                "success": True,
                "task_id": task_id,
                "optimal_parameters": optimal_params
            }
        else:
            raise HTTPException(status_code=400, detail="该算法类型不支持参数优化建议")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取最优参数失败: {str(e)}")


@router.post("/generate-parameters")
async def generate_parameters(
    algorithm_type: AlgorithmType,
    config: Dict[str, Any] = {},
    current_user: User = Depends(get_current_user)
):
    """生成算法参数"""
    try:
        params = await ParameterGenerator.generate_parameters(algorithm_type, config)
        
        return {
            "success": True,
            "algorithm_type": algorithm_type.value,
            "parameters": params
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成参数失败: {str(e)}")


@router.post("/generate-random-parameters")
async def generate_random_parameters(
    algorithm_type: AlgorithmType,
    current_user: User = Depends(get_current_user)
):
    """生成随机参数"""
    try:
        params = ParameterGenerator.generate_random_parameters(algorithm_type)
        
        return {
            "success": True,
            "algorithm_type": algorithm_type.value,
            "random_parameters": params
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成随机参数失败: {str(e)}")


@router.get("/supported-algorithms")
async def get_supported_algorithms():
    """获取支持的算法类型"""
    try:
        algorithms = TrainerFactory.get_supported_algorithms()
        
        algorithm_info = {}
        for alg_type in algorithms:
            trainer = TrainerFactory.get_trainer(alg_type)
            params = await trainer.generate_parameters({})
            algorithm_info[alg_type.value] = params
        
        return {
            "success": True,
            "algorithms": algorithm_info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取算法列表失败: {str(e)}")


@router.post("/evaluate-model")
async def evaluate_model(
    model_path: str,
    test_data_path: str,
    feature_columns: List[str],
    target_column: str,
    model_format: str = 'joblib',
    current_user: User = Depends(get_current_user)
):
    """评估模型性能"""
    try:
        # 加载模型
        model = ModelManager.load_model(model_path, model_format)
        
        # 加载测试数据
        import pandas as pd
        test_data = pd.read_csv(test_data_path)
        X_test = test_data[feature_columns]
        y_test = test_data[target_column]
        
        # 评估模型
        metrics = ModelEvaluator.evaluate_model(model, X_test, y_test)
        
        return {
            "success": True,
            "model_path": model_path,
            "metrics": metrics
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"模型评估失败: {str(e)}")


@router.get("/download-model/{task_id}")
async def download_model(task_id: str):
    """下载训练好的模型"""
    try:
        result = await training_manager.get_task_status(task_id)
        if result is None:
            raise HTTPException(status_code=404, detail="任务不存在")
        
        if result.status != TrainingStatus.COMPLETED:
            raise HTTPException(status_code=400, detail="训练任务尚未完成")
        
        model_path = result.model_path
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail="模型文件不存在")
        
        # 返回模型文件
        return FileResponse(
            path=model_path,
            filename=os.path.basename(model_path),
            media_type='application/octet-stream'
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"下载模型失败: {str(e)}")


@router.get("/download-parameters/{task_id}")
async def download_parameters(task_id: str):
    """下载模型参数"""
    try:
        result = await training_manager.get_task_status(task_id)
        if result is None:
            raise HTTPException(status_code=404, detail="任务不存在")
        
        if result.status != TrainingStatus.COMPLETED:
            raise HTTPException(status_code=400, detail="训练任务尚未完成")
        
        parameters_path = result.parameters_path
        if not os.path.exists(parameters_path):
            raise HTTPException(status_code=404, detail="参数文件不存在")
        
        # 返回参数文件
        return FileResponse(
            path=parameters_path,
            filename=os.path.basename(parameters_path),
            media_type='application/json'
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"下载参数失败: {str(e)}")


@router.post("/predict")
async def predict(
    model_path: str,
    data: Dict[str, Any],
    model_format: str = 'joblib',
    current_user: User = Depends(get_current_user)
):
    """使用模型进行预测"""
    try:
        # 加载模型
        model = ModelManager.load_model(model_path, model_format)
        
        # 准备数据
        import pandas as pd
        input_data = pd.DataFrame([data])
        
        # 进行预测
        if hasattr(model, 'predict_proba'):
            prediction = model.predict_proba(input_data)
            result = {
                "prediction": model.predict(input_data).tolist(),
                "probabilities": prediction.tolist()
            }
        else:
            prediction = model.predict(input_data)
            result = {
                "prediction": prediction.tolist()
            }
        
        return {
            "success": True,
            "input_data": data,
            "result": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}") 


@router.post("/vibration/interactive-training")
async def vibration_interactive_training(
    task_id: str,
    training_params: Dict[str, Any],
    current_user: User = Depends(get_current_user)
):
    """振动算法交互式训练"""
    try:
        # 获取原始训练结果
        original_result = await training_manager.get_task_status(task_id)
        if original_result is None:
            raise HTTPException(status_code=404, detail="原始训练任务不存在")
        
        # 获取振动训练器
        trainer = TrainerFactory.get_trainer(original_result.algorithm_type)
        
        # 执行交互式训练
        if hasattr(trainer, 'interactive_training'):
            result = await trainer.interactive_training(training_params)
            
            return {
                "success": True,
                "task_id": task_id,
                "result": result,
                "message": "振动算法交互式训练完成"
            }
        else:
            raise HTTPException(status_code=400, detail="该算法类型不支持交互式训练")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"振动交互式训练失败: {str(e)}")


@router.get("/vibration/analysis/{task_id}")
async def get_vibration_analysis(task_id: str):
    """获取振动分析结果"""
    try:
        result = await training_manager.get_task_status(task_id)
        if result is None:
            raise HTTPException(status_code=404, detail="任务不存在")
        
        if result.status != TrainingStatus.COMPLETED:
            raise HTTPException(status_code=400, detail="训练任务尚未完成")
        
        # 获取振动训练器
        trainer = TrainerFactory.get_trainer(result.algorithm_type)
        
        if hasattr(trainer, 'get_vibration_analysis'):
            analysis = await trainer.get_vibration_analysis()
            
            return {
                "success": True,
                "task_id": task_id,
                "vibration_analysis": analysis
            }
        else:
            raise HTTPException(status_code=400, detail="该算法类型不支持振动分析")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取振动分析失败: {str(e)}")


@router.get("/vibration/visualization/{task_id}")
async def get_vibration_visualization(task_id: str):
    """获取振动可视化数据"""
    try:
        result = await training_manager.get_task_status(task_id)
        if result is None:
            raise HTTPException(status_code=404, detail="任务不存在")
        
        if result.status != TrainingStatus.COMPLETED:
            raise HTTPException(status_code=400, detail="训练任务尚未完成")
        
        # 获取振动训练器
        trainer = TrainerFactory.get_trainer(result.algorithm_type)
        
        if hasattr(trainer, 'get_vibration_visualization'):
            visualization = await trainer.get_vibration_visualization()
            
            return {
                "success": True,
                "task_id": task_id,
                "visualization_data": visualization
            }
        else:
            raise HTTPException(status_code=400, detail="该算法类型不支持振动可视化")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取振动可视化失败: {str(e)}")


@router.get("/vibration/real-time-config/{task_id}")
async def get_vibration_real_time_config(task_id: str):
    """获取振动实时配置"""
    try:
        result = await training_manager.get_task_status(task_id)
        if result is None:
            raise HTTPException(status_code=404, detail="任务不存在")
        
        # 获取振动训练器
        trainer = TrainerFactory.get_trainer(result.algorithm_type)
        
        if hasattr(trainer, 'get_real_time_config'):
            config = await trainer.get_real_time_config()
            
            return {
                "success": True,
                "task_id": task_id,
                "real_time_config": config
            }
        else:
            raise HTTPException(status_code=400, detail="该算法类型不支持实时配置")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取实时配置失败: {str(e)}")


@router.post("/vibration/apply-frequency-filter")
async def apply_frequency_filter(
    task_id: str,
    filter_config: Dict[str, Any],
    current_user: User = Depends(get_current_user)
):
    """应用频率过滤器"""
    try:
        # 获取原始训练结果
        original_result = await training_manager.get_task_status(task_id)
        if original_result is None:
            raise HTTPException(status_code=404, detail="原始训练任务不存在")
        
        # 获取振动训练器
        trainer = TrainerFactory.get_trainer(original_result.algorithm_type)
        
        # 应用频率过滤
        training_params = {
            'frequency_filtering': filter_config
        }
        
        if hasattr(trainer, 'interactive_training'):
            result = await trainer.interactive_training(training_params)
            
            return {
                "success": True,
                "task_id": task_id,
                "result": result,
                "message": "频率过滤应用完成"
            }
        else:
            raise HTTPException(status_code=400, detail="该算法类型不支持频率过滤")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"应用频率过滤失败: {str(e)}")


@router.post("/vibration/apply-amplitude-thresholds")
async def apply_amplitude_thresholds(
    task_id: str,
    threshold_config: Dict[str, Any],
    current_user: User = Depends(get_current_user)
):
    """应用振幅阈值"""
    try:
        # 获取原始训练结果
        original_result = await training_manager.get_task_status(task_id)
        if original_result is None:
            raise HTTPException(status_code=404, detail="原始训练任务不存在")
        
        # 获取振动训练器
        trainer = TrainerFactory.get_trainer(original_result.algorithm_type)
        
        # 应用振幅阈值
        training_params = {
            'amplitude_thresholds': threshold_config
        }
        
        if hasattr(trainer, 'interactive_training'):
            result = await trainer.interactive_training(training_params)
            
            return {
                "success": True,
                "task_id": task_id,
                "result": result,
                "message": "振幅阈值应用完成"
            }
        else:
            raise HTTPException(status_code=400, detail="该算法类型不支持振幅阈值")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"应用振幅阈值失败: {str(e)}")


@router.post("/vibration/select-data-range")
async def select_data_range(
    task_id: str,
    selection_config: Dict[str, Any],
    current_user: User = Depends(get_current_user)
):
    """选择数据范围"""
    try:
        # 获取原始训练结果
        original_result = await training_manager.get_task_status(task_id)
        if original_result is None:
            raise HTTPException(status_code=404, detail="原始训练任务不存在")
        
        # 获取振动训练器
        trainer = TrainerFactory.get_trainer(original_result.algorithm_type)
        
        # 应用数据选择
        training_params = {
            'data_selection': selection_config
        }
        
        if hasattr(trainer, 'interactive_training'):
            result = await trainer.interactive_training(training_params)
            
            return {
                "success": True,
                "task_id": task_id,
                "result": result,
                "message": "数据范围选择完成"
            }
        else:
            raise HTTPException(status_code=400, detail="该算法类型不支持数据选择")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"选择数据范围失败: {str(e)}")


@router.post("/vibration/update-real-time-config")
async def update_real_time_config(
    task_id: str,
    real_time_config: Dict[str, Any],
    current_user: User = Depends(get_current_user)
):
    """更新实时配置"""
    try:
        # 获取原始训练结果
        original_result = await training_manager.get_task_status(task_id)
        if original_result is None:
            raise HTTPException(status_code=404, detail="原始训练任务不存在")
        
        # 获取振动训练器
        trainer = TrainerFactory.get_trainer(original_result.algorithm_type)
        
        # 应用实时配置
        training_params = {
            'real_time_adjustment': real_time_config
        }
        
        if hasattr(trainer, 'interactive_training'):
            result = await trainer.interactive_training(training_params)
            
            return {
                "success": True,
                "task_id": task_id,
                "result": result,
                "message": "实时配置更新完成"
            }
        else:
            raise HTTPException(status_code=400, detail="该算法类型不支持实时配置")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"更新实时配置失败: {str(e)}") 
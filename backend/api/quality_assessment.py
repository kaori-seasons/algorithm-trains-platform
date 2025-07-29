 """
质量评估API接口
提供质量评估和重新预处理功能
"""
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from typing import Dict, Any, List, Optional
import os
import json
from datetime import datetime

from ..quality_assessment.service import QualityAssessmentService, QualityAssessment
from ..preprocessing.pipeline import PreprocessingPipeline, PreprocessingResult
from ..annotation.service import AnnotationService, AnnotationTask, AnnotationSegment, AnnotationType, SpeedLevel
from ..shared.database import get_db
from ..shared.models import User
from ..auth.auth import get_current_user

router = APIRouter(prefix="/api/v1/quality", tags=["质量评估"])

# 全局服务实例
quality_service = QualityAssessmentService()
preprocessing_pipeline = PreprocessingPipeline()
annotation_service = AnnotationService()


@router.post("/assess-vibration-quality")
async def assess_vibration_quality(
    task_id: str,
    assessment_config: Dict[str, Any] = {},
    current_user: User = Depends(get_current_user)
):
    """评估振动数据质量"""
    try:
        # 从训练结果中获取数据路径
        # 这里需要从训练管理器获取任务信息
        from ..algorithm_engine.trainers import AsyncTrainingManager
        training_manager = AsyncTrainingManager()
        
        result = await training_manager.get_task_status(task_id)
        if result is None:
            raise HTTPException(status_code=404, detail="训练任务不存在")
        
        # 加载振动数据
        import pandas as pd
        data = pd.read_csv(result.metadata.get('train_data_path', ''))
        
        # 执行质量评估
        assessment = await quality_service.assess_vibration_quality(data, assessment_config)
        
        return {
            "success": True,
            "task_id": task_id,
            "assessment": {
                "overall_score": assessment.overall_score,
                "quality_level": assessment.quality_level.value,
                "issues_count": len(assessment.issues),
                "channel_scores": assessment.channel_scores,
                "assessment_time": assessment.assessment_time.isoformat(),
                "issues": [
                    {
                        "type": issue.issue_type.value,
                        "severity": issue.severity,
                        "description": issue.description,
                        "suggested_action": issue.suggested_action,
                        "affected_channels": issue.affected_channels
                    }
                    for issue in assessment.issues
                ]
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"质量评估失败: {str(e)}")


@router.post("/reprocess-data")
async def reprocess_data(
    task_id: str,
    preprocessing_config: Dict[str, Any] = {},
    current_user: User = Depends(get_current_user)
):
    """重新预处理数据"""
    try:
        # 获取训练结果
        from ..algorithm_engine.trainers import AsyncTrainingManager
        training_manager = AsyncTrainingManager()
        
        result = await training_manager.get_task_status(task_id)
        if result is None:
            raise HTTPException(status_code=404, detail="训练任务不存在")
        
        # 加载原始数据
        import pandas as pd
        data = pd.read_csv(result.metadata.get('train_data_path', ''))
        
        # 获取质量评估结果
        assessment = await quality_service.assess_vibration_quality(data, {})
        
        # 执行重新预处理
        preprocessing_result = await preprocessing_pipeline.process_data(
            data, assessment.issues, preprocessing_config
        )
        
        # 保存预处理后的数据
        output_path = f"processed_data_{task_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        preprocessing_result.processed_data.to_csv(output_path, index=False)
        
        return {
            "success": True,
            "task_id": task_id,
            "preprocessing_result": {
                "original_shape": preprocessing_result.original_data.shape,
                "processed_shape": preprocessing_result.processed_data.shape,
                "quality_improvement": preprocessing_result.quality_improvement,
                "processing_time": preprocessing_result.processing_time,
                "output_path": output_path,
                "steps": [
                    {
                        "type": step.step_type.value,
                        "description": step.description,
                        "applied_channels": step.applied_channels,
                        "parameters": step.parameters
                    }
                    for step in preprocessing_result.steps
                ]
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"重新预处理失败: {str(e)}")


@router.post("/create-annotation-task")
async def create_annotation_task(
    data_path: str,
    task_config: Dict[str, Any] = {},
    current_user: User = Depends(get_current_user)
):
    """创建标注任务"""
    try:
        task_id = await annotation_service.create_annotation_task(data_path, task_config)
        
        return {
            "success": True,
            "task_id": task_id,
            "message": "标注任务创建成功"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"创建标注任务失败: {str(e)}")


@router.get("/annotation-task/{task_id}")
async def get_annotation_task(task_id: str):
    """获取标注任务"""
    try:
        task = await annotation_service.get_annotation_task(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="标注任务不存在")
        
        annotation_data = await annotation_service.get_annotation_data(task_id)
        
        return {
            "success": True,
            "task": annotation_data
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取标注任务失败: {str(e)}")


@router.post("/add-annotation-segment")
async def add_annotation_segment(
    task_id: str,
    segment_data: Dict[str, Any],
    current_user: User = Depends(get_current_user)
):
    """添加标注段"""
    try:
        # 创建标注段对象
        segment = AnnotationSegment(
            start_time=datetime.fromisoformat(segment_data['start_time']),
            end_time=datetime.fromisoformat(segment_data['end_time']),
            annotation_type=AnnotationType(segment_data['annotation_type']),
            label=segment_data['label'],
            confidence=segment_data.get('confidence', 1.0),
            metadata=segment_data.get('metadata', {})
        )
        
        success = await annotation_service.add_annotation_segment(task_id, segment)
        
        if success:
            return {
                "success": True,
                "message": "标注段添加成功"
            }
        else:
            raise HTTPException(status_code=400, detail="添加标注段失败")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"添加标注段失败: {str(e)}")


@router.delete("/remove-annotation-segment/{task_id}/{segment_index}")
async def remove_annotation_segment(
    task_id: str,
    segment_index: int,
    current_user: User = Depends(get_current_user)
):
    """删除标注段"""
    try:
        success = await annotation_service.remove_annotation_segment(task_id, segment_index)
        
        if success:
            return {
                "success": True,
                "message": "标注段删除成功"
            }
        else:
            raise HTTPException(status_code=400, detail="删除标注段失败")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除标注段失败: {str(e)}")


@router.get("/speed-annotations/{task_id}")
async def get_speed_annotations(task_id: str):
    """获取转速标注"""
    try:
        annotations = await annotation_service.get_speed_annotations(task_id)
        
        return {
            "success": True,
            "task_id": task_id,
            "speed_annotations": annotations
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取转速标注失败: {str(e)}")


@router.get("/quality-annotations/{task_id}")
async def get_quality_annotations(task_id: str):
    """获取质量标注"""
    try:
        annotations = await annotation_service.get_quality_annotations(task_id)
        
        return {
            "success": True,
            "task_id": task_id,
            "quality_annotations": annotations
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取质量标注失败: {str(e)}")


@router.get("/annotation-statistics/{task_id}")
async def get_annotation_statistics(task_id: str):
    """获取标注统计信息"""
    try:
        statistics = await annotation_service.get_annotation_statistics(task_id)
        
        return {
            "success": True,
            "task_id": task_id,
            "statistics": statistics
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取标注统计失败: {str(e)}")


@router.get("/export-annotations/{task_id}")
async def export_annotations(
    task_id: str,
    format: str = 'json'
):
    """导出标注结果"""
    try:
        export_data = await annotation_service.export_annotations(task_id, format)
        
        if format == 'json':
            return {
                "success": True,
                "task_id": task_id,
                "format": format,
                "data": json.loads(export_data)
            }
        else:
            # 返回CSV文件
            filename = f"annotations_{task_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            
            return FileResponse(
                content=export_data,
                filename=filename,
                media_type='text/csv'
            )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"导出标注失败: {str(e)}")


@router.get("/quality-assessment-history")
async def get_quality_assessment_history():
    """获取质量评估历史"""
    try:
        history = await quality_service.get_assessment_history()
        
        return {
            "success": True,
            "history": [
                {
                    "overall_score": assessment.overall_score,
                    "quality_level": assessment.quality_level.value,
                    "issues_count": len(assessment.issues),
                    "assessment_time": assessment.assessment_time.isoformat()
                }
                for assessment in history[-10:]  # 最近10次评估
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取评估历史失败: {str(e)}")


@router.get("/preprocessing-history")
async def get_preprocessing_history():
    """获取预处理历史"""
    try:
        history = await preprocessing_pipeline.get_processing_history()
        
        return {
            "success": True,
            "history": [
                {
                    "original_shape": result.original_data.shape,
                    "processed_shape": result.processed_data.shape,
                    "processing_time": result.processing_time,
                    "steps_count": len(result.steps)
                }
                for result in history[-10:]  # 最近10次处理
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取预处理历史失败: {str(e)}")


@router.get("/quality-assessment-summary")
async def get_quality_assessment_summary():
    """获取质量评估摘要"""
    try:
        summary = await quality_service.get_assessment_summary()
        
        return {
            "success": True,
            "summary": summary
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取评估摘要失败: {str(e)}")


@router.get("/preprocessing-summary")
async def get_preprocessing_summary():
    """获取预处理摘要"""
    try:
        summary = await preprocessing_pipeline.get_processing_summary()
        
        return {
            "success": True,
            "summary": summary
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取预处理摘要失败: {str(e)}")
 """
标注界面服务
提供类似Label Studio的时序数据标注功能
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import logging
from dataclasses import dataclass
from enum import Enum
import json

logger = logging.getLogger(__name__)


class AnnotationType(Enum):
    """标注类型"""
    SPEED_RANGE = "speed_range"  # 转速范围标注
    TIME_RANGE = "time_range"    # 时间范围标注
    ANOMALY_MARK = "anomaly_mark"  # 异常标记
    QUALITY_MARK = "quality_mark"  # 质量标记


class SpeedLevel(Enum):
    """转速等级"""
    LOW_SPEED = "low_speed"      # 低转速
    MEDIUM_SPEED = "medium_speed"  # 中转速
    HIGH_SPEED = "high_speed"    # 高转速


@dataclass
class AnnotationSegment:
    """标注段"""
    start_time: datetime
    end_time: datetime
    annotation_type: AnnotationType
    label: str
    confidence: float
    metadata: Dict[str, Any]


@dataclass
class AnnotationTask:
    """标注任务"""
    task_id: str
    data_path: str
    segments: List[AnnotationSegment]
    created_time: datetime
    updated_time: datetime
    status: str  # pending, in_progress, completed
    assigned_user: Optional[str] = None


class AnnotationService:
    """标注服务"""
    
    def __init__(self):
        self.annotation_tasks = {}
        self.annotation_history = []
    
    async def create_annotation_task(self, 
                                   data_path: str,
                                   task_config: Dict[str, Any]) -> str:
        """创建标注任务"""
        try:
            task_id = f"annotation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # 加载数据以获取时间范围
            data = pd.read_csv(data_path)
            if 'timestamp' in data.columns:
                data['timestamp'] = pd.to_datetime(data['timestamp'])
                start_time = data['timestamp'].min()
                end_time = data['timestamp'].max()
            else:
                start_time = datetime.now()
                end_time = datetime.now()
            
            # 创建标注任务
            task = AnnotationTask(
                task_id=task_id,
                data_path=data_path,
                segments=[],
                created_time=datetime.now(),
                updated_time=datetime.now(),
                status='pending',
                assigned_user=task_config.get('assigned_user')
            )
            
            self.annotation_tasks[task_id] = task
            logger.info(f"创建标注任务: {task_id}")
            
            return task_id
            
        except Exception as e:
            logger.error(f"创建标注任务失败: {str(e)}")
            raise
    
    async def get_annotation_task(self, task_id: str) -> Optional[AnnotationTask]:
        """获取标注任务"""
        return self.annotation_tasks.get(task_id)
    
    async def get_all_tasks(self) -> List[AnnotationTask]:
        """获取所有标注任务"""
        return list(self.annotation_tasks.values())
    
    async def add_annotation_segment(self, 
                                   task_id: str,
                                   segment: AnnotationSegment) -> bool:
        """添加标注段"""
        try:
            if task_id not in self.annotation_tasks:
                raise ValueError(f"标注任务不存在: {task_id}")
            
            task = self.annotation_tasks[task_id]
            task.segments.append(segment)
            task.updated_time = datetime.now()
            
            logger.info(f"添加标注段到任务 {task_id}: {segment.annotation_type.value}")
            return True
            
        except Exception as e:
            logger.error(f"添加标注段失败: {str(e)}")
            return False
    
    async def remove_annotation_segment(self, 
                                      task_id: str,
                                      segment_index: int) -> bool:
        """删除标注段"""
        try:
            if task_id not in self.annotation_tasks:
                raise ValueError(f"标注任务不存在: {task_id}")
            
            task = self.annotation_tasks[task_id]
            if 0 <= segment_index < len(task.segments):
                removed_segment = task.segments.pop(segment_index)
                task.updated_time = datetime.now()
                
                logger.info(f"删除标注段: {removed_segment.annotation_type.value}")
                return True
            else:
                raise ValueError(f"标注段索引无效: {segment_index}")
                
        except Exception as e:
            logger.error(f"删除标注段失败: {str(e)}")
            return False
    
    async def update_annotation_segment(self, 
                                      task_id: str,
                                      segment_index: int,
                                      updated_segment: AnnotationSegment) -> bool:
        """更新标注段"""
        try:
            if task_id not in self.annotation_tasks:
                raise ValueError(f"标注任务不存在: {task_id}")
            
            task = self.annotation_tasks[task_id]
            if 0 <= segment_index < len(task.segments):
                task.segments[segment_index] = updated_segment
                task.updated_time = datetime.now()
                
                logger.info(f"更新标注段: {updated_segment.annotation_type.value}")
                return True
            else:
                raise ValueError(f"标注段索引无效: {segment_index}")
                
        except Exception as e:
            logger.error(f"更新标注段失败: {str(e)}")
            return False
    
    async def get_annotation_data(self, task_id: str) -> Dict[str, Any]:
        """获取标注数据（用于前端显示）"""
        try:
            if task_id not in self.annotation_tasks:
                raise ValueError(f"标注任务不存在: {task_id}")
            
            task = self.annotation_tasks[task_id]
            
            # 加载原始数据
            data = pd.read_csv(task.data_path)
            
            # 准备标注数据
            annotation_data = {
                'task_id': task_id,
                'data_path': task.data_path,
                'segments': [],
                'time_series_data': {},
                'metadata': {
                    'total_rows': len(data),
                    'columns': list(data.columns),
                    'created_time': task.created_time.isoformat(),
                    'updated_time': task.updated_time.isoformat(),
                    'status': task.status
                }
            }
            
            # 转换标注段
            for segment in task.segments:
                annotation_data['segments'].append({
                    'start_time': segment.start_time.isoformat(),
                    'end_time': segment.end_time.isoformat(),
                    'annotation_type': segment.annotation_type.value,
                    'label': segment.label,
                    'confidence': segment.confidence,
                    'metadata': segment.metadata
                })
            
            # 准备时间序列数据（用于可视化）
            if 'timestamp' in data.columns:
                data['timestamp'] = pd.to_datetime(data['timestamp'])
                
                # 获取振动通道
                vibration_cols = [col for col in data.columns if 'accel' in col.lower() or 'vibration' in col.lower()]
                
                for col in vibration_cols:
                    # 采样数据以减少传输量
                    sample_size = min(1000, len(data))
                    sample_indices = np.linspace(0, len(data)-1, sample_size, dtype=int)
                    
                    annotation_data['time_series_data'][col] = {
                        'timestamps': data['timestamp'].iloc[sample_indices].dt.strftime('%Y-%m-%d %H:%M:%S').tolist(),
                        'values': data[col].iloc[sample_indices].tolist()
                    }
            
            return annotation_data
            
        except Exception as e:
            logger.error(f"获取标注数据失败: {str(e)}")
            raise
    
    async def export_annotations(self, task_id: str, format: str = 'json') -> str:
        """导出标注结果"""
        try:
            if task_id not in self.annotation_tasks:
                raise ValueError(f"标注任务不存在: {task_id}")
            
            task = self.annotation_tasks[task_id]
            
            if format == 'json':
                export_data = {
                    'task_id': task_id,
                    'export_time': datetime.now().isoformat(),
                    'segments': []
                }
                
                for segment in task.segments:
                    export_data['segments'].append({
                        'start_time': segment.start_time.isoformat(),
                        'end_time': segment.end_time.isoformat(),
                        'annotation_type': segment.annotation_type.value,
                        'label': segment.label,
                        'confidence': segment.confidence,
                        'metadata': segment.metadata
                    })
                
                return json.dumps(export_data, indent=2, ensure_ascii=False)
            
            elif format == 'csv':
                # 导出为CSV格式
                export_rows = []
                for segment in task.segments:
                    export_rows.append({
                        'start_time': segment.start_time.isoformat(),
                        'end_time': segment.end_time.isoformat(),
                        'annotation_type': segment.annotation_type.value,
                        'label': segment.label,
                        'confidence': segment.confidence
                    })
                
                df = pd.DataFrame(export_rows)
                return df.to_csv(index=False)
            
            else:
                raise ValueError(f"不支持的导出格式: {format}")
                
        except Exception as e:
            logger.error(f"导出标注失败: {str(e)}")
            raise
    
    async def get_speed_annotations(self, task_id: str) -> List[Dict[str, Any]]:
        """获取转速标注"""
        try:
            if task_id not in self.annotation_tasks:
                return []
            
            task = self.annotation_tasks[task_id]
            speed_annotations = []
            
            for segment in task.segments:
                if segment.annotation_type == AnnotationType.SPEED_RANGE:
                    speed_annotations.append({
                        'start_time': segment.start_time.isoformat(),
                        'end_time': segment.end_time.isoformat(),
                        'speed_level': segment.label,
                        'confidence': segment.confidence,
                        'metadata': segment.metadata
                    })
            
            return speed_annotations
            
        except Exception as e:
            logger.error(f"获取转速标注失败: {str(e)}")
            return []
    
    async def get_quality_annotations(self, task_id: str) -> List[Dict[str, Any]]:
        """获取质量标注"""
        try:
            if task_id not in self.annotation_tasks:
                return []
            
            task = self.annotation_tasks[task_id]
            quality_annotations = []
            
            for segment in task.segments:
                if segment.annotation_type == AnnotationType.QUALITY_MARK:
                    quality_annotations.append({
                        'start_time': segment.start_time.isoformat(),
                        'end_time': segment.end_time.isoformat(),
                        'quality_level': segment.label,
                        'confidence': segment.confidence,
                        'metadata': segment.metadata
                    })
            
            return quality_annotations
            
        except Exception as e:
            logger.error(f"获取质量标注失败: {str(e)}")
            return []
    
    async def get_annotation_statistics(self, task_id: str) -> Dict[str, Any]:
        """获取标注统计信息"""
        try:
            if task_id not in self.annotation_tasks:
                return {}
            
            task = self.annotation_tasks[task_id]
            
            # 统计各类型标注数量
            type_counts = {}
            label_counts = {}
            
            for segment in task.segments:
                # 按类型统计
                annotation_type = segment.annotation_type.value
                type_counts[annotation_type] = type_counts.get(annotation_type, 0) + 1
                
                # 按标签统计
                label = segment.label
                label_counts[label] = label_counts.get(label, 0) + 1
            
            # 计算总标注时长
            total_duration = 0
            for segment in task.segments:
                duration = (segment.end_time - segment.start_time).total_seconds()
                total_duration += duration
            
            return {
                'task_id': task_id,
                'total_segments': len(task.segments),
                'type_distribution': type_counts,
                'label_distribution': label_counts,
                'total_duration_seconds': total_duration,
                'average_confidence': np.mean([s.confidence for s in task.segments]) if task.segments else 0
            }
            
        except Exception as e:
            logger.error(f"获取标注统计失败: {str(e)}")
            return {}
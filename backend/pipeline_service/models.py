"""
Pipeline服务数据模型
定义Pipeline、Task、TaskExecution等核心数据模型
"""
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, Boolean, DateTime, ForeignKey, JSON, DECIMAL, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from pydantic import BaseModel, Field

from shared.database import Base

logger = logging.getLogger(__name__)


class Pipeline(Base):
    """Pipeline模型"""
    __tablename__ = "pipelines"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    version = Column(String(50), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    config = Column(JSON, nullable=False)
    status = Column(String(50), default="created")
    description = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    user = relationship("User", back_populates="pipelines")
    tasks = relationship("Task", back_populates="pipeline", cascade="all, delete-orphan")

    __table_args__ = (
        Index('idx_pipelines_user_id', 'user_id'),
        Index('idx_pipelines_status', 'status'),
    )


class Task(Base):
    """Task模型"""
    __tablename__ = "tasks"

    id = Column(Integer, primary_key=True, index=True)
    pipeline_id = Column(Integer, ForeignKey("pipelines.id"), nullable=False)
    name = Column(String(255), nullable=False)
    task_type = Column(String(100), nullable=False)  # doris_query, feast_feature, training, etc.
    config = Column(JSON, nullable=False)
    dependencies = Column(JSON, default=list)  # 依赖的任务ID列表
    status = Column(String(50), default="pending")
    order_index = Column(Integer, default=0)  # 执行顺序
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    pipeline = relationship("Pipeline", back_populates="tasks")
    executions = relationship("TaskExecution", back_populates="task", cascade="all, delete-orphan")

    __table_args__ = (
        Index('idx_tasks_pipeline_id', 'pipeline_id'),
        Index('idx_tasks_status', 'status'),
        Index('idx_tasks_type', 'task_type'),
    )


class TaskExecution(Base):
    """Task执行记录模型"""
    __tablename__ = "task_executions"

    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(Integer, ForeignKey("tasks.id"), nullable=False)
    pipeline_execution_id = Column(Integer, ForeignKey("pipeline_executions.id"), nullable=False)
    status = Column(String(50), default="pending")  # pending, running, completed, failed, cancelled
    start_time = Column(DateTime(timezone=True))
    end_time = Column(DateTime(timezone=True))
    duration = Column(Integer)  # 执行时长（秒）
    result = Column(JSON)  # 执行结果
    error_message = Column(Text)  # 错误信息
    logs = Column(Text)  # 执行日志
    resource_usage = Column(JSON)  # 资源使用情况
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    task = relationship("Task", back_populates="executions")
    pipeline_execution = relationship("PipelineExecution", back_populates="task_executions")

    __table_args__ = (
        Index('idx_task_executions_task_id', 'task_id'),
        Index('idx_task_executions_status', 'status'),
        Index('idx_task_executions_pipeline_execution_id', 'pipeline_execution_id'),
    )


class PipelineExecution(Base):
    """Pipeline执行记录模型"""
    __tablename__ = "pipeline_executions"

    id = Column(Integer, primary_key=True, index=True)
    pipeline_id = Column(Integer, ForeignKey("pipelines.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    status = Column(String(50), default="pending")  # pending, running, completed, failed, cancelled
    start_time = Column(DateTime(timezone=True))
    end_time = Column(DateTime(timezone=True))
    duration = Column(Integer)  # 执行时长（秒）
    parameters = Column(JSON)  # 执行参数
    result = Column(JSON)  # 执行结果
    error_message = Column(Text)  # 错误信息
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    pipeline = relationship("Pipeline")
    user = relationship("User")
    task_executions = relationship("TaskExecution", back_populates="pipeline_execution", cascade="all, delete-orphan")

    __table_args__ = (
        Index('idx_pipeline_executions_pipeline_id', 'pipeline_id'),
        Index('idx_pipeline_executions_user_id', 'user_id'),
        Index('idx_pipeline_executions_status', 'status'),
    )


# Pydantic模型用于API
class PipelineCreate(BaseModel):
    """创建Pipeline请求模型"""
    name: str = Field(..., description="Pipeline名称")
    version: str = Field(..., description="Pipeline版本")
    config: Dict[str, Any] = Field(..., description="Pipeline配置")
    description: Optional[str] = Field(None, description="Pipeline描述")


class PipelineUpdate(BaseModel):
    """更新Pipeline请求模型"""
    name: Optional[str] = Field(None, description="Pipeline名称")
    config: Optional[Dict[str, Any]] = Field(None, description="Pipeline配置")
    description: Optional[str] = Field(None, description="Pipeline描述")
    status: Optional[str] = Field(None, description="Pipeline状态")


class PipelineResponse(BaseModel):
    """Pipeline响应模型"""
    id: int
    name: str
    version: str
    user_id: int
    config: Dict[str, Any]
    status: str
    description: Optional[str]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class TaskCreate(BaseModel):
    """创建Task请求模型"""
    name: str = Field(..., description="Task名称")
    task_type: str = Field(..., description="Task类型")
    config: Dict[str, Any] = Field(..., description="Task配置")
    dependencies: List[int] = Field(default=list, description="依赖的Task ID列表")
    order_index: int = Field(default=0, description="执行顺序")


class TaskUpdate(BaseModel):
    """更新Task请求模型"""
    name: Optional[str] = Field(None, description="Task名称")
    config: Optional[Dict[str, Any]] = Field(None, description="Task配置")
    dependencies: Optional[List[int]] = Field(None, description="依赖的Task ID列表")
    order_index: Optional[int] = Field(None, description="执行顺序")
    status: Optional[str] = Field(None, description="Task状态")


class TaskResponse(BaseModel):
    """Task响应模型"""
    id: int
    pipeline_id: int
    name: str
    task_type: str
    config: Dict[str, Any]
    dependencies: List[int]
    status: str
    order_index: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class PipelineExecutionCreate(BaseModel):
    """创建Pipeline执行请求模型"""
    parameters: Optional[Dict[str, Any]] = Field(None, description="执行参数")


class PipelineExecutionResponse(BaseModel):
    """Pipeline执行响应模型"""
    id: int
    pipeline_id: int
    user_id: int
    status: str
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    duration: Optional[int]
    parameters: Optional[Dict[str, Any]]
    result: Optional[Dict[str, Any]]
    error_message: Optional[str]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class TaskExecutionResponse(BaseModel):
    """Task执行响应模型"""
    id: int
    task_id: int
    pipeline_execution_id: int
    status: str
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    duration: Optional[int]
    result: Optional[Dict[str, Any]]
    error_message: Optional[str]
    logs: Optional[str]
    resource_usage: Optional[Dict[str, Any]]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# 任务类型定义
class TaskType:
    """任务类型常量"""
    DORIS_QUERY = "doris_query"
    FEAST_FEATURE = "feast_feature"
    TRAINING_SET_GENERATION = "training_set_generation"
    MODEL_TRAINING = "model_training"
    MODEL_EVALUATION = "model_evaluation"
    MODEL_DEPLOYMENT = "model_deployment"
    DATA_PREPROCESSING = "data_preprocessing"
    FEATURE_ENGINEERING = "feature_engineering"
    INCREMENTAL_LEARNING = "incremental_learning"
    CUSTOM_SCRIPT = "custom_script"


# Pipeline状态定义
class PipelineStatus:
    """Pipeline状态常量"""
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


# Task状态定义
class TaskStatus:
    """Task状态常量"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"


# 任务配置模板
class TaskConfigTemplates:
    """任务配置模板"""
    
    @staticmethod
    def doris_query_config(table_name: str, time_range: Dict[str, str], 
                          filters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Doris查询任务配置模板"""
        return {
            "table_name": table_name,
            "time_range": time_range,
            "filters": filters or {},
            "limit": None,
            "output_format": "parquet"
        }
    
    @staticmethod
    def feast_feature_config(feature_view_name: str, entity_df_path: str) -> Dict[str, Any]:
        """Feast特征工程任务配置模板"""
        return {
            "feature_view_name": feature_view_name,
            "entity_df_path": entity_df_path,
            "output_path": None,  # 自动生成
            "ttl": None
        }
    
    @staticmethod
    def training_set_generation_config(training_set_name: str, 
                                     feature_views: List[str]) -> Dict[str, Any]:
        """训练集生成任务配置模板"""
        return {
            "training_set_name": training_set_name,
            "feature_views": feature_views,
            "quality_assessment": True,
            "version_control": True
        }
    
    @staticmethod
    def model_training_config(model_type: str, training_data_path: str,
                            hyperparameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """模型训练任务配置模板"""
        return {
            "model_type": model_type,
            "training_data_path": training_data_path,
            "hyperparameters": hyperparameters or {},
            "validation_split": 0.2,
            "epochs": 100,
            "batch_size": 32,
            "output_path": None  # 自动生成
        }
    
    @staticmethod
    def incremental_learning_config(base_model_path: str, new_data_path: str,
                                  learning_rate: float = 0.001) -> Dict[str, Any]:
        """增量学习任务配置模板"""
        return {
            "base_model_path": base_model_path,
            "new_data_path": new_data_path,
            "learning_rate": learning_rate,
            "epochs": 10,
            "batch_size": 16,
            "knowledge_distillation": True,
            "output_path": None  # 自动生成
        } 
"""
数据库模型定义
"""
from datetime import datetime
from typing import Optional, Dict, Any, List
from sqlalchemy import Column, Integer, String, Text, Boolean, DateTime, ForeignKey, JSON, DECIMAL, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

Base = declarative_base()


class User(Base):
    """用户模型"""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(100), unique=True, index=True, nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    full_name = Column(String(255))
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # 关系
    pipelines = relationship("Pipeline", back_populates="user")
    storage_mounts = relationship("StorageMount", back_populates="user")
    training_set_versions = relationship("TrainingSetVersion", back_populates="user")
    model_versions = relationship("ModelVersion", back_populates="user")


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

    # 关系
    user = relationship("User", back_populates="pipelines")
    tasks = relationship("Task", back_populates="pipeline", cascade="all, delete-orphan")

    __table_args__ = (
        Index('idx_pipelines_user_id', 'user_id'),
        Index('idx_pipelines_status', 'status'),
    )


class Task(Base):
    """任务模型"""
    __tablename__ = "tasks"

    id = Column(Integer, primary_key=True, index=True)
    pipeline_id = Column(Integer, ForeignKey("pipelines.id"), nullable=False)
    name = Column(String(255), nullable=False)
    type = Column(String(100), nullable=False)
    config = Column(JSON, nullable=False)
    status = Column(String(50), default="pending")
    dependencies = Column(JSON, default=list)
    resources = Column(JSON, default=dict)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # 关系
    pipeline = relationship("Pipeline", back_populates="tasks")
    executions = relationship("TaskExecution", back_populates="task", cascade="all, delete-orphan")

    __table_args__ = (
        Index('idx_tasks_pipeline_id', 'pipeline_id'),
        Index('idx_tasks_status', 'status'),
    )


class TaskExecution(Base):
    """任务执行记录模型"""
    __tablename__ = "task_executions"

    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(Integer, ForeignKey("tasks.id"), nullable=False)
    pipeline_execution_id = Column(String(255), nullable=False)
    status = Column(String(50), nullable=False)
    start_time = Column(DateTime(timezone=True))
    end_time = Column(DateTime(timezone=True))
    logs = Column(Text)
    result = Column(JSON)
    error_message = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # 关系
    task = relationship("Task", back_populates="executions")

    __table_args__ = (
        Index('idx_task_executions_pipeline_execution_id', 'pipeline_execution_id'),
    )


class TrainingSetVersion(Base):
    """训练集版本模型"""
    __tablename__ = "training_set_versions"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    version_id = Column(String(255), unique=True, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    doris_query_config = Column(JSON, nullable=False)
    feast_config = Column(JSON, nullable=False)
    quality_score = Column(DECIMAL(3, 2))
    status = Column(String(50), default="created")
    data_path = Column(Text)
    metadata = Column(JSON, default=dict)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # 关系
    user = relationship("User", back_populates="training_set_versions")
    model_versions = relationship("ModelVersion", back_populates="training_set_version")

    __table_args__ = (
        Index('idx_training_set_versions_user_id', 'user_id'),
        Index('idx_training_set_versions_status', 'status'),
    )


class FeatureSnapshot(Base):
    """特征快照模型"""
    __tablename__ = "feature_snapshots"

    id = Column(Integer, primary_key=True, index=True)
    uuid = Column(String(255), nullable=False)
    node_id = Column(String(255), nullable=False)
    time = Column(DateTime(timezone=True), nullable=False)
    features = Column(JSON, nullable=False)
    is_tag = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index('idx_uuid', 'uuid'),
        Index('idx_node_id', 'node_id'),
        Index('idx_time', 'time'),
    )


class StorageMount(Base):
    """存储挂载模型"""
    __tablename__ = "storage_mounts"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    mount_path = Column(String(500), nullable=False)
    storage_type = Column(String(50), nullable=False)
    config = Column(JSON, nullable=False)
    status = Column(String(50), default="mounted")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # 关系
    user = relationship("User", back_populates="storage_mounts")

    __table_args__ = (
        Index('idx_storage_mounts_user_id', 'user_id'),
    )


class ModelVersion(Base):
    """模型版本模型"""
    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    version = Column(String(50), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    model_path = Column(Text, nullable=False)
    config = Column(JSON, nullable=False)
    metrics = Column(JSON, default=dict)
    status = Column(String(50), default="created")
    training_set_version_id = Column(Integer, ForeignKey("training_set_versions.id"))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # 关系
    user = relationship("User", back_populates="model_versions")
    training_set_version = relationship("TrainingSetVersion", back_populates="model_versions")

    __table_args__ = (
        Index('idx_model_versions_user_id', 'user_id'),
    )


class IncrementalLearningRecord(Base):
    """增量学习记录模型"""
    __tablename__ = "incremental_learning_records"

    id = Column(Integer, primary_key=True, index=True)
    base_model_version_id = Column(Integer, ForeignKey("model_versions.id"), nullable=False)
    new_model_version_id = Column(Integer, ForeignKey("model_versions.id"), nullable=False)
    training_data_path = Column(Text, nullable=False)
    learning_config = Column(JSON, nullable=False)
    performance_improvement = Column(JSON)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class SystemConfig(Base):
    """系统配置模型"""
    __tablename__ = "system_configs"

    id = Column(Integer, primary_key=True, index=True)
    key = Column(String(255), unique=True, nullable=False)
    value = Column(JSON, nullable=False)
    description = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class AuditLog(Base):
    """审计日志模型"""
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    action = Column(String(100), nullable=False)
    resource_type = Column(String(100), nullable=False)
    resource_id = Column(String(255))
    details = Column(JSON)
    ip_address = Column(String(45))  # IPv6支持
    user_agent = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index('idx_audit_logs_user_id', 'user_id'),
        Index('idx_audit_logs_created_at', 'created_at'),
    ) 
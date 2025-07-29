"""
算法引擎数据模型
定义算法训练相关的数据结构
"""
from enum import Enum
from typing import Dict, Any, Optional, List
from datetime import datetime
from pydantic import BaseModel, Field


class AlgorithmType(str, Enum):
    """算法类型枚举"""
    STATUS_RECOGNITION = "status_recognition"      # 状态识别
    HEALTH_ASSESSMENT = "health_assessment"        # 健康度评估
    VIBRATION_ANALYSIS = "vibration_analysis"      # 振动分析
    SIMULATION = "simulation"                      # 模拟算法


class TrainingStatus(str, Enum):
    """训练状态枚举"""
    PENDING = "pending"       # 等待中
    RUNNING = "running"       # 运行中
    COMPLETED = "completed"   # 已完成
    FAILED = "failed"         # 失败
    CANCELLED = "cancelled"   # 已取消


class ModelFormat(str, Enum):
    """模型格式枚举"""
    JSON = "json"             # JSON参数
    PICKLE = "pickle"         # Python pickle
    ONNX = "onnx"            # ONNX格式
    TENSORFLOW = "tensorflow" # TensorFlow SavedModel
    PYTORCH = "pytorch"       # PyTorch模型
    CUSTOM = "custom"         # 自定义格式


class TrainingConfig(BaseModel):
    """训练配置模型"""
    name: str = Field(..., description="训练任务名称")
    algorithm_type: AlgorithmType = Field(..., description="算法类型")
    model_format: ModelFormat = Field(default=ModelFormat.JSON, description="模型输出格式")
    
    # 训练参数
    epochs: int = Field(default=1, description="训练轮次")
    batch_size: int = Field(default=32, description="批次大小")
    learning_rate: float = Field(default=0.001, description="学习率")
    
    # 数据配置
    train_data_path: str = Field(..., description="训练数据路径")
    validation_data_path: Optional[str] = Field(None, description="验证数据路径")
    test_data_path: Optional[str] = Field(None, description="测试数据路径")
    
    # 特征配置
    feature_columns: List[str] = Field(default=[], description="特征列名")
    target_column: str = Field(..., description="目标列名")
    
    # 算法特定参数
    algorithm_params: Dict[str, Any] = Field(default={}, description="算法特定参数")
    
    # 输出配置
    output_path: str = Field(..., description="输出路径")
    save_model: bool = Field(default=True, description="是否保存模型")
    save_parameters: bool = Field(default=True, description="是否保存参数")
    
    class Config:
        schema_extra = {
            "example": {
                "name": "设备状态识别模型_v1",
                "algorithm_type": "status_recognition",
                "model_format": "json",
                "epochs": 10,
                "batch_size": 64,
                "learning_rate": 0.001,
                "train_data_path": "/data/train.csv",
                "validation_data_path": "/data/val.csv",
                "feature_columns": ["temp", "pressure", "vibration"],
                "target_column": "status",
                "algorithm_params": {
                    "threshold": 0.5,
                    "n_estimators": 100
                },
                "output_path": "/models/status_recognition_v1"
            }
        }


class TrainingResult(BaseModel):
    """训练结果模型"""
    task_id: str = Field(..., description="训练任务ID")
    algorithm_type: AlgorithmType = Field(..., description="算法类型")
    status: TrainingStatus = Field(..., description="训练状态")
    
    # 训练指标
    accuracy: Optional[float] = Field(None, description="准确率")
    precision: Optional[float] = Field(None, description="精确率")
    recall: Optional[float] = Field(None, description="召回率")
    f1_score: Optional[float] = Field(None, description="F1分数")
    
    # 训练时间
    start_time: datetime = Field(..., description="开始时间")
    end_time: Optional[datetime] = Field(None, description="结束时间")
    duration: Optional[float] = Field(None, description="训练时长(秒)")
    
    # 输出文件
    model_path: Optional[str] = Field(None, description="模型文件路径")
    parameters_path: Optional[str] = Field(None, description="参数文件路径")
    log_path: Optional[str] = Field(None, description="日志文件路径")
    
    # 错误信息
    error_message: Optional[str] = Field(None, description="错误信息")
    
    # 元数据
    metadata: Dict[str, Any] = Field(default={}, description="元数据")
    
    class Config:
        schema_extra = {
            "example": {
                "task_id": "status_recognition_20241201_143022_设备状态识别模型_v1",
                "algorithm_type": "status_recognition",
                "status": "completed",
                "accuracy": 0.95,
                "precision": 0.94,
                "recall": 0.96,
                "f1_score": 0.95,
                "start_time": "2024-12-01T14:30:22",
                "end_time": "2024-12-01T14:35:45",
                "duration": 323.5,
                "model_path": "/models/status_recognition_v1/model.json",
                "parameters_path": "/models/status_recognition_v1/parameters.json",
                "metadata": {
                    "version": "1.0.0",
                    "framework": "sklearn"
                }
            }
        }


class ParameterConfig(BaseModel):
    """参数配置模型"""
    algorithm_type: AlgorithmType = Field(..., description="算法类型")
    parameter_type: str = Field(..., description="参数类型")
    
    # 参数定义
    parameters: Dict[str, Any] = Field(..., description="参数定义")
    
    # 参数验证规则
    validation_rules: Dict[str, Any] = Field(default={}, description="验证规则")
    
    # 参数模板
    template_name: Optional[str] = Field(None, description="模板名称")
    
    class Config:
        schema_extra = {
            "example": {
                "algorithm_type": "vibration_analysis",
                "parameter_type": "bearing_parameters",
                "parameters": {
                    "bearing_type": "SKF_6205",
                    "inner_diameter": 25,
                    "outer_diameter": 52,
                    "width": 15,
                    "ball_diameter": 7.94,
                    "ball_count": 9
                },
                "validation_rules": {
                    "inner_diameter": {"min": 10, "max": 100},
                    "outer_diameter": {"min": 20, "max": 200}
                }
            }
        }


class InferenceRequest(BaseModel):
    """推理请求模型"""
    model_id: str = Field(..., description="模型ID")
    input_data: Dict[str, Any] = Field(..., description="输入数据")
    parameters: Optional[Dict[str, Any]] = Field(None, description="推理参数")
    
    class Config:
        schema_extra = {
            "example": {
                "model_id": "status_recognition_v1",
                "input_data": {
                    "temp": 45.2,
                    "pressure": 2.1,
                    "vibration": 0.15
                },
                "parameters": {
                    "threshold": 0.5
                }
            }
        }


class InferenceResult(BaseModel):
    """推理结果模型"""
    model_id: str = Field(..., description="模型ID")
    prediction: Any = Field(..., description="预测结果")
    confidence: Optional[float] = Field(None, description="置信度")
    processing_time: float = Field(..., description="处理时间(毫秒)")
    metadata: Dict[str, Any] = Field(default={}, description="元数据")
    
    class Config:
        schema_extra = {
            "example": {
                "model_id": "status_recognition_v1",
                "prediction": "normal",
                "confidence": 0.95,
                "processing_time": 12.5,
                "metadata": {
                    "algorithm_type": "status_recognition",
                    "version": "1.0.0"
                }
            }
        }
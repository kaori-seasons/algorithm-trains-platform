"""
交互式参数调优模块
支持可视化参数调整和实时效果预览
"""
import asyncio
import logging
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from .models import AlgorithmType, TrainingConfig
from .trainers import TrainerFactory

logger = logging.getLogger(__name__)


class ParameterType(Enum):
    """参数类型枚举"""
    THRESHOLD = "threshold"           # 阈值参数
    FEATURE_SELECTION = "feature"     # 特征选择
    HYPERPARAMETER = "hyperparameter" # 超参数
    ALGORITHM_SPECIFIC = "algorithm"  # 算法特定参数


@dataclass
class ParameterRange:
    """参数范围定义"""
    name: str
    min_value: float
    max_value: float
    step: float
    current_value: float
    description: str


@dataclass
class VisualizationData:
    """可视化数据"""
    chart_type: str
    data: Dict[str, Any]
    title: str
    x_label: str
    y_label: str
    interactive_points: List[Dict[str, Any]]


class ParameterValidator:
    """参数验证器"""
    
    def __init__(self):
        self.validation_rules = {}
    
    def validate_parameter(self, param_name: str, value: Any, algorithm_type: AlgorithmType) -> bool:
        """验证参数值是否有效"""
        try:
            if algorithm_type == AlgorithmType.STATUS_RECOGNITION:
                return self._validate_status_recognition_param(param_name, value)
            elif algorithm_type == AlgorithmType.HEALTH_ASSESSMENT:
                return self._validate_health_assessment_param(param_name, value)
            elif algorithm_type == AlgorithmType.VIBRATION_ANALYSIS:
                return self._validate_vibration_analysis_param(param_name, value)
            elif algorithm_type == AlgorithmType.SIMULATION:
                return self._validate_simulation_param(param_name, value)
            else:
                return True
        except Exception as e:
            logger.error(f"参数验证失败: {param_name}={value}, 错误: {e}")
            return False
    
    def _validate_status_recognition_param(self, param_name: str, value: Any) -> bool:
        """验证状态识别算法参数"""
        if param_name == "threshold":
            return 0.0 <= value <= 1.0
        elif param_name == "n_estimators":
            return isinstance(value, int) and 1 <= value <= 1000
        elif param_name == "max_depth":
            return value is None or (isinstance(value, int) and value > 0)
        return True
    
    def _validate_health_assessment_param(self, param_name: str, value: Any) -> bool:
        """验证健康度评估算法参数"""
        if param_name == "health_threshold":
            return 0.0 <= value <= 1.0
        elif param_name == "assessment_period":
            return isinstance(value, int) and value > 0
        return True
    
    def _validate_vibration_analysis_param(self, param_name: str, value: Any) -> bool:
        """验证振动分析算法参数"""
        if param_name == "frequency_threshold":
            return value > 0
        elif param_name == "amplitude_threshold":
            return value > 0
        elif param_name == "sampling_rate":
            return isinstance(value, int) and value > 0
        return True
    
    def _validate_simulation_param(self, param_name: str, value: Any) -> bool:
        """验证仿真算法参数"""
        if param_name == "simulation_steps":
            return isinstance(value, int) and value > 0
        elif param_name == "time_step":
            return value > 0
        return True


class VisualizationEngine:
    """可视化引擎"""
    
    def __init__(self):
        self.chart_templates = {}
        self._init_chart_templates()
    
    def _init_chart_templates(self):
        """初始化图表模板"""
        self.chart_templates = {
            "roc_curve": {
                "type": "line",
                "title": "ROC曲线",
                "x_label": "假正率 (FPR)",
                "y_label": "真正率 (TPR)",
                "interactive": True
            },
            "feature_importance": {
                "type": "bar",
                "title": "特征重要性",
                "x_label": "特征名称",
                "y_label": "重要性分数",
                "interactive": True
            },
            "parameter_sensitivity": {
                "type": "line",
                "title": "参数敏感性分析",
                "x_label": "参数值",
                "y_label": "模型性能",
                "interactive": True
            },
            "confusion_matrix": {
                "type": "heatmap",
                "title": "混淆矩阵",
                "x_label": "预测类别",
                "y_label": "真实类别",
                "interactive": False
            },
            "learning_curve": {
                "type": "line",
                "title": "学习曲线",
                "x_label": "训练样本数",
                "y_label": "准确率",
                "interactive": False
            }
        }
    
    def create_roc_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                        thresholds: List[float] = None) -> VisualizationData:
        """创建ROC曲线可视化"""
        from sklearn.metrics import roc_curve, auc
        
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        # 生成交互点（阈值点）
        if thresholds is None:
            thresholds = np.linspace(0, 1, 20)
        
        interactive_points = []
        for threshold in thresholds:
            y_pred_binary = (y_pred_proba >= threshold).astype(int)
            from sklearn.metrics import confusion_matrix
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
            tpr_point = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr_point = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            interactive_points.append({
                'x': fpr_point,
                'y': tpr_point,
                'threshold': threshold,
                'tpr': tpr_point,
                'fpr': fpr_point
            })
        
        return VisualizationData(
            chart_type="roc_curve",
            data={
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'auc': roc_auc,
                'interactive_points': interactive_points
            },
            title="ROC曲线",
            x_label="假正率 (FPR)",
            y_label="真正率 (TPR)",
            interactive_points=interactive_points
        )
    
    def create_feature_importance(self, feature_names: List[str], 
                                importance_scores: np.ndarray) -> VisualizationData:
        """创建特征重要性可视化"""
        # 排序特征重要性
        sorted_indices = np.argsort(importance_scores)[::-1]
        sorted_features = [feature_names[i] for i in sorted_indices]
        sorted_scores = importance_scores[sorted_indices]
        
        interactive_points = []
        for i, (feature, score) in enumerate(zip(sorted_features, sorted_scores)):
            interactive_points.append({
                'x': i,
                'y': score,
                'feature': feature,
                'importance': score,
                'selected': True  # 默认全选
            })
        
        return VisualizationData(
            chart_type="feature_importance",
            data={
                'features': sorted_features,
                'scores': sorted_scores.tolist(),
                'interactive_points': interactive_points
            },
            title="特征重要性",
            x_label="特征名称",
            y_label="重要性分数",
            interactive_points=interactive_points
        )
    
    def create_parameter_sensitivity(self, param_name: str, param_values: List[float], 
                                  performance_scores: List[float]) -> VisualizationData:
        """创建参数敏感性分析可视化"""
        interactive_points = []
        for i, (value, score) in enumerate(zip(param_values, performance_scores)):
            interactive_points.append({
                'x': value,
                'y': score,
                'param_value': value,
                'performance': score,
                'selected': False
            })
        
        return VisualizationData(
            chart_type="parameter_sensitivity",
            data={
                'param_values': param_values,
                'performance_scores': performance_scores,
                'interactive_points': interactive_points
            },
            title=f"{param_name}参数敏感性分析",
            x_label="参数值",
            y_label="模型性能",
            interactive_points=interactive_points
        )
    
    def create_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> VisualizationData:
        """创建混淆矩阵可视化"""
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)
        
        return VisualizationData(
            chart_type="confusion_matrix",
            data={
                'matrix': cm.tolist(),
                'labels': ['正常', '异常']  # 根据实际情况调整
            },
            title="混淆矩阵",
            x_label="预测类别",
            y_label="真实类别",
            interactive_points=[]
        )


class RealTimePreview:
    """实时预览引擎"""
    
    def __init__(self):
        self.preview_cache = {}
        self.preview_interval = 1.0  # 预览更新间隔（秒）
    
    async def setup_preview(self, algorithm_config: Dict[str, Any]) -> Dict[str, Any]:
        """设置实时预览"""
        preview_config = {
            'enabled': True,
            'update_interval': self.preview_interval,
            'metrics': ['accuracy', 'precision', 'recall', 'f1_score'],
            'visualization_types': ['roc_curve', 'confusion_matrix']
        }
        
        return {
            'config': preview_config,
            'status': 'ready'
        }
    
    async def update_preview(self, new_params: Dict[str, Any], 
                           algorithm_type: AlgorithmType) -> Dict[str, Any]:
        """更新预览结果"""
        try:
            # 使用新参数快速训练模型
            trainer = TrainerFactory.get_trainer(algorithm_type)
            
            # 创建临时配置
            temp_config = TrainingConfig(
                name="preview_training",
                algorithm_type=algorithm_type,
                train_data_path="",  # 使用缓存数据
                feature_columns=[],
                target_column="",
                output_path="",
                algorithm_params=new_params
            )
            
            # 快速预览训练（使用小数据集）
            preview_result = await self._quick_preview_training(trainer, temp_config)
            
            return {
                'status': 'success',
                'metrics': preview_result.get('metrics', {}),
                'visualization': preview_result.get('visualization', {}),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"预览更新失败: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def _quick_preview_training(self, trainer, config: TrainingConfig) -> Dict[str, Any]:
        """快速预览训练"""
        # 这里应该使用小数据集进行快速训练
        # 实际实现中需要根据具体的数据和算法类型来调整
        return {
            'metrics': {
                'accuracy': 0.85,
                'precision': 0.83,
                'recall': 0.87,
                'f1_score': 0.85
            },
            'visualization': {
                'roc_curve': {
                    'auc': 0.89,
                    'points': []
                }
            }
        }


class InteractiveParameterTuner:
    """
    交互式参数调优器
    支持可视化参数调整和实时效果预览
    """
    
    def __init__(self):
        self.visualization_engine = VisualizationEngine()
        self.parameter_validator = ParameterValidator()
        self.real_time_preview = RealTimePreview()
        self.current_session = None
        self.parameter_history = []
        
        logger.info("交互式参数调优器初始化完成")
    
    async def create_parameter_interface(self, algorithm_config: Dict[str, Any]) -> Dict[str, Any]:
        """创建参数调优界面"""
        try:
            algorithm_type = AlgorithmType(algorithm_config.get('algorithm_type'))
            
            # 生成参数配置界面
            interface_config = await self._generate_interface_config(algorithm_config)
            
            # 创建可视化组件
            visualization = await self._create_initial_visualization(algorithm_type, algorithm_config)
            
            # 设置实时预览
            preview = await self.real_time_preview.setup_preview(algorithm_config)
            
            # 保存当前会话
            self.current_session = {
                'algorithm_type': algorithm_type,
                'config': algorithm_config,
                'interface': interface_config,
                'visualization': visualization,
                'preview': preview,
                'created_at': datetime.now()
            }
            
            return {
                'interface': interface_config,
                'visualization': visualization,
                'preview': preview,
                'session_id': id(self.current_session)
            }
            
        except Exception as e:
            logger.error(f"创建参数调优界面失败: {e}")
            raise
    
    async def _generate_interface_config(self, algorithm_config: Dict[str, Any]) -> Dict[str, Any]:
        """生成界面配置"""
        algorithm_type = AlgorithmType(algorithm_config.get('algorithm_type'))
        
        # 根据算法类型生成不同的参数界面
        if algorithm_type == AlgorithmType.STATUS_RECOGNITION:
            return self._generate_status_recognition_interface(algorithm_config)
        elif algorithm_type == AlgorithmType.HEALTH_ASSESSMENT:
            return self._generate_health_assessment_interface(algorithm_config)
        elif algorithm_type == AlgorithmType.VIBRATION_ANALYSIS:
            return self._generate_vibration_analysis_interface(algorithm_config)
        elif algorithm_type == AlgorithmType.SIMULATION:
            return self._generate_simulation_interface(algorithm_config)
        else:
            return self._generate_generic_interface(algorithm_config)
    
    def _generate_status_recognition_interface(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """生成状态识别算法界面配置"""
        return {
            'algorithm_type': 'status_recognition',
            'parameters': {
                'threshold': {
                    'type': 'slider',
                    'min': 0.0,
                    'max': 1.0,
                    'step': 0.01,
                    'default': 0.5,
                    'description': '分类阈值'
                },
                'n_estimators': {
                    'type': 'number',
                    'min': 10,
                    'max': 1000,
                    'default': 100,
                    'description': '决策树数量'
                },
                'max_depth': {
                    'type': 'number',
                    'min': 1,
                    'max': 50,
                    'default': None,
                    'description': '最大深度'
                }
            },
            'visualizations': ['roc_curve', 'confusion_matrix', 'feature_importance'],
            'interactive_elements': ['threshold_selection', 'feature_selection']
        }
    
    def _generate_health_assessment_interface(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """生成健康度评估算法界面配置"""
        return {
            'algorithm_type': 'health_assessment',
            'parameters': {
                'health_threshold': {
                    'type': 'slider',
                    'min': 0.0,
                    'max': 1.0,
                    'step': 0.01,
                    'default': 0.7,
                    'description': '健康度阈值'
                },
                'assessment_period': {
                    'type': 'number',
                    'min': 1,
                    'max': 365,
                    'default': 30,
                    'description': '评估周期（天）'
                }
            },
            'visualizations': ['health_trend', 'assessment_distribution'],
            'interactive_elements': ['threshold_selection']
        }
    
    def _generate_vibration_analysis_interface(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """生成振动分析算法界面配置"""
        return {
            'algorithm_type': 'vibration_analysis',
            'parameters': {
                'frequency_threshold': {
                    'type': 'number',
                    'min': 0.1,
                    'max': 1000.0,
                    'default': 50.0,
                    'description': '频率阈值（Hz）'
                },
                'amplitude_threshold': {
                    'type': 'number',
                    'min': 0.01,
                    'max': 10.0,
                    'default': 0.5,
                    'description': '振幅阈值'
                },
                'sampling_rate': {
                    'type': 'number',
                    'min': 100,
                    'max': 10000,
                    'default': 1000,
                    'description': '采样率（Hz）'
                }
            },
            'visualizations': ['frequency_spectrum', 'amplitude_time', 'vibration_pattern'],
            'interactive_elements': ['frequency_selection', 'amplitude_selection']
        }
    
    def _generate_simulation_interface(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """生成仿真算法界面配置"""
        return {
            'algorithm_type': 'simulation',
            'parameters': {
                'simulation_steps': {
                    'type': 'number',
                    'min': 10,
                    'max': 10000,
                    'default': 1000,
                    'description': '仿真步数'
                },
                'time_step': {
                    'type': 'number',
                    'min': 0.001,
                    'max': 1.0,
                    'default': 0.01,
                    'description': '时间步长'
                }
            },
            'visualizations': ['simulation_trajectory', 'parameter_sensitivity'],
            'interactive_elements': ['parameter_adjustment']
        }
    
    def _generate_generic_interface(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """生成通用算法界面配置"""
        return {
            'algorithm_type': 'generic',
            'parameters': {},
            'visualizations': ['generic_chart'],
            'interactive_elements': []
        }
    
    async def _create_initial_visualization(self, algorithm_type: AlgorithmType, 
                                          config: Dict[str, Any]) -> Dict[str, Any]:
        """创建初始可视化"""
        # 这里应该根据实际数据创建可视化
        # 暂时返回模拟数据
        return {
            'roc_curve': {
                'type': 'roc_curve',
                'data': {
                    'fpr': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                    'tpr': [0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 1.0],
                    'auc': 0.85
                },
                'title': 'ROC曲线',
                'interactive': True
            },
            'feature_importance': {
                'type': 'feature_importance',
                'data': {
                    'features': ['特征1', '特征2', '特征3', '特征4', '特征5'],
                    'scores': [0.3, 0.25, 0.2, 0.15, 0.1]
                },
                'title': '特征重要性',
                'interactive': True
            }
        }
    
    async def update_parameters(self, new_params: Dict[str, Any]) -> Dict[str, Any]:
        """更新参数并返回新的结果"""
        try:
            if not self.current_session:
                raise ValueError("没有活跃的调参会话")
            
            # 验证参数
            algorithm_type = self.current_session['algorithm_type']
            for param_name, value in new_params.items():
                if not self.parameter_validator.validate_parameter(param_name, value, algorithm_type):
                    raise ValueError(f"参数验证失败: {param_name}={value}")
            
            # 更新预览
            preview_result = await self.real_time_preview.update_preview(new_params, algorithm_type)
            
            # 记录参数历史
            self.parameter_history.append({
                'params': new_params.copy(),
                'timestamp': datetime.now(),
                'preview_result': preview_result
            })
            
            return {
                'status': 'success',
                'preview': preview_result,
                'parameter_history': self.parameter_history[-5:],  # 最近5次
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"参数更新失败: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def apply_parameter_selection(self, selection_data: Dict[str, Any]) -> Dict[str, Any]:
        """应用用户选择的参数"""
        try:
            selection_type = selection_data.get('type')
            selection_value = selection_data.get('value')
            
            if selection_type == 'threshold':
                # 用户选择了阈值
                new_params = {'threshold': selection_value}
            elif selection_type == 'feature':
                # 用户选择了特征
                selected_features = selection_data.get('features', [])
                new_params = {'selected_features': selected_features}
            else:
                raise ValueError(f"不支持的参数选择类型: {selection_type}")
            
            # 更新参数
            result = await self.update_parameters(new_params)
            
            return {
                'status': 'success',
                'applied_selection': selection_data,
                'new_params': new_params,
                'preview_result': result
            }
            
        except Exception as e:
            logger.error(f"参数选择应用失败: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    async def get_parameter_history(self) -> List[Dict[str, Any]]:
        """获取参数调整历史"""
        return self.parameter_history
    
    async def export_optimal_parameters(self) -> Dict[str, Any]:
        """导出最优参数"""
        if not self.parameter_history:
            return {'status': 'error', 'message': '没有参数调整历史'}
        
        # 找到性能最好的参数组合
        best_result = max(self.parameter_history, 
                         key=lambda x: x.get('preview_result', {}).get('metrics', {}).get('f1_score', 0))
        
        return {
            'status': 'success',
            'optimal_params': best_result['params'],
            'performance': best_result['preview_result']['metrics'],
            'timestamp': best_result['timestamp'].isoformat()
        }
    
    async def reset_session(self) -> Dict[str, Any]:
        """重置调参会话"""
        self.current_session = None
        self.parameter_history = []
        
        return {
            'status': 'success',
            'message': '会话已重置'
        } 
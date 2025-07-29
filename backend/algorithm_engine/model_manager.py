"""
模型版本管理模块
支持模型版本控制、回滚、对比
"""
import os
import json
import shutil
import hashlib
import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum
import pickle
import joblib

from .models import AlgorithmType, TrainingResult

logger = logging.getLogger(__name__)


class ModelStatus(Enum):
    """模型状态枚举"""
    ACTIVE = "active"           # 活跃版本
    DEPRECATED = "deprecated"   # 已弃用
    ARCHIVED = "archived"       # 已归档
    TESTING = "testing"         # 测试中


@dataclass
class ModelMetadata:
    """模型元数据"""
    version_id: str
    algorithm_type: AlgorithmType
    model_path: str
    parameters_path: str
    created_at: datetime
    created_by: str
    description: str
    performance_metrics: Dict[str, float]
    training_config: Dict[str, Any]
    status: ModelStatus
    tags: List[str]
    dependencies: Dict[str, str]


class GitVersionControl:
    """Git版本控制模拟器"""
    
    def __init__(self, model_repo_path: str = "models"):
        self.model_repo_path = model_repo_path
        self.versions_file = os.path.join(model_repo_path, "versions.json")
        self._ensure_repo_exists()
    
    def _ensure_repo_exists(self):
        """确保模型仓库存在"""
        os.makedirs(self.model_repo_path, exist_ok=True)
        if not os.path.exists(self.versions_file):
            with open(self.versions_file, 'w') as f:
                json.dump({'versions': [], 'current_version': None}, f)
    
    def create_version(self, model, metadata: ModelMetadata) -> str:
        """创建模型版本"""
        # 生成版本ID
        version_id = self._generate_version_id(metadata)
        
        # 保存模型文件
        model_dir = os.path.join(self.model_repo_path, version_id)
        os.makedirs(model_dir, exist_ok=True)
        
        # 保存模型
        model_path = os.path.join(model_dir, "model.joblib")
        joblib.dump(model, model_path)
        
        # 保存元数据
        metadata_path = os.path.join(model_dir, "metadata.json")
        metadata_dict = asdict(metadata)
        metadata_dict['version_id'] = version_id
        metadata_dict['model_path'] = model_path
        metadata_dict['created_at'] = metadata.created_at.isoformat()
        metadata_dict['algorithm_type'] = metadata.algorithm_type.value
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata_dict, f, indent=2, ensure_ascii=False)
        
        # 更新版本记录
        self._add_version_record(version_id, metadata_dict)
        
        logger.info(f"创建模型版本: {version_id}")
        return version_id
    
    def _generate_version_id(self, metadata: ModelMetadata) -> str:
        """生成版本ID"""
        # 基于算法类型、时间戳和描述生成唯一ID
        content = f"{metadata.algorithm_type.value}_{metadata.created_at.isoformat()}_{metadata.description}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _add_version_record(self, version_id: str, metadata: Dict[str, Any]):
        """添加版本记录"""
        with open(self.versions_file, 'r') as f:
            data = json.load(f)
        
        data['versions'].append({
            'version_id': version_id,
            'created_at': metadata['created_at'],
            'algorithm_type': metadata['algorithm_type'],
            'status': metadata['status'].value,
            'description': metadata['description']
        })
        
        with open(self.versions_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    async def rollback(self, target_version: str) -> bool:
        """回滚到指定版本"""
        try:
            # 验证目标版本是否存在
            if not self._version_exists(target_version):
                raise ValueError(f"版本不存在: {target_version}")
            
            # 更新当前版本
            with open(self.versions_file, 'r') as f:
                data = json.load(f)
            
            data['current_version'] = target_version
            
            with open(self.versions_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"成功回滚到版本: {target_version}")
            return True
            
        except Exception as e:
            logger.error(f"回滚失败: {e}")
            return False
    
    def _version_exists(self, version_id: str) -> bool:
        """检查版本是否存在"""
        version_dir = os.path.join(self.model_repo_path, version_id)
        return os.path.exists(version_dir)
    
    def get_version_info(self, version_id: str) -> Optional[Dict[str, Any]]:
        """获取版本信息"""
        try:
            metadata_path = os.path.join(self.model_repo_path, version_id, "metadata.json")
            if not os.path.exists(metadata_path):
                return None
            
            with open(metadata_path, 'r', encoding='utf-8') as f:
                return json.load(f)
                
        except Exception as e:
            logger.error(f"获取版本信息失败: {e}")
            return None
    
    def list_versions(self) -> List[Dict[str, Any]]:
        """列出所有版本"""
        try:
            with open(self.versions_file, 'r') as f:
                data = json.load(f)
            return data.get('versions', [])
        except Exception as e:
            logger.error(f"列出版本失败: {e}")
            return []


class ModelRegistry:
    """模型注册表"""
    
    def __init__(self):
        self.registry_file = "model_registry.json"
        self._load_registry()
    
    def _load_registry(self):
        """加载注册表"""
        if os.path.exists(self.registry_file):
            with open(self.registry_file, 'r') as f:
                self.registry = json.load(f)
        else:
            self.registry = {'models': {}}
    
    def _save_registry(self):
        """保存注册表"""
        with open(self.registry_file, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    async def register_model(self, version_id: str, metadata: Dict[str, Any]):
        """注册模型"""
        self.registry['models'][version_id] = {
            'metadata': metadata,
            'registered_at': datetime.now().isoformat(),
            'status': 'registered'
        }
        self._save_registry()
        logger.info(f"注册模型: {version_id}")
    
    async def unregister_model(self, version_id: str):
        """注销模型"""
        if version_id in self.registry['models']:
            del self.registry['models'][version_id]
            self._save_registry()
            logger.info(f"注销模型: {version_id}")
    
    def get_registered_models(self) -> List[Dict[str, Any]]:
        """获取已注册的模型列表"""
        return list(self.registry['models'].values())


class PerformanceTracker:
    """性能跟踪器"""
    
    def __init__(self):
        self.performance_file = "model_performance.json"
        self._load_performance_data()
    
    def _load_performance_data(self):
        """加载性能数据"""
        if os.path.exists(self.performance_file):
            with open(self.performance_file, 'r') as f:
                self.performance_data = json.load(f)
        else:
            self.performance_data = {'models': {}}
    
    def _save_performance_data(self):
        """保存性能数据"""
        with open(self.performance_file, 'w') as f:
            json.dump(self.performance_data, f, indent=2)
    
    async def evaluate(self, model, test_data: Dict[str, Any] = None) -> Dict[str, float]:
        """评估模型性能"""
        # 这里应该实现实际的模型评估逻辑
        # 暂时返回模拟数据
        performance = {
            'accuracy': 0.85,
            'precision': 0.83,
            'recall': 0.87,
            'f1_score': 0.85,
            'inference_time': 0.05,
            'memory_usage': 128.5
        }
        
        return performance
    
    async def record_performance(self, version_id: str, performance: Dict[str, float]):
        """记录模型性能"""
        self.performance_data['models'][version_id] = {
            'performance': performance,
            'recorded_at': datetime.now().isoformat()
        }
        self._save_performance_data()
        logger.info(f"记录模型性能: {version_id}")
    
    def get_performance_history(self, version_id: str) -> List[Dict[str, Any]]:
        """获取性能历史"""
        return self.performance_data['models'].get(version_id, {}).get('history', [])


class ModelVersionManager:
    """
    模型版本管理器
    支持模型版本控制、回滚、对比
    """
    
    def __init__(self, model_repo_path: str = "models"):
        self.version_control = GitVersionControl(model_repo_path)
        self.model_registry = ModelRegistry()
        self.performance_tracker = PerformanceTracker()
        
        logger.info("模型版本管理器初始化完成")
    
    async def create_version(self, model, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """创建模型版本"""
        try:
            # 创建模型元数据
            model_metadata = ModelMetadata(
                version_id="",  # 将由版本控制生成
                algorithm_type=AlgorithmType(metadata.get('algorithm_type')),
                model_path="",
                parameters_path=metadata.get('parameters_path', ''),
                created_at=datetime.now(),
                created_by=metadata.get('created_by', 'system'),
                description=metadata.get('description', ''),
                performance_metrics=metadata.get('performance_metrics', {}),
                training_config=metadata.get('training_config', {}),
                status=ModelStatus.ACTIVE,
                tags=metadata.get('tags', []),
                dependencies=metadata.get('dependencies', {})
            )
            
            # 创建版本
            version_id = self.version_control.create_version(model, model_metadata)
            
            # 注册模型
            await self.model_registry.register_model(version_id, asdict(model_metadata))
            
            # 评估模型性能
            performance = await self.performance_tracker.evaluate(model)
            await self.performance_tracker.record_performance(version_id, performance)
            
            return {
                'version_id': version_id,
                'metadata': asdict(model_metadata),
                'performance': performance,
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"创建模型版本失败: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    async def rollback_version(self, target_version: str) -> Dict[str, Any]:
        """回滚到指定版本"""
        try:
            success = await self.version_control.rollback(target_version)
            
            if success:
                return {
                    'status': 'success',
                    'message': f'成功回滚到版本: {target_version}',
                    'target_version': target_version
                }
            else:
                return {
                    'status': 'error',
                    'message': f'回滚失败: {target_version}'
                }
                
        except Exception as e:
            logger.error(f"回滚版本失败: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    async def compare_versions(self, version1: str, version2: str) -> Dict[str, Any]:
        """比较两个版本"""
        try:
            # 获取版本信息
            info1 = self.version_control.get_version_info(version1)
            info2 = self.version_control.get_version_info(version2)
            
            if not info1 or not info2:
                return {
                    'status': 'error',
                    'message': '版本信息不存在'
                }
            
            # 比较性能指标
            perf1 = info1.get('performance_metrics', {})
            perf2 = info2.get('performance_metrics', {})
            
            comparison = {
                'version1': {
                    'version_id': version1,
                    'created_at': info1.get('created_at'),
                    'performance': perf1
                },
                'version2': {
                    'version_id': version2,
                    'created_at': info2.get('created_at'),
                    'performance': perf2
                },
                'differences': {
                    'accuracy_diff': perf2.get('accuracy', 0) - perf1.get('accuracy', 0),
                    'precision_diff': perf2.get('precision', 0) - perf1.get('precision', 0),
                    'recall_diff': perf2.get('recall', 0) - perf1.get('recall', 0),
                    'f1_diff': perf2.get('f1_score', 0) - perf1.get('f1_score', 0)
                }
            }
            
            return {
                'status': 'success',
                'comparison': comparison
            }
            
        except Exception as e:
            logger.error(f"版本比较失败: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    async def list_versions(self) -> List[Dict[str, Any]]:
        """列出所有版本"""
        try:
            versions = self.version_control.list_versions()
            
            # 添加性能信息
            for version in versions:
                version_id = version['version_id']
                performance = self.performance_tracker.get_performance_history(version_id)
                if performance:
                    version['latest_performance'] = performance[-1]
            
            return versions
            
        except Exception as e:
            logger.error(f"列出版本失败: {e}")
            return []
    
    async def get_version_details(self, version_id: str) -> Dict[str, Any]:
        """获取版本详细信息"""
        try:
            # 获取版本信息
            version_info = self.version_control.get_version_info(version_id)
            
            if not version_info:
                return {
                    'status': 'error',
                    'message': '版本不存在'
                }
            
            # 获取性能历史
            performance_history = self.performance_tracker.get_performance_history(version_id)
            
            return {
                'status': 'success',
                'version_info': version_info,
                'performance_history': performance_history
            }
            
        except Exception as e:
            logger.error(f"获取版本详情失败: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    async def update_version_status(self, version_id: str, status: ModelStatus) -> Dict[str, Any]:
        """更新版本状态"""
        try:
            version_info = self.version_control.get_version_info(version_id)
            
            if not version_info:
                return {
                    'status': 'error',
                    'message': '版本不存在'
                }
            
            # 更新状态
            version_info['status'] = status.value
            
            # 保存更新
            metadata_path = os.path.join(self.version_control.model_repo_path, version_id, "metadata.json")
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(version_info, f, indent=2, ensure_ascii=False)
            
            return {
                'status': 'success',
                'message': f'版本状态已更新为: {status.value}'
            }
            
        except Exception as e:
            logger.error(f"更新版本状态失败: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    async def delete_version(self, version_id: str) -> Dict[str, Any]:
        """删除版本"""
        try:
            # 检查版本是否存在
            if not self.version_control._version_exists(version_id):
                return {
                    'status': 'error',
                    'message': '版本不存在'
                }
            
            # 删除版本目录
            version_dir = os.path.join(self.version_control.model_repo_path, version_id)
            shutil.rmtree(version_dir)
            
            # 从注册表中移除
            await self.model_registry.unregister_model(version_id)
            
            return {
                'status': 'success',
                'message': f'版本已删除: {version_id}'
            }
            
        except Exception as e:
            logger.error(f"删除版本失败: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    async def export_version(self, version_id: str, export_path: str) -> Dict[str, Any]:
        """导出版本"""
        try:
            version_dir = os.path.join(self.version_control.model_repo_path, version_id)
            
            if not os.path.exists(version_dir):
                return {
                    'status': 'error',
                    'message': '版本不存在'
                }
            
            # 创建导出目录
            os.makedirs(export_path, exist_ok=True)
            
            # 复制版本文件
            export_version_dir = os.path.join(export_path, version_id)
            shutil.copytree(version_dir, export_version_dir)
            
            return {
                'status': 'success',
                'message': f'版本已导出到: {export_version_dir}',
                'export_path': export_version_dir
            }
            
        except Exception as e:
            logger.error(f"导出版本失败: {e}")
            return {
                'status': 'error',
                'error': str(e)
            } 
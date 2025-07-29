"""
训练集版本管理器
提供训练集版本控制、质量评估和版本选择功能
"""
import logging
import os
import json
import hashlib
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dataclasses import dataclass
import asyncio

from shared.config import config

logger = logging.getLogger(__name__)


@dataclass
class TrainingSetVersion:
    """训练集版本"""
    version_id: str
    name: str
    user_id: int
    doris_query_config: Dict[str, Any]
    feast_config: Dict[str, Any]
    quality_score: Optional[float] = None
    status: str = "created"
    data_path: Optional[str] = None
    metadata: Dict[str, Any] = None
    created_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()
        if self.metadata is None:
            self.metadata = {}


@dataclass
class QualityCriteria:
    """质量评估标准"""
    data_completeness_threshold: float = 0.95
    data_consistency_threshold: float = 0.90
    feature_coverage_threshold: float = 0.85
    time_coverage_threshold: float = 0.80
    sample_size_minimum: int = 1000
    duplicate_ratio_threshold: float = 0.05


class QualityAssessor:
    """质量评估器"""
    
    def __init__(self, criteria: QualityCriteria = None):
        self.criteria = criteria or QualityCriteria()
    
    def assess_training_set(self, training_set_data: pd.DataFrame, 
                           metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        评估训练集质量
        
        Args:
            training_set_data: 训练集数据
            metadata: 元数据信息
            
        Returns:
            质量评估结果
        """
        try:
            assessment = {
                'overall_score': 0.0,
                'metrics': {},
                'issues': [],
                'recommendations': []
            }
            
            # 数据完整性评估
            completeness_score = self._assess_data_completeness(training_set_data)
            assessment['metrics']['completeness'] = completeness_score
            
            # 数据一致性评估
            consistency_score = self._assess_data_consistency(training_set_data)
            assessment['metrics']['consistency'] = consistency_score
            
            # 特征覆盖度评估
            feature_coverage_score = self._assess_feature_coverage(training_set_data)
            assessment['metrics']['feature_coverage'] = feature_coverage_score
            
            # 时间覆盖度评估
            time_coverage_score = self._assess_time_coverage(training_set_data, metadata)
            assessment['metrics']['time_coverage'] = time_coverage_score
            
            # 样本量评估
            sample_size_score = self._assess_sample_size(training_set_data)
            assessment['metrics']['sample_size'] = sample_size_score
            
            # 重复数据评估
            duplicate_score = self._assess_duplicate_ratio(training_set_data)
            assessment['metrics']['duplicate_ratio'] = duplicate_score
            
            # 计算总体评分
            weights = {
                'completeness': 0.25,
                'consistency': 0.20,
                'feature_coverage': 0.20,
                'time_coverage': 0.15,
                'sample_size': 0.15,
                'duplicate_ratio': 0.05
            }
            
            overall_score = sum(
                assessment['metrics'][metric] * weight
                for metric, weight in weights.items()
            )
            
            assessment['overall_score'] = round(overall_score, 3)
            
            # 生成问题和建议
            self._generate_issues_and_recommendations(assessment)
            
            return assessment
            
        except Exception as e:
            logger.error(f"质量评估失败: {e}")
            return {
                'overall_score': 0.0,
                'metrics': {},
                'issues': [f"质量评估失败: {str(e)}"],
                'recommendations': ["请检查数据格式和内容"]
            }
    
    def _assess_data_completeness(self, data: pd.DataFrame) -> float:
        """评估数据完整性"""
        try:
            # 计算非空值比例
            non_null_ratio = data.notna().mean().mean()
            return min(1.0, non_null_ratio / self.criteria.data_completeness_threshold)
        except Exception:
            return 0.0
    
    def _assess_data_consistency(self, data: pd.DataFrame) -> float:
        """评估数据一致性"""
        try:
            # 检查数据类型一致性
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) == 0:
                return 0.5  # 没有数值列，给中等分数
            
            # 检查数值范围合理性
            consistency_scores = []
            for col in numeric_columns:
                if data[col].notna().sum() > 0:
                    # 检查是否有异常值（超过3个标准差）
                    mean_val = data[col].mean()
                    std_val = data[col].std()
                    if std_val > 0:
                        outlier_ratio = ((data[col] - mean_val).abs() > 3 * std_val).mean()
                        consistency_scores.append(1.0 - outlier_ratio)
                    else:
                        consistency_scores.append(1.0)
            
            return np.mean(consistency_scores) if consistency_scores else 0.5
            
        except Exception:
            return 0.5
    
    def _assess_feature_coverage(self, data: pd.DataFrame) -> float:
        """评估特征覆盖度"""
        try:
            # 计算特征列的数量和覆盖度
            feature_columns = [col for col in data.columns if not col.startswith(('uuid', 'node_id', 'time', 'timestamp'))]
            
            if len(feature_columns) == 0:
                return 0.0
            
            # 计算每个特征的非空值比例
            feature_coverage = data[feature_columns].notna().mean()
            avg_coverage = feature_coverage.mean()
            
            return min(1.0, avg_coverage / self.criteria.feature_coverage_threshold)
            
        except Exception:
            return 0.0
    
    def _assess_time_coverage(self, data: pd.DataFrame, metadata: Dict[str, Any]) -> float:
        """评估时间覆盖度"""
        try:
            # 查找时间列
            time_columns = [col for col in data.columns if 'time' in col.lower()]
            
            if not time_columns:
                return 0.5  # 没有时间列，给中等分数
            
            time_col = time_columns[0]
            
            # 计算时间范围
            if pd.api.types.is_datetime64_any_dtype(data[time_col]):
                time_range = data[time_col].max() - data[time_col].min()
                expected_range = metadata.get('expected_time_range', timedelta(days=30))
                
                if isinstance(expected_range, str):
                    expected_range = pd.Timedelta(expected_range)
                
                coverage_ratio = min(1.0, time_range.total_seconds() / expected_range.total_seconds())
                return min(1.0, coverage_ratio / self.criteria.time_coverage_threshold)
            
            return 0.5
            
        except Exception:
            return 0.5
    
    def _assess_sample_size(self, data: pd.DataFrame) -> float:
        """评估样本量"""
        try:
            sample_size = len(data)
            if sample_size >= self.criteria.sample_size_minimum:
                return 1.0
            else:
                return sample_size / self.criteria.sample_size_minimum
        except Exception:
            return 0.0
    
    def _assess_duplicate_ratio(self, data: pd.DataFrame) -> float:
        """评估重复数据比例"""
        try:
            # 计算重复行比例
            duplicate_ratio = data.duplicated().mean()
            # 重复比例越低越好，所以用1减去比例
            return max(0.0, 1.0 - duplicate_ratio / self.criteria.duplicate_ratio_threshold)
        except Exception:
            return 0.5
    
    def _generate_issues_and_recommendations(self, assessment: Dict[str, Any]):
        """生成问题和建议"""
        metrics = assessment['metrics']
        issues = []
        recommendations = []
        
        # 数据完整性问题
        if metrics.get('completeness', 0) < 0.8:
            issues.append("数据完整性不足，存在较多缺失值")
            recommendations.append("检查数据源，确保数据完整性")
        
        # 数据一致性问题
        if metrics.get('consistency', 0) < 0.7:
            issues.append("数据一致性较差，可能存在异常值")
            recommendations.append("进行数据清洗，处理异常值")
        
        # 特征覆盖度问题
        if metrics.get('feature_coverage', 0) < 0.8:
            issues.append("特征覆盖度不足")
            recommendations.append("检查特征工程流程，确保特征完整性")
        
        # 样本量问题
        if metrics.get('sample_size', 0) < 0.8:
            issues.append("样本量不足")
            recommendations.append("增加数据收集，扩大样本规模")
        
        # 重复数据问题
        if metrics.get('duplicate_ratio', 0) < 0.8:
            issues.append("存在较多重复数据")
            recommendations.append("进行数据去重处理")
        
        assessment['issues'] = issues
        assessment['recommendations'] = recommendations


class LineageTracker:
    """数据血缘追踪器"""
    
    def __init__(self, lineage_file: str = "data_lineage.json"):
        self.lineage_file = lineage_file
        self.lineage_data = self._load_lineage_data()
    
    def _load_lineage_data(self) -> Dict[str, Any]:
        """加载血缘数据"""
        try:
            if os.path.exists(self.lineage_file):
                with open(self.lineage_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {'versions': {}, 'relationships': {}}
        except Exception as e:
            logger.error(f"加载血缘数据失败: {e}")
            return {'versions': {}, 'relationships': {}}
    
    def _save_lineage_data(self):
        """保存血缘数据"""
        try:
            with open(self.lineage_file, 'w', encoding='utf-8') as f:
                json.dump(self.lineage_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"保存血缘数据失败: {e}")
    
    def track_version(self, version: TrainingSetVersion, parent_versions: List[str] = None):
        """追踪版本血缘"""
        try:
            version_info = {
                'version_id': version.version_id,
                'name': version.name,
                'user_id': version.user_id,
                'created_at': version.created_at.isoformat(),
                'quality_score': version.quality_score,
                'doris_query_config': version.doris_query_config,
                'feast_config': version.feast_config,
                'parent_versions': parent_versions or []
            }
            
            self.lineage_data['versions'][version.version_id] = version_info
            
            # 记录版本关系
            if parent_versions:
                for parent_id in parent_versions:
                    if parent_id not in self.lineage_data['relationships']:
                        self.lineage_data['relationships'][parent_id] = []
                    self.lineage_data['relationships'][parent_id].append(version.version_id)
            
            self._save_lineage_data()
            logger.info(f"版本血缘追踪成功: {version.version_id}")
            
        except Exception as e:
            logger.error(f"版本血缘追踪失败: {e}")
    
    def get_version_lineage(self, version_id: str) -> Dict[str, Any]:
        """获取版本血缘信息"""
        try:
            lineage = {
                'version_id': version_id,
                'ancestors': [],
                'descendants': [],
                'siblings': []
            }
            
            # 获取祖先版本
            version_info = self.lineage_data['versions'].get(version_id)
            if version_info:
                lineage['ancestors'] = self._get_ancestors(version_id)
                lineage['descendants'] = self.lineage_data['relationships'].get(version_id, [])
                lineage['siblings'] = self._get_siblings(version_id)
            
            return lineage
            
        except Exception as e:
            logger.error(f"获取版本血缘失败: {e}")
            return {}
    
    def _get_ancestors(self, version_id: str) -> List[str]:
        """获取祖先版本"""
        ancestors = []
        visited = set()
        
        def dfs(current_id):
            if current_id in visited:
                return
            visited.add(current_id)
            
            version_info = self.lineage_data['versions'].get(current_id)
            if version_info:
                for parent_id in version_info.get('parent_versions', []):
                    if parent_id not in ancestors:
                        ancestors.append(parent_id)
                    dfs(parent_id)
        
        dfs(version_id)
        return ancestors
    
    def _get_siblings(self, version_id: str) -> List[str]:
        """获取兄弟版本"""
        siblings = []
        version_info = self.lineage_data['versions'].get(version_id)
        if version_info:
            parent_versions = version_info.get('parent_versions', [])
            for parent_id in parent_versions:
                children = self.lineage_data['relationships'].get(parent_id, [])
                for child_id in children:
                    if child_id != version_id and child_id not in siblings:
                        siblings.append(child_id)
        return siblings


class TrainingSetVersionManager:
    """训练集版本管理器"""
    
    def __init__(self, storage_path: str = "training_set_versions"):
        self.storage_path = storage_path
        self.quality_assessor = QualityAssessor()
        self.lineage_tracker = LineageTracker()
        
        # 确保存储目录存在
        os.makedirs(self.storage_path, exist_ok=True)
        
        logger.info(f"训练集版本管理器初始化成功: {storage_path}")
    
    def _generate_version_id(self, name: str, user_id: int) -> str:
        """生成版本ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        content = f"{name}_{user_id}_{timestamp}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    async def create_version(self, training_set_config: Dict[str, Any], 
                           user_id: int, parent_versions: List[str] = None) -> TrainingSetVersion:
        """
        创建训练集版本
        
        Args:
            training_set_config: 训练集配置
            user_id: 用户ID
            parent_versions: 父版本列表
            
        Returns:
            训练集版本
        """
        try:
            # 生成版本ID
            version_id = self._generate_version_id(
                training_set_config['name'], user_id
            )
            
            # 创建版本对象
            version = TrainingSetVersion(
                version_id=version_id,
                name=training_set_config['name'],
                user_id=user_id,
                doris_query_config=training_set_config.get('doris_query_config', {}),
                feast_config=training_set_config.get('feast_config', {}),
                metadata=training_set_config.get('metadata', {})
            )
            
            # 保存版本信息
            await self._save_version(version)
            
            # 追踪血缘关系
            self.lineage_tracker.track_version(version, parent_versions)
            
            logger.info(f"训练集版本创建成功: {version_id}")
            return version
            
        except Exception as e:
            logger.error(f"创建训练集版本失败: {e}")
            raise
    
    async def _save_version(self, version: TrainingSetVersion):
        """保存版本信息"""
        try:
            version_file = os.path.join(self.storage_path, f"{version.version_id}.json")
            
            version_data = {
                'version_id': version.version_id,
                'name': version.name,
                'user_id': version.user_id,
                'doris_query_config': version.doris_query_config,
                'feast_config': version.feast_config,
                'quality_score': version.quality_score,
                'status': version.status,
                'data_path': version.data_path,
                'metadata': version.metadata,
                'created_at': version.created_at.isoformat(),
                'updated_at': version.updated_at.isoformat()
            }
            
            with open(version_file, 'w', encoding='utf-8') as f:
                json.dump(version_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"保存版本信息失败: {e}")
            raise
    
    async def get_version(self, version_id: str) -> Optional[TrainingSetVersion]:
        """获取版本信息"""
        try:
            version_file = os.path.join(self.storage_path, f"{version_id}.json")
            
            if not os.path.exists(version_file):
                return None
            
            with open(version_file, 'r', encoding='utf-8') as f:
                version_data = json.load(f)
            
            version = TrainingSetVersion(
                version_id=version_data['version_id'],
                name=version_data['name'],
                user_id=version_data['user_id'],
                doris_query_config=version_data['doris_query_config'],
                feast_config=version_data['feast_config'],
                quality_score=version_data.get('quality_score'),
                status=version_data.get('status', 'created'),
                data_path=version_data.get('data_path'),
                metadata=version_data.get('metadata', {}),
                created_at=datetime.fromisoformat(version_data['created_at']),
                updated_at=datetime.fromisoformat(version_data['updated_at'])
            )
            
            return version
            
        except Exception as e:
            logger.error(f"获取版本信息失败: {e}")
            return None
    
    async def list_versions(self, user_id: Optional[int] = None, 
                          status: Optional[str] = None) -> List[TrainingSetVersion]:
        """列出版本"""
        try:
            versions = []
            
            for filename in os.listdir(self.storage_path):
                if filename.endswith('.json'):
                    version_id = filename.replace('.json', '')
                    version = await self.get_version(version_id)
                    
                    if version:
                        # 过滤条件
                        if user_id and version.user_id != user_id:
                            continue
                        if status and version.status != status:
                            continue
                        
                        versions.append(version)
            
            # 按创建时间排序
            versions.sort(key=lambda v: v.created_at, reverse=True)
            return versions
            
        except Exception as e:
            logger.error(f"列出版本失败: {e}")
            return []
    
    async def assess_version_quality(self, version_id: str, 
                                   training_set_data: pd.DataFrame) -> Dict[str, Any]:
        """
        评估版本质量
        
        Args:
            version_id: 版本ID
            training_set_data: 训练集数据
            
        Returns:
            质量评估结果
        """
        try:
            version = await self.get_version(version_id)
            if not version:
                raise ValueError(f"版本不存在: {version_id}")
            
            # 进行质量评估
            assessment = self.quality_assessor.assess_training_set(
                training_set_data, version.metadata
            )
            
            # 更新版本质量分数
            version.quality_score = assessment['overall_score']
            version.updated_at = datetime.now()
            await self._save_version(version)
            
            logger.info(f"版本质量评估完成: {version_id}, 分数: {version.quality_score}")
            
            return assessment
            
        except Exception as e:
            logger.error(f"版本质量评估失败: {e}")
            raise
    
    async def select_quality_version(self, criteria: Dict[str, Any]) -> Optional[TrainingSetVersion]:
        """
        选择优质版本
        
        Args:
            criteria: 选择标准
            
        Returns:
            选中的版本
        """
        try:
            min_quality_score = criteria.get('min_quality_score', 0.7)
            user_id = criteria.get('user_id')
            status = criteria.get('status', 'ready')
            
            # 获取符合条件的版本
            versions = await self.list_versions(user_id, status)
            
            # 过滤质量分数
            qualified_versions = [
                v for v in versions 
                if v.quality_score and v.quality_score >= min_quality_score
            ]
            
            if not qualified_versions:
                logger.warning(f"没有找到符合条件的版本: {criteria}")
                return None
            
            # 按质量分数排序，选择最高分的版本
            best_version = max(qualified_versions, key=lambda v: v.quality_score or 0)
            
            logger.info(f"选择优质版本: {best_version.version_id}, 分数: {best_version.quality_score}")
            
            return best_version
            
        except Exception as e:
            logger.error(f"选择优质版本失败: {e}")
            return None
    
    async def get_version_lineage(self, version_id: str) -> Dict[str, Any]:
        """获取版本血缘信息"""
        return self.lineage_tracker.get_version_lineage(version_id)
    
    async def delete_version(self, version_id: str) -> bool:
        """删除版本"""
        try:
            version_file = os.path.join(self.storage_path, f"{version_id}.json")
            
            if os.path.exists(version_file):
                os.remove(version_file)
                logger.info(f"版本删除成功: {version_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"删除版本失败: {e}")
            return False
    
    async def update_version_status(self, version_id: str, status: str) -> bool:
        """更新版本状态"""
        try:
            version = await self.get_version(version_id)
            if not version:
                return False
            
            version.status = status
            version.updated_at = datetime.now()
            await self._save_version(version)
            
            logger.info(f"版本状态更新成功: {version_id} -> {status}")
            return True
            
        except Exception as e:
            logger.error(f"更新版本状态失败: {e}")
            return False


# 全局训练集版本管理器实例
training_set_version_manager = TrainingSetVersionManager() 
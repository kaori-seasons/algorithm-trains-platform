"""
Feast特征工程管理器
提供特征视图和训练集管理功能
"""
import logging
import os
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
from feast import FeatureStore, FeatureView, Entity, Field, FeatureService
from feast.types import Float32, String, Int64
from feast.data_source import PushSource, RequestSource
from feast.on_demand_feature_view import on_demand_feature_view
from feast.value_type import ValueType

from shared.config import config

logger = logging.getLogger(__name__)


# 配置类
class FeatureViewConfig:
    """特征视图配置"""
    def __init__(self, name: str, entity_name: str, source, fields: List[Dict], 
                 ttl_days: int = 30, description: str = ""):
        self.name = name
        self.entity_name = entity_name
        self.source = source
        self.fields = fields
        self.ttl_days = ttl_days
        self.description = description


class TrainingSetConfig:
    """训练集配置"""
    def __init__(self, name: str, feature_views: List[str], entities: List[str],
                 features: List[str], description: str = ""):
        self.name = name
        self.feature_views = feature_views
        self.entities = entities
        self.features = features
        self.description = description


class FeastManager:
    """Feast特征工程管理器"""
    
    def __init__(self):
        self.store = None
        self.registry_path = config.feast.registry_path
        self.provider = config.feast.provider
        # 延迟初始化，不在模块加载时立即初始化
        self._initialized = False
    
    def _ensure_initialized(self):
        """确保已初始化"""
        if not self._initialized:
            self._initialize_store()
            self._initialized = True
    
    def _initialize_store(self):
        """初始化特征存储"""
        try:
            # 确保注册表目录存在
            os.makedirs(os.path.dirname(self.registry_path), exist_ok=True)
            
            # 创建特征存储 - 使用新版本API
            self.store = FeatureStore(
                repo_path=self.registry_path
            )
            logger.info(f"✅ Feast特征存储初始化成功: {self.registry_path}")
            
        except Exception as e:
            logger.error(f"❌ Feast特征存储初始化失败: {e}")
            # 不抛出异常，允许系统继续运行
            self.store = None
    
    def create_entity(self, name: str, join_keys: List[str], description: str = "") -> Entity:
        """创建实体"""
        try:
            self._ensure_initialized()
            if not self.store:
                raise Exception("Feast存储未初始化")
                
            entity = Entity(
                name=name,
                join_keys=join_keys,
                description=description
            )
            self.store.apply([entity])
            logger.info(f"✅ 实体创建成功: {name}")
            return entity
            
        except Exception as e:
            logger.error(f"❌ 实体创建失败: {e}")
            raise
    
    def create_feature_view(self, config: FeatureViewConfig) -> FeatureView:
        """创建特征视图"""
        try:
            self._ensure_initialized()
            if not self.store:
                raise Exception("Feast存储未初始化")
                
            # 创建特征视图
            feature_view = FeatureView(
                name=config.name,
                entities=[config.entity_name],
                ttl=timedelta(days=config.ttl_days),
                schema=[
                    Field(name=field.name, dtype=field.dtype)
                    for field in config.fields
                ],
                source=config.source,
                description=config.description
            )
            
            self.store.apply([feature_view])
            logger.info(f"✅ 特征视图创建成功: {config.name}")
            return feature_view
            
        except Exception as e:
            logger.error(f"❌ 特征视图创建失败: {e}")
            raise
    
    def get_feature_view(self, name: str) -> Optional[FeatureView]:
        """获取特征视图"""
        try:
            self._ensure_initialized()
            if not self.store:
                return None
                
            return self.store.get_feature_view(name)
        except Exception as e:
            logger.error(f"❌ 获取特征视图失败: {e}")
            return None
    
    def list_feature_views(self) -> List[FeatureView]:
        """列出所有特征视图"""
        try:
            self._ensure_initialized()
            if not self.store:
                return []
                
            return self.store.list_feature_views()
        except Exception as e:
            logger.error(f"❌ 列出特征视图失败: {e}")
            return []
    
    def delete_feature_view(self, name: str) -> bool:
        """删除特征视图"""
        try:
            self._ensure_initialized()
            if not self.store:
                return False
                
            self.store.delete_feature_view(name)
            logger.info(f"✅ 特征视图删除成功: {name}")
            return True
        except Exception as e:
            logger.error(f"❌ 特征视图删除失败: {e}")
            return False
    
    def get_online_features(self, feature_refs: List[str], entity_rows: List[Dict]) -> pd.DataFrame:
        """获取在线特征"""
        try:
            self._ensure_initialized()
            if not self.store:
                raise Exception("Feast存储未初始化")
                
            return self.store.get_online_features(
                features=feature_refs,
                entity_rows=entity_rows
            ).to_df()
        except Exception as e:
            logger.error(f"❌ 获取在线特征失败: {e}")
            raise
    
    def get_historical_features(self, feature_refs: List[str], entity_df: pd.DataFrame) -> pd.DataFrame:
        """获取历史特征"""
        try:
            self._ensure_initialized()
            if not self.store:
                raise Exception("Feast存储未初始化")
                
            return self.store.get_historical_features(
                features=feature_refs,
                entity_df=entity_df
            ).to_df()
        except Exception as e:
            logger.error(f"❌ 获取历史特征失败: {e}")
            raise
    
    def create_feature_service(self, name: str, feature_views: List[str], description: str = "") -> FeatureService:
        """创建特征服务"""
        try:
            self._ensure_initialized()
            if not self.store:
                raise Exception("Feast存储未初始化")
                
            feature_service = FeatureService(
                name=name,
                features=self.store.get_feature_service(name).features if self.store.get_feature_service(name) else [],
                description=description
            )
            self.store.apply([feature_service])
            logger.info(f"✅ 特征服务创建成功: {name}")
            return feature_service
            
        except Exception as e:
            logger.error(f"❌ 特征服务创建失败: {e}")
            raise
    
    def test_connection(self) -> bool:
        """测试连接"""
        try:
            self._ensure_initialized()
            return self.store is not None
        except Exception as e:
            logger.error(f"❌ Feast连接测试失败: {e}")
            return False


# 全局实例
feast_manager = FeastManager() 
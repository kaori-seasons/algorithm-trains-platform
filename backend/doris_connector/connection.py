"""
Doris数据库连接管理器
提供Doris数据库的连接、查询和数据处理功能
"""
import asyncio
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
import pymysql
from pymysql.cursors import DictCursor
from contextlib import asynccontextmanager
import json
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

from shared.config import config

logger = logging.getLogger(__name__)


@dataclass
class DorisConnectionConfig:
    """Doris连接配置"""
    host: str = config.doris.host
    port: int = config.doris.port
    user: str = config.doris.username
    password: str = config.doris.password
    database: str = config.doris.database
    charset: str = config.doris.charset
    autocommit: bool = True
    max_connections: int = config.doris.max_connections
    connection_timeout: int = config.doris.connection_timeout
    read_timeout: int = config.doris.read_timeout


class DorisConnectionPool:
    """Doris连接池管理器"""
    
    def __init__(self, config: DorisConnectionConfig):
        self.config = config
        self.pool = []
        self.max_connections = config.max_connections
        self._lock = asyncio.Lock()
        self._executor = ThreadPoolExecutor(max_workers=max(4, config.max_connections))
    
    async def get_connection(self):
        """获取数据库连接"""
        async with self._lock:
            if self.pool:
                return self.pool.pop()
            
            # 创建新连接
            loop = asyncio.get_event_loop()
            connection = await loop.run_in_executor(
                self._executor,
                self._create_connection
            )
            return connection
    
    def _create_connection(self):
        """创建数据库连接"""
        try:
            connection = pymysql.connect(
                host=self.config.host,
                port=self.config.port,
                user=self.config.user,
                password=self.config.password,
                database=self.config.database,
                charset=self.config.charset,
                autocommit=self.config.autocommit,
                connect_timeout=self.config.connection_timeout,
                read_timeout=self.config.read_timeout,
                cursorclass=DictCursor
            )
            logger.debug(f"创建Doris连接成功: {self.config.host}:{self.config.port}")
            return connection
        except Exception as e:
            logger.error(f"创建Doris连接失败: {e}")
            raise
    
    async def return_connection(self, connection):
        """归还连接到连接池"""
        async with self._lock:
            if len(self.pool) < self.max_connections:
                self.pool.append(connection)
            else:
                connection.close()
    
    async def close_all(self):
        """关闭所有连接"""
        async with self._lock:
            for conn in self.pool:
                conn.close()
            self.pool.clear()
            self._executor.shutdown(wait=True)


class DorisQueryBuilder:
    """Doris查询构建器"""
    
    def __init__(self):
        self.base_table = "feature_snapshots"  # 默认特征快照表
    
    def build_time_range_query(self, start_time: datetime, end_time: datetime, 
                              filters: Dict[str, Any] = None, 
                              limit: int = None) -> str:
        """
        构建时间范围查询SQL
        
        Args:
            start_time: 开始时间
            end_time: 结束时间
            filters: 过滤条件
            limit: 限制返回记录数
        """
        query_parts = [
            f"SELECT * FROM {self.base_table}",
            f"WHERE time >= '{start_time.strftime('%Y-%m-%d %H:%M:%S')}'",
            f"AND time <= '{end_time.strftime('%Y-%m-%d %H:%M:%S')}'"
        ]
        
        # 添加过滤条件
        if filters:
            for key, value in filters.items():
                if isinstance(value, (list, tuple)):
                    placeholders = ','.join(['%s'] * len(value))
                    query_parts.append(f"AND {key} IN ({placeholders})")
                else:
                    query_parts.append(f"AND {key} = %s")
        
        # 添加排序
        query_parts.append("ORDER BY time ASC")
        
        # 添加限制
        if limit:
            query_parts.append(f"LIMIT {limit}")
        
        return " ".join(query_parts)
    
    def build_aggregation_query(self, time_range: str, 
                               aggregation_fields: List[str],
                               group_by: List[str] = None) -> str:
        """
        构建聚合查询SQL
        
        Args:
            time_range: 时间范围（如 '1h', '1d', '7d'）
            aggregation_fields: 聚合字段列表
            group_by: 分组字段列表
        """
        query_parts = [
            f"SELECT DATE_TRUNC('{time_range}', time) as time_bucket"
        ]
        
        # 添加聚合字段
        for field in aggregation_fields:
            query_parts.append(f", AVG({field}) as {field}_avg")
            query_parts.append(f", MAX({field}) as {field}_max")
            query_parts.append(f", MIN({field}) as {field}_min")
        
        query_parts.append(f"FROM {self.base_table}")
        
        # 添加分组
        if group_by:
            query_parts.append(f"GROUP BY time_bucket, {', '.join(group_by)}")
        else:
            query_parts.append("GROUP BY time_bucket")
        
        query_parts.append("ORDER BY time_bucket ASC")
        
        return " ".join(query_parts)


class FeatureSnapshotParser:
    """特征快照解析器"""
    
    def __init__(self):
        self.feature_extractors = {
            'meanLf': self.extract_numeric_feature,
            'std': self.extract_numeric_feature,
            'peakPowers': self.extract_array_feature,
            'peakFreqs': self.extract_array_feature,
            'spectralCentroid': self.extract_numeric_feature,
            'spectralRolloff': self.extract_numeric_feature,
            'zeroCrossingRate': self.extract_numeric_feature,
            'rms': self.extract_numeric_feature,
            'mfcc': self.extract_array_feature,
            'chroma': self.extract_array_feature,
            'extend': self.extract_json_feature
        }
    
    def parse_feature_snapshots(self, raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        解析特征快照数据
        
        Args:
            raw_data: 原始数据列表
            
        Returns:
            解析后的特征数据列表
        """
        parsed_features = []
        
        for record in raw_data:
            try:
                parsed_snapshot = self.parse_single_snapshot(record)
                if parsed_snapshot:
                    parsed_features.append(parsed_snapshot)
            except Exception as e:
                logger.warning(f"解析特征快照失败: {e}, record: {record}")
                continue
        
        logger.info(f"成功解析 {len(parsed_features)} 个特征快照")
        return parsed_features
    
    def parse_single_snapshot(self, record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        解析单个特征快照
        
        Args:
            record: 单条记录
            
        Returns:
            解析后的特征快照
        """
        try:
            feature_data = record.get('feature', {})
            
            parsed_snapshot = {
                'uuid': feature_data.get('uuid'),
                'nodeId': feature_data.get('nodeId'),
                'time': feature_data.get('time'),
                'is_tag': feature_data.get('is_tag', False),
                'features': {}
            }
            
            # 解析各个特征字段
            for feature_name, feature_value in feature_data.items():
                if feature_name in self.feature_extractors:
                    try:
                        parsed_value = self.feature_extractors[feature_name](feature_value)
                        parsed_snapshot['features'][feature_name] = parsed_value
                    except Exception as e:
                        logger.warning(f"解析特征 {feature_name} 失败: {e}")
                        continue
            
            return parsed_snapshot
            
        except Exception as e:
            logger.error(f"解析特征快照失败: {e}")
            return None
    
    def extract_numeric_feature(self, value) -> float:
        """提取数值特征"""
        if value is None:
            return 0.0
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0
    
    def extract_array_feature(self, value) -> List[float]:
        """提取数组特征"""
        if value is None:
            return []
        try:
            if isinstance(value, str):
                # 尝试解析JSON字符串
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return [float(x) for x in parsed if x is not None]
                else:
                    return [float(parsed)]
            elif isinstance(value, list):
                return [float(x) for x in value if x is not None]
            else:
                return [float(value)]
        except (ValueError, TypeError, json.JSONDecodeError):
            return []
    
    def extract_json_feature(self, value) -> Dict[str, Any]:
        """提取JSON特征"""
        if value is None:
            return {}
        try:
            if isinstance(value, str):
                return json.loads(value)
            elif isinstance(value, dict):
                return value
            else:
                return {}
        except (json.JSONDecodeError, TypeError):
            return {}


class DorisConnectionManager:
    """Doris数据库连接管理器"""
    
    def __init__(self, connection_config: DorisConnectionConfig = None):
        self.config = connection_config or DorisConnectionConfig()
        self.connection_pool = DorisConnectionPool(self.config)
        self.query_builder = DorisQueryBuilder()
        self.data_parser = FeatureSnapshotParser()
        self._cache = {}
    
    async def query_features_by_time_range(self, start_time: datetime, end_time: datetime, 
                                          filters: Dict[str, Any] = None,
                                          limit: int = None) -> List[Dict[str, Any]]:
        """
        根据时间范围查询特征数据
        
        Args:
            start_time: 开始时间
            end_time: 结束时间
            filters: 过滤条件
            limit: 限制返回记录数
            
        Returns:
            特征数据列表
        """
        try:
            # 构建查询SQL
            query = self.query_builder.build_time_range_query(
                start_time, end_time, filters, limit
            )
            
            # 执行查询
            raw_data = await self.execute_query(query, filters)
            
            # 解析特征数据
            parsed_features = self.data_parser.parse_feature_snapshots(raw_data)
            
            logger.info(f"查询特征数据成功: {len(parsed_features)} 条记录")
            return parsed_features
            
        except Exception as e:
            logger.error(f"查询特征数据失败: {e}")
            raise
    
    async def query_aggregated_features(self, time_range: str, 
                                       aggregation_fields: List[str],
                                       group_by: List[str] = None,
                                       start_time: datetime = None,
                                       end_time: datetime = None) -> List[Dict[str, Any]]:
        """
        查询聚合特征数据
        
        Args:
            time_range: 时间范围（如 '1h', '1d', '7d'）
            aggregation_fields: 聚合字段列表
            group_by: 分组字段列表
            start_time: 开始时间
            end_time: 结束时间
            
        Returns:
            聚合特征数据列表
        """
        try:
            # 构建聚合查询SQL
            query = self.query_builder.build_aggregation_query(
                time_range, aggregation_fields, group_by
            )
            
            # 添加时间范围过滤
            if start_time and end_time:
                query += f" WHERE time >= '{start_time.strftime('%Y-%m-%d %H:%M:%S')}'"
                query += f" AND time <= '{end_time.strftime('%Y-%m-%d %H:%M:%S')}'"
            
            # 执行查询
            result = await self.execute_query(query)
            
            logger.info(f"查询聚合特征数据成功: {len(result)} 条记录")
            return result
            
        except Exception as e:
            logger.error(f"查询聚合特征数据失败: {e}")
            raise
    
    async def execute_query(self, query: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        执行SQL查询
        
        Args:
            query: SQL查询语句
            params: 查询参数
            
        Returns:
            查询结果列表
        """
        connection = None
        try:
            connection = await self.connection_pool.get_connection()
            
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self._execute_query_sync,
                connection,
                query,
                params
            )
            
            return result
            
        except Exception as e:
            logger.error(f"执行查询失败: {e}")
            raise
        finally:
            if connection:
                await self.connection_pool.return_connection(connection)
    
    def _execute_query_sync(self, connection, query: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """同步执行查询"""
        try:
            with connection.cursor() as cursor:
                if params:
                    # 处理IN查询的参数
                    if isinstance(params, dict):
                        param_values = []
                        for key, value in params.items():
                            if isinstance(value, (list, tuple)):
                                param_values.extend(value)
                            else:
                                param_values.append(value)
                        cursor.execute(query, param_values)
                    else:
                        cursor.execute(query, params)
                else:
                    cursor.execute(query)
                
                result = cursor.fetchall()
                return result
                
        except Exception as e:
            logger.error(f"同步执行查询失败: {e}")
            raise
    
    async def get_table_schema(self, table_name: str) -> List[Dict[str, Any]]:
        """
        获取表结构信息
        
        Args:
            table_name: 表名
            
        Returns:
            表结构信息列表
        """
        query = f"DESCRIBE {table_name}"
        return await self.execute_query(query)
    
    async def get_table_statistics(self, table_name: str) -> Dict[str, Any]:
        """
        获取表统计信息
        
        Args:
            table_name: 表名
            
        Returns:
            表统计信息
        """
        try:
            # 获取表行数
            count_query = f"SELECT COUNT(*) as total_rows FROM {table_name}"
            count_result = await self.execute_query(count_query)
            total_rows = count_result[0]['total_rows'] if count_result else 0
            
            # 获取时间范围
            time_range_query = f"""
                SELECT 
                    MIN(time) as min_time,
                    MAX(time) as max_time
                FROM {table_name}
            """
            time_result = await self.execute_query(time_range_query)
            time_range = time_result[0] if time_result else {}
            
            return {
                'table_name': table_name,
                'total_rows': total_rows,
                'time_range': time_range
            }
            
        except Exception as e:
            logger.error(f"获取表统计信息失败: {e}")
            raise
    
    async def test_connection(self) -> bool:
        """测试数据库连接"""
        try:
            result = await self.execute_query("SELECT 1 as test")
            return len(result) > 0 and result[0]['test'] == 1
        except Exception as e:
            logger.error(f"测试数据库连接失败: {e}")
            return False
    
    async def close(self):
        """关闭连接管理器"""
        await self.connection_pool.close_all()


# 全局Doris连接管理器实例
doris_manager = DorisConnectionManager()

def get_doris_manager() -> DorisConnectionManager:
    """获取全局Doris连接管理器实例"""
    return doris_manager 
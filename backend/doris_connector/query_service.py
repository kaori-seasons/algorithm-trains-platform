"""
Doris查询服务
提供Doris数据库的查询和管理功能
"""
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import asyncio

from doris_connector.connection import DorisConnectionManager, DorisQueryBuilder, get_doris_manager
from feature_parser import FeatureSnapshotParser, ParsedFeature

logger = logging.getLogger(__name__)


class DorisQueryService:
    """Doris数据查询服务"""
    
    def __init__(self):
        self.parser = FeatureSnapshotParser()
    
    async def query_features_by_time_range(
        self,
        table_name: str,
        start_time: datetime,
        end_time: datetime,
        node_ids: Optional[List[str]] = None,
        limit: Optional[int] = None,
        additional_filters: Optional[Dict[str, Any]] = None
    ) -> List[ParsedFeature]:
        """
        根据时间范围查询特征数据
        
        Args:
            table_name: Doris表名
            start_time: 开始时间
            end_time: 结束时间
            node_ids: 节点ID列表，可选
            limit: 限制返回数量，可选
            additional_filters: 额外过滤条件，可选
            
        Returns:
            解析后的特征数据列表
        """
        try:
            # 获取Doris连接管理器
            doris_manager = get_doris_manager()
            
            # 构建查询
            query_builder = DorisQueryBuilder(table_name)
            query_builder.select("*")
            query_builder.where_time_range("time", start_time, end_time)
            
            # 添加节点ID过滤
            if node_ids:
                node_ids_str = "', '".join(node_ids)
                query_builder.where(f"nodeId IN ('{node_ids_str}')")
            
            # 添加额外过滤条件
            if additional_filters:
                for field, value in additional_filters.items():
                    if isinstance(value, (list, tuple)):
                        values_str = "', '".join(str(v) for v in value)
                        query_builder.where(f"{field} IN ('{values_str}')")
                    else:
                        query_builder.where(f"{field} = %({field})s", {field: value})
            
            # 按时间排序
            query_builder.order_by("time", "ASC")
            
            # 设置限制
            if limit:
                query_builder.limit(limit)
            
            # 构建SQL和参数
            sql, params = query_builder.build()
            
            logger.info(f"执行Doris查询: {sql}")
            logger.debug(f"查询参数: {params}")
            
            # 执行查询
            raw_data = await doris_manager.execute_query(sql, params)
            
            logger.info(f"查询返回 {len(raw_data)} 条原始记录")
            
            # 解析特征数据
            parsed_features = self.parser.parse_feature_snapshots(raw_data)
            
            # 验证数据质量
            valid_features = []
            for feature in parsed_features:
                if self.parser.validate_feature_snapshot(feature):
                    valid_features.append(feature)
                else:
                    logger.warning(f"特征快照验证失败: {feature.uuid}")
            
            logger.info(f"成功解析 {len(valid_features)} 个有效特征快照")
            
            return valid_features
            
        except Exception as e:
            logger.error(f"查询特征数据失败: {e}")
            raise
    
    async def query_features_by_config(
        self,
        query_config: Dict[str, Any]
    ) -> List[ParsedFeature]:
        """
        根据配置查询特征数据
        
        Args:
            query_config: 查询配置
                {
                    "table_name": "feature_table",
                    "time_range": {
                        "start_time": "2024-01-01T00:00:00",
                        "end_time": "2024-01-02T00:00:00"
                    },
                    "filters": {
                        "node_ids": ["node1", "node2"],
                        "is_tag": True,
                        "feature_filters": {
                            "meanLf": {"min": 1000, "max": 10000}
                        }
                    },
                    "limit": 1000
                }
            
        Returns:
            解析后的特征数据列表
        """
        # 解析配置
        table_name = query_config.get("table_name")
        if not table_name:
            raise ValueError("查询配置中缺少table_name")
        
        # 解析时间范围
        time_range = query_config.get("time_range", {})
        start_time_str = time_range.get("start_time")
        end_time_str = time_range.get("end_time")
        
        if not start_time_str or not end_time_str:
            raise ValueError("查询配置中缺少时间范围")
        
        try:
            start_time = datetime.fromisoformat(start_time_str.replace('Z', '+00:00'))
            end_time = datetime.fromisoformat(end_time_str.replace('Z', '+00:00'))
        except ValueError as e:
            raise ValueError(f"时间格式错误: {e}")
        
        # 解析过滤条件
        filters = query_config.get("filters", {})
        node_ids = filters.get("node_ids")
        additional_filters = {}
        
        # 处理标签过滤
        if "is_tag" in filters:
            additional_filters["is_tag"] = filters["is_tag"]
        
        # 处理其他过滤条件
        for key, value in filters.items():
            if key not in ["node_ids", "is_tag", "feature_filters"]:
                additional_filters[key] = value
        
        # 获取限制数量
        limit = query_config.get("limit")
        
        # 执行查询
        features = await self.query_features_by_time_range(
            table_name=table_name,
            start_time=start_time,
            end_time=end_time,
            node_ids=node_ids,
            limit=limit,
            additional_filters=additional_filters
        )
        
        # 应用特征过滤
        feature_filters = filters.get("feature_filters", {})
        if feature_filters:
            features = self.parser.filter_features(features, {"feature_filters": feature_filters})
        
        return features
    
    async def get_feature_statistics(
        self,
        table_name: str,
        start_time: datetime,
        end_time: datetime,
        node_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        获取特征数据统计信息
        
        Args:
            table_name: Doris表名
            start_time: 开始时间
            end_time: 结束时间
            node_ids: 节点ID列表，可选
            
        Returns:
            统计信息
        """
        try:
            # 查询数据
            features = await self.query_features_by_time_range(
                table_name=table_name,
                start_time=start_time,
                end_time=end_time,
                node_ids=node_ids,
                limit=10000  # 限制数量以获取统计信息
            )
            
            # 获取统计信息
            stats = self.parser.get_feature_statistics(features)
            
            # 添加查询条件信息
            stats.update({
                "query_conditions": {
                    "table_name": table_name,
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat(),
                    "node_ids": node_ids
                }
            })
            
            return stats
            
        except Exception as e:
            logger.error(f"获取特征统计信息失败: {e}")
            raise
    
    async def get_available_tables(self) -> List[str]:
        """
        获取可用的表列表
        
        Returns:
            表名列表
        """
        try:
            doris_manager = await get_doris_manager()
            
            # 查询系统表获取表列表
            sql = "SHOW TABLES"
            result = await doris_manager.execute_query(sql)
            
            tables = [row["Tables_in_current_database"] for row in result]
            logger.info(f"获取到 {len(tables)} 个可用表")
            
            return tables
            
        except Exception as e:
            logger.error(f"获取可用表列表失败: {e}")
            raise
    
    async def get_table_schema(self, table_name: str) -> Dict[str, Any]:
        """
        获取表结构信息
        
        Args:
            table_name: 表名
            
        Returns:
            表结构信息
        """
        try:
            doris_manager = await get_doris_manager()
            
            # 查询表结构
            sql = f"DESCRIBE {table_name}"
            result = await doris_manager.execute_query(sql)
            
            schema = {
                "table_name": table_name,
                "columns": []
            }
            
            for row in result:
                column_info = {
                    "field": row["Field"],
                    "type": row["Type"],
                    "null": row["Null"],
                    "key": row["Key"],
                    "default": row["Default"],
                    "extra": row["Extra"]
                }
                schema["columns"].append(column_info)
            
            logger.info(f"获取表 {table_name} 结构信息，包含 {len(schema['columns'])} 个字段")
            
            return schema
            
        except Exception as e:
            logger.error(f"获取表结构信息失败: {e}")
            raise
    
    async def test_connection(self) -> bool:
        """
        测试Doris连接
        
        Returns:
            连接是否正常
        """
        try:
            doris_manager = await get_doris_manager()
            
            # 执行简单查询测试连接
            sql = "SELECT 1 as test"
            result = await doris_manager.execute_query(sql)
            
            if result and len(result) > 0:
                logger.info("Doris连接测试成功")
                return True
            else:
                logger.error("Doris连接测试失败：查询返回空结果")
                return False
                
        except Exception as e:
            logger.error(f"Doris连接测试失败: {e}")
            return False
    
    async def get_data_quality_report(
        self,
        table_name: str,
        start_time: datetime,
        end_time: datetime
    ) -> Dict[str, Any]:
        """
        获取数据质量报告
        
        Args:
            table_name: 表名
            start_time: 开始时间
            end_time: 结束时间
            
        Returns:
            数据质量报告
        """
        try:
            # 查询原始数据
            features = await self.query_features_by_time_range(
                table_name=table_name,
                start_time=start_time,
                end_time=end_time,
                limit=10000
            )
            
            if not features:
                return {
                    "status": "no_data",
                    "message": "指定时间范围内没有数据"
                }
            
            # 生成质量报告
            report = {
                "status": "success",
                "total_records": len(features),
                "time_range": {
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat()
                },
                "quality_metrics": {
                    "completeness": 0.0,
                    "validity": 0.0,
                    "consistency": 0.0
                },
                "feature_analysis": {},
                "issues": []
            }
            
            # 计算完整性
            total_features = len(features)
            complete_features = sum(1 for f in features if f.uuid and f.node_id)
            report["quality_metrics"]["completeness"] = complete_features / total_features if total_features > 0 else 0.0
            
            # 计算有效性
            valid_features = sum(1 for f in features if self.parser.validate_feature_snapshot(f))
            report["quality_metrics"]["validity"] = valid_features / total_features if total_features > 0 else 0.0
            
            # 分析特征分布
            feature_counts = {}
            for feature in features:
                for feature_name in feature.features.keys():
                    feature_counts[feature_name] = feature_counts.get(feature_name, 0) + 1
            
            report["feature_analysis"] = {
                "feature_distribution": feature_counts,
                "unique_node_ids": len(set(f.node_id for f in features)),
                "tagged_ratio": sum(1 for f in features if f.is_tag) / total_features if total_features > 0 else 0.0
            }
            
            # 检查问题
            issues = []
            if report["quality_metrics"]["completeness"] < 0.9:
                issues.append("数据完整性较低")
            if report["quality_metrics"]["validity"] < 0.8:
                issues.append("数据有效性较低")
            
            report["issues"] = issues
            
            logger.info(f"生成数据质量报告: {report['status']}")
            
            return report
            
        except Exception as e:
            logger.error(f"生成数据质量报告失败: {e}")
            return {
                "status": "error",
                "message": str(e)
            } 
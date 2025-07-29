"""
Doris数据库路由
提供Doris相关API接口
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
from datetime import datetime

from shared.database import get_db
from doris_connector.connection import doris_manager

router = APIRouter()


@router.get("/health")
async def check_doris_health():
    """检查Doris连接健康状态"""
    try:
        is_healthy = await doris_manager.test_connection()
        return {
            "status": "healthy" if is_healthy else "unhealthy",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query/features")
async def query_features_by_time_range(query_config: Dict[str, Any]):
    """根据时间范围查询特征数据"""
    try:
        start_time = datetime.fromisoformat(query_config["start_time"])
        end_time = datetime.fromisoformat(query_config["end_time"])
        filters = query_config.get("filters", {})
        limit = query_config.get("limit")
        
        features = await doris_manager.query_features_by_time_range(
            start_time, end_time, filters, limit
        )
        
        return {
            "features": features,
            "count": len(features),
            "time_range": {
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat()
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query/aggregated")
async def query_aggregated_features(query_config: Dict[str, Any]):
    """查询聚合特征数据"""
    try:
        time_range = query_config["time_range"]
        aggregation_fields = query_config["aggregation_fields"]
        group_by = query_config.get("group_by")
        start_time = datetime.fromisoformat(query_config["start_time"]) if query_config.get("start_time") else None
        end_time = datetime.fromisoformat(query_config["end_time"]) if query_config.get("end_time") else None
        
        result = await doris_manager.query_aggregated_features(
            time_range, aggregation_fields, group_by, start_time, end_time
        )
        
        return {
            "aggregated_features": result,
            "count": len(result),
            "time_range": time_range,
            "aggregation_fields": aggregation_fields
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tables/{table_name}/schema")
async def get_table_schema(table_name: str):
    """获取表结构信息"""
    try:
        schema = await doris_manager.get_table_schema(table_name)
        return {
            "table_name": table_name,
            "schema": schema
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tables/{table_name}/statistics")
async def get_table_statistics(table_name: str):
    """获取表统计信息"""
    try:
        statistics = await doris_manager.get_table_statistics(table_name)
        return statistics
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query/custom")
async def execute_custom_query(query_config: Dict[str, Any]):
    """执行自定义查询"""
    try:
        query = query_config["query"]
        params = query_config.get("params", {})
        
        result = await doris_manager.execute_query(query, params)
        
        return {
            "result": result,
            "count": len(result),
            "query": query
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/databases")
async def list_databases():
    """列出所有数据库"""
    try:
        # 这里需要实际的数据库列表查询
        # 暂时返回模拟数据
        databases = [
            {"name": "default", "tables": ["feature_snapshots", "sensor_data"]},
            {"name": "analytics", "tables": ["user_behavior", "system_metrics"]}
        ]
        
        return {
            "databases": databases,
            "count": len(databases)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/databases/{database_name}/tables")
async def list_tables(database_name: str):
    """列出数据库中的所有表"""
    try:
        # 这里需要实际的表列表查询
        # 暂时返回模拟数据
        tables = {
            "default": ["feature_snapshots", "sensor_data", "device_info"],
            "analytics": ["user_behavior", "system_metrics", "performance_logs"]
        }
        
        table_list = tables.get(database_name, [])
        
        return {
            "database": database_name,
            "tables": table_list,
            "count": len(table_list)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sample-data/{table_name}")
async def get_sample_data(table_name: str, limit: int = 10):
    """获取表样本数据"""
    try:
        query = f"SELECT * FROM {table_name} LIMIT {limit}"
        result = await doris_manager.execute_query(query)
        
        return {
            "table_name": table_name,
            "sample_data": result,
            "limit": limit,
            "count": len(result)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 
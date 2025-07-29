"""
Doris连接器测试
"""
import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

from backend.doris_connector.connection import (
    DorisConnectionConfig,
    DorisConnectionManager,
    DorisQueryBuilder
)
from backend.doris_connector.feature_parser import (
    FeatureSnapshotParser,
    ParsedFeature
)
from backend.doris_connector.query_service import DorisQueryService


class TestDorisConnectionConfig:
    """测试Doris连接配置"""
    
    def test_connection_config_defaults(self):
        """测试连接配置默认值"""
        config = DorisConnectionConfig(host="localhost")
        
        assert config.host == "localhost"
        assert config.port == 9030
        assert config.user == "root"
        assert config.password == ""
        assert config.database == ""
        assert config.charset == "utf8"
        assert config.autocommit is True
        assert config.max_connections == 10
    
    def test_connection_config_custom(self):
        """测试自定义连接配置"""
        config = DorisConnectionConfig(
            host="doris.example.com",
            port=9031,
            user="test_user",
            password="test_password",
            database="test_db",
            max_connections=20
        )
        
        assert config.host == "doris.example.com"
        assert config.port == 9031
        assert config.user == "test_user"
        assert config.password == "test_password"
        assert config.database == "test_db"
        assert config.max_connections == 20


class TestDorisQueryBuilder:
    """测试Doris查询构建器"""
    
    def test_query_builder_basic(self):
        """测试基本查询构建"""
        builder = DorisQueryBuilder("test_table")
        sql, params = builder.build()
        
        assert sql == "SELECT * FROM test_table"
        assert params == {}
    
    def test_query_builder_with_where(self):
        """测试带WHERE条件的查询构建"""
        builder = DorisQueryBuilder("test_table")
        builder.where("status = %(status)s", {"status": "active"})
        sql, params = builder.build()
        
        assert "WHERE status = %(status)s" in sql
        assert params["status"] == "active"
    
    def test_query_builder_with_time_range(self):
        """测试时间范围查询构建"""
        builder = DorisQueryBuilder("test_table")
        start_time = datetime(2024, 1, 1, 0, 0, 0)
        end_time = datetime(2024, 1, 2, 0, 0, 0)
        builder.where_time_range("time", start_time, end_time)
        sql, params = builder.build()
        
        assert "WHERE time >= %(start_time)s AND time <= %(end_time)s" in sql
        assert params["start_time"] == "2024-01-01 00:00:00"
        assert params["end_time"] == "2024-01-02 00:00:00"
    
    def test_query_builder_with_order_and_limit(self):
        """测试带排序和限制的查询构建"""
        builder = DorisQueryBuilder("test_table")
        builder.order_by("time", "DESC")
        builder.limit(100)
        builder.offset(50)
        sql, params = builder.build()
        
        assert "ORDER BY time DESC" in sql
        assert "LIMIT 100 OFFSET 50" in sql


class TestFeatureSnapshotParser:
    """测试特征快照解析器"""
    
    def setup_method(self):
        """设置测试环境"""
        self.parser = FeatureSnapshotParser()
    
    def test_parse_numeric_feature(self):
        """测试数值特征解析"""
        result = self.parser.extract_numeric_feature("123.45")
        assert result == 123.45
        
        result = self.parser.extract_numeric_feature("")
        assert result is None
        
        result = self.parser.extract_numeric_feature(None)
        assert result is None
    
    def test_parse_array_feature(self):
        """测试数组特征解析"""
        result = self.parser.extract_array_feature("1.1,2.2,3.3")
        assert result == [1.1, 2.2, 3.3]
        
        result = self.parser.extract_array_feature("123.45")
        assert result == [123.45]
        
        result = self.parser.extract_array_feature([1, 2, 3])
        assert result == [1.0, 2.0, 3.0]
    
    def test_parse_json_feature(self):
        """测试JSON特征解析"""
        json_str = '{"key": "value", "number": 123}'
        result = self.parser.extract_json_feature(json_str)
        assert result == {"key": "value", "number": 123}
        
        json_dict = {"key": "value"}
        result = self.parser.extract_json_feature(json_dict)
        assert result == {"key": "value"}
    
    def test_parse_single_snapshot(self):
        """测试单个快照解析"""
        record = {
            "feature": {
                "uuid": "test-uuid",
                "nodeId": "test-node",
                "time": "1640995200000",  # 2022-01-01 00:00:00
                "is_tag": True,
                "meanLf": "123.45",
                "peakPowers": "1.1,2.2,3.3",
                "extend": '{"SerialData": "", "GpioData": -1}'
            }
        }
        
        result = self.parser.parse_single_snapshot(record)
        
        assert result is not None
        assert result.uuid == "test-uuid"
        assert result.node_id == "test-node"
        assert result.is_tag is True
        assert result.features["meanLf"] == 123.45
        assert result.features["peakPowers"] == [1.1, 2.2, 3.3]
        assert result.features["extend"] == {"SerialData": "", "GpioData": -1}
    
    def test_validate_feature_snapshot(self):
        """测试特征快照验证"""
        # 有效快照
        valid_feature = ParsedFeature(
            uuid="test-uuid",
            node_id="test-node",
            time=datetime.now(),
            is_tag=True,
            features={"meanLf": 123.45},
            raw_data={}
        )
        assert self.parser.validate_feature_snapshot(valid_feature) is True
        
        # 无效快照 - 缺少UUID
        invalid_feature = ParsedFeature(
            uuid="",
            node_id="test-node",
            time=datetime.now(),
            is_tag=True,
            features={"meanLf": 123.45},
            raw_data={}
        )
        assert self.parser.validate_feature_snapshot(invalid_feature) is False


class TestDorisQueryService:
    """测试Doris查询服务"""
    
    def setup_method(self):
        """设置测试环境"""
        self.service = DorisQueryService()
    
    @pytest.mark.asyncio
    async def test_query_features_by_time_range(self):
        """测试时间范围查询"""
        # Mock数据
        mock_features = [
            ParsedFeature(
                uuid="test-uuid-1",
                node_id="test-node",
                time=datetime.now(),
                is_tag=True,
                features={"meanLf": 123.45},
                raw_data={}
            ),
            ParsedFeature(
                uuid="test-uuid-2",
                node_id="test-node",
                time=datetime.now(),
                is_tag=False,
                features={"meanLf": 234.56},
                raw_data={}
            )
        ]
        
        # Mock解析器
        with patch.object(self.service.parser, 'parse_feature_snapshots', return_value=mock_features):
            with patch.object(self.service.parser, 'validate_feature_snapshot', return_value=True):
                # Mock Doris管理器
                with patch('backend.doris_connector.query_service.get_doris_manager') as mock_get_manager:
                    mock_manager = AsyncMock()
                    mock_manager.execute_query.return_value = [{"feature": {}}]
                    mock_get_manager.return_value = mock_manager
                    
                    # 执行测试
                    start_time = datetime(2024, 1, 1, 0, 0, 0)
                    end_time = datetime(2024, 1, 2, 0, 0, 0)
                    
                    result = await self.service.query_features_by_time_range(
                        table_name="test_table",
                        start_time=start_time,
                        end_time=end_time
                    )
                    
                    assert len(result) == 2
                    assert result[0].uuid == "test-uuid-1"
                    assert result[1].uuid == "test-uuid-2"
    
    @pytest.mark.asyncio
    async def test_get_feature_statistics(self):
        """测试获取特征统计信息"""
        # Mock数据
        mock_features = [
            ParsedFeature(
                uuid="test-uuid-1",
                node_id="node-1",
                time=datetime(2024, 1, 1, 0, 0, 0),
                is_tag=True,
                features={"meanLf": 123.45},
                raw_data={}
            ),
            ParsedFeature(
                uuid="test-uuid-2",
                node_id="node-2",
                time=datetime(2024, 1, 2, 0, 0, 0),
                is_tag=False,
                features={"meanLf": 234.56},
                raw_data={}
            )
        ]
        
        # Mock查询方法
        with patch.object(self.service, 'query_features_by_time_range', return_value=mock_features):
            start_time = datetime(2024, 1, 1, 0, 0, 0)
            end_time = datetime(2024, 1, 2, 0, 0, 0)
            
            result = await self.service.get_feature_statistics(
                table_name="test_table",
                start_time=start_time,
                end_time=end_time
            )
            
            assert result["total_count"] == 2
            assert result["tagged_count"] == 1
            assert "node-1" in result["node_ids"]
            assert "node-2" in result["node_ids"]
            assert len(result["node_ids"]) == 2


if __name__ == "__main__":
    pytest.main([__file__]) 
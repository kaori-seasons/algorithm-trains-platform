"""
特征快照解析器
解析Doris中的特征快照数据
"""
import json
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ParsedFeature:
    """解析后的特征数据"""
    uuid: str
    node_id: str
    time: datetime
    is_tag: bool
    features: Dict[str, Any]
    raw_data: Dict[str, Any]


class FeatureSnapshotParser:
    """特征快照解析器"""
    
    def __init__(self):
        # 定义特征提取器映射
        self.feature_extractors = {
            'meanLf': self.extract_numeric_feature,
            'std': self.extract_numeric_feature,
            'mean': self.extract_numeric_feature,
            'meanHf': self.extract_numeric_feature,
            'temperature': self.extract_numeric_feature,
            'peakPowers': self.extract_array_feature,
            'peakFreqs': self.extract_array_feature,
            'bandSpectrum': self.extract_array_feature,
            'feature1': self.extract_array_feature,
            'feature2': self.extract_array_feature,
            'feature3': self.extract_array_feature,
            'feature4': self.extract_array_feature,
            'extend': self.extract_json_feature,
            'customFeature': self.extract_string_feature,
            'status': self.extract_string_feature,
            'SerialData': self.extract_string_feature,
            'GpioData': self.extract_numeric_feature,
        }
    
    def parse_feature_snapshots(self, raw_data: List[Dict[str, Any]]) -> List[ParsedFeature]:
        """
        解析特征快照数据列表
        
        Args:
            raw_data: 原始数据列表，每个元素包含feature字段
            
        Returns:
            解析后的特征数据列表
        """
        parsed_features = []
        
        for record in raw_data:
            try:
                parsed_feature = self.parse_single_snapshot(record)
                if parsed_feature:
                    parsed_features.append(parsed_feature)
            except Exception as e:
                logger.warning(f"解析特征快照失败: {e}, 记录: {record}")
                continue
        
        logger.info(f"成功解析 {len(parsed_features)} 个特征快照")
        return parsed_features
    
    def parse_single_snapshot(self, record: Dict[str, Any]) -> Optional[ParsedFeature]:
        """
        解析单个特征快照
        
        Args:
            record: 包含feature字段的记录
            
        Returns:
            解析后的特征数据
        """
        if 'feature' not in record:
            logger.warning(f"记录中缺少feature字段: {record}")
            return None
        
        feature_data = record['feature']
        
        # 提取基础信息
        uuid = feature_data.get('uuid', '')
        node_id = feature_data.get('nodeId', '')
        time_str = feature_data.get('time', '')
        is_tag = feature_data.get('is_tag', False)
        
        # 解析时间
        try:
            if time_str:
                # 假设时间戳是毫秒级的
                timestamp = int(time_str) / 1000
                time = datetime.fromtimestamp(timestamp)
            else:
                time = datetime.now()
        except (ValueError, TypeError) as e:
            logger.warning(f"时间解析失败: {time_str}, 错误: {e}")
            time = datetime.now()
        
        # 解析特征
        features = {}
        for feature_name, feature_value in feature_data.items():
            if feature_name in ['uuid', 'nodeId', 'time', 'is_tag']:
                continue  # 跳过基础字段
            
            if feature_name in self.feature_extractors:
                try:
                    parsed_value = self.feature_extractors[feature_name](feature_value)
                    features[feature_name] = parsed_value
                except Exception as e:
                    logger.warning(f"特征 {feature_name} 解析失败: {e}")
                    features[feature_name] = None
            else:
                # 未知特征，尝试通用解析
                features[feature_name] = self.extract_generic_feature(feature_value)
        
        return ParsedFeature(
            uuid=uuid,
            node_id=node_id,
            time=time,
            is_tag=is_tag,
            features=features,
            raw_data=record
        )
    
    def extract_numeric_feature(self, value: Any) -> Optional[float]:
        """提取数值特征"""
        if value is None or value == '':
            return None
        
        try:
            return float(value)
        except (ValueError, TypeError):
            logger.warning(f"无法解析数值特征: {value}")
            return None
    
    def extract_array_feature(self, value: Any) -> Optional[List[float]]:
        """提取数组特征"""
        if value is None or value == '':
            return None
        
        try:
            if isinstance(value, str):
                # 处理逗号分隔的字符串
                if ',' in value:
                    return [float(x.strip()) for x in value.split(',') if x.strip()]
                else:
                    # 单个数值
                    return [float(value)]
            elif isinstance(value, (list, tuple)):
                return [float(x) for x in value]
            else:
                return [float(value)]
        except (ValueError, TypeError) as e:
            logger.warning(f"无法解析数组特征: {value}, 错误: {e}")
            return None
    
    def extract_json_feature(self, value: Any) -> Optional[Dict[str, Any]]:
        """提取JSON特征"""
        if value is None or value == '':
            return None
        
        try:
            if isinstance(value, str):
                return json.loads(value)
            elif isinstance(value, dict):
                return value
            else:
                logger.warning(f"JSON特征格式错误: {value}")
                return None
        except json.JSONDecodeError as e:
            logger.warning(f"JSON解析失败: {value}, 错误: {e}")
            return None
    
    def extract_string_feature(self, value: Any) -> Optional[str]:
        """提取字符串特征"""
        if value is None:
            return None
        
        return str(value)
    
    def extract_generic_feature(self, value: Any) -> Any:
        """通用特征提取"""
        if value is None:
            return None
        
        # 尝试数值解析
        try:
            return float(value)
        except (ValueError, TypeError):
            pass
        
        # 尝试JSON解析
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                pass
        
        # 返回原始值
        return value
    
    def validate_feature_snapshot(self, parsed_feature: ParsedFeature) -> bool:
        """
        验证解析后的特征快照
        
        Args:
            parsed_feature: 解析后的特征数据
            
        Returns:
            是否有效
        """
        # 检查必需字段
        if not parsed_feature.uuid:
            logger.warning("特征快照缺少UUID")
            return False
        
        if not parsed_feature.node_id:
            logger.warning("特征快照缺少节点ID")
            return False
        
        # 检查特征数量
        if not parsed_feature.features:
            logger.warning("特征快照没有有效特征")
            return False
        
        # 检查特征质量
        valid_features = 0
        for feature_name, feature_value in parsed_feature.features.items():
            if feature_value is not None:
                valid_features += 1
        
        if valid_features == 0:
            logger.warning("特征快照没有有效特征值")
            return False
        
        return True
    
    def get_feature_statistics(self, parsed_features: List[ParsedFeature]) -> Dict[str, Any]:
        """
        获取特征统计信息
        
        Args:
            parsed_features: 解析后的特征数据列表
            
        Returns:
            统计信息
        """
        if not parsed_features:
            return {}
        
        stats = {
            'total_count': len(parsed_features),
            'tagged_count': sum(1 for f in parsed_features if f.is_tag),
            'feature_counts': {},
            'node_ids': set(),
            'time_range': {
                'start': None,
                'end': None
            }
        }
        
        # 统计特征出现次数
        for feature in parsed_features:
            stats['node_ids'].add(feature.node_id)
            
            # 更新时间范围
            if stats['time_range']['start'] is None or feature.time < stats['time_range']['start']:
                stats['time_range']['start'] = feature.time
            if stats['time_range']['end'] is None or feature.time > stats['time_range']['end']:
                stats['time_range']['end'] = feature.time
            
            # 统计特征出现次数
            for feature_name in feature.features.keys():
                stats['feature_counts'][feature_name] = stats['feature_counts'].get(feature_name, 0) + 1
        
        # 转换集合为列表以便JSON序列化
        stats['node_ids'] = list(stats['node_ids'])
        
        # 转换时间格式
        if stats['time_range']['start']:
            stats['time_range']['start'] = stats['time_range']['start'].isoformat()
        if stats['time_range']['end']:
            stats['time_range']['end'] = stats['time_range']['end'].isoformat()
        
        return stats
    
    def filter_features(self, parsed_features: List[ParsedFeature], 
                       filters: Dict[str, Any]) -> List[ParsedFeature]:
        """
        过滤特征数据
        
        Args:
            parsed_features: 解析后的特征数据列表
            filters: 过滤条件
            
        Returns:
            过滤后的特征数据列表
        """
        filtered_features = []
        
        for feature in parsed_features:
            # 检查是否通过所有过滤条件
            passed = True
            
            # UUID过滤
            if 'uuid' in filters and feature.uuid != filters['uuid']:
                passed = False
            
            # 节点ID过滤
            if 'node_id' in filters and feature.node_id != filters['node_id']:
                passed = False
            
            # 时间范围过滤
            if 'start_time' in filters and feature.time < filters['start_time']:
                passed = False
            if 'end_time' in filters and feature.time > filters['end_time']:
                passed = False
            
            # 标签过滤
            if 'is_tag' in filters and feature.is_tag != filters['is_tag']:
                passed = False
            
            # 特征值过滤
            if 'feature_filters' in filters:
                for feature_name, filter_value in filters['feature_filters'].items():
                    if feature_name in feature.features:
                        feature_value = feature.features[feature_name]
                        if not self._check_feature_filter(feature_value, filter_value):
                            passed = False
                            break
            
            if passed:
                filtered_features.append(feature)
        
        return filtered_features
    
    def _check_feature_filter(self, feature_value: Any, filter_value: Any) -> bool:
        """检查特征值是否满足过滤条件"""
        if feature_value is None:
            return False
        
        if isinstance(filter_value, dict):
            # 范围过滤
            if 'min' in filter_value and feature_value < filter_value['min']:
                return False
            if 'max' in filter_value and feature_value > filter_value['max']:
                return False
            if 'equals' in filter_value and feature_value != filter_value['equals']:
                return False
        else:
            # 精确匹配
            if feature_value != filter_value:
                return False
        
        return True 
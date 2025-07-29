"""
成本优化器
用于评估各个挂载点的资源消耗并优化账单支出
"""
import os
import json
import logging
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import subprocess
import psutil
import kubernetes
from kubernetes import client, config
from kubernetes.client.rest import ApiException
import pandas as pd
import numpy as np
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StorageType(Enum):
    """存储类型枚举"""
    PVC = "pvc"
    HOST_PATH = "host_path"
    NFS = "nfs"
    S3 = "s3"
    MINIO = "minio"
    MEMORY = "memory"


class ResourceType(Enum):
    """资源类型枚举"""
    CPU = "cpu"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK = "network"


@dataclass
class ResourceUsage:
    """资源使用情况"""
    resource_type: ResourceType
    current_usage: float
    peak_usage: float
    average_usage: float
    unit: str
    timestamp: datetime


@dataclass
class StorageMount:
    """存储挂载点信息"""
    mount_path: str
    storage_type: StorageType
    total_size: float
    used_size: float
    available_size: float
    usage_percentage: float
    mount_time: datetime
    last_access: datetime
    access_pattern: Dict[str, Any]
    cost_per_gb_month: float


@dataclass
class CostAnalysis:
    """成本分析结果"""
    mount_path: str
    storage_type: StorageType
    monthly_cost: float
    cost_efficiency: float  # 成本效率 (使用率/成本)
    optimization_suggestions: List[str]
    potential_savings: float


class CostOptimizer:
    """
    成本优化器
    分析存储挂载点的资源消耗并提供优化建议
    """
    
    def __init__(self, kube_config_path: Optional[str] = None):
        """
        初始化成本优化器
        
        Args:
            kube_config_path: Kubernetes配置文件路径
        """
        self.kube_config_path = kube_config_path
        self.core_v1_api = None
        self.storage_v1_api = None
        self._init_kubernetes_client()
        
        # 成本配置
        self.cost_config = {
            StorageType.PVC: {
                "standard": 0.10,  # $0.10/GB/月
                "fast-ssd": 0.25,  # $0.25/GB/月
                "premium-ssd": 0.50,  # $0.50/GB/月
            },
            StorageType.HOST_PATH: {
                "default": 0.05,  # $0.05/GB/月 (本地存储)
            },
            StorageType.NFS: {
                "default": 0.08,  # $0.08/GB/月
            },
            StorageType.S3: {
                "standard": 0.023,  # $0.023/GB/月
                "infrequent_access": 0.0125,  # $0.0125/GB/月
                "glacier": 0.004,  # $0.004/GB/月
            },
            StorageType.MINIO: {
                "default": 0.05,  # $0.05/GB/月 (自建存储)
            },
            StorageType.MEMORY: {
                "default": 0.15,  # $0.15/GB/月 (内存存储)
            }
        }
        
        # 资源使用历史
        self.usage_history = {}
        
        # 优化建议模板
        self.optimization_templates = {
            "low_usage": "存储使用率低于20%，建议降级到更便宜的存储类型或合并存储",
            "high_cost": "存储成本过高，建议迁移到成本更低的存储类型",
            "unused_storage": "存储长期未使用，建议删除或归档",
            "over_provisioned": "存储容量过度配置，建议减少容量",
            "fragmented": "存储碎片化严重，建议整理和合并",
            "old_data": "数据长期未访问，建议迁移到冷存储"
        }
    
    def _init_kubernetes_client(self):
        """初始化Kubernetes客户端"""
        try:
            if self.kube_config_path and os.path.exists(self.kube_config_path):
                config.load_kube_config(config_file=self.kube_config_path)
            else:
                try:
                    config.load_incluster_config()
                except Exception:
                    try:
                        config.load_kube_config()
                    except Exception:
                        logger.warning("无法加载Kubernetes配置，PVC分析功能将不可用")
                        return
            
            self.core_v1_api = client.CoreV1Api()
            self.storage_v1_api = client.StorageV1Api()
        except Exception as e:
            logger.error(f"初始化Kubernetes客户端失败: {e}")
    
    async def analyze_all_mounts(self) -> List[StorageMount]:
        """
        分析所有存储挂载点
        
        Returns:
            存储挂载点列表
        """
        mounts = []
        
        # 分析系统挂载点
        system_mounts = await self._analyze_system_mounts()
        mounts.extend(system_mounts)
        
        # 分析Kubernetes PVC
        if self.core_v1_api:
            pvc_mounts = await self._analyze_pvc_mounts()
            mounts.extend(pvc_mounts)
        
        # 分析Docker卷
        docker_mounts = await self._analyze_docker_mounts()
        mounts.extend(docker_mounts)
        
        return mounts
    
    async def _analyze_system_mounts(self) -> List[StorageMount]:
        """分析系统挂载点"""
        mounts = []
        
        try:
            # 获取系统挂载信息
            result = subprocess.run(['df', '-h'], capture_output=True, text=True)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')[1:]  # 跳过标题行
                
                for line in lines:
                    parts = line.split()
                    if len(parts) >= 6:
                        filesystem, total_size, used_size, available_size, usage_percent, mount_path = parts[:6]
                        
                        # 解析大小
                        total_gb = self._parse_size_to_gb(total_size)
                        used_gb = self._parse_size_to_gb(used_size)
                        available_gb = self._parse_size_to_gb(available_size)
                        usage_percentage = float(usage_percent.rstrip('%'))
                        
                        # 确定存储类型
                        storage_type = self._determine_storage_type(filesystem, mount_path)
                        
                        # 获取访问模式
                        access_pattern = await self._analyze_access_pattern(mount_path)
                        
                        # 计算成本
                        cost_per_gb_month = self._get_cost_per_gb_month(storage_type, "default")
                        
                        mount = StorageMount(
                            mount_path=mount_path,
                            storage_type=storage_type,
                            total_size=total_gb,
                            used_size=used_gb,
                            available_size=available_gb,
                            usage_percentage=usage_percentage,
                            mount_time=datetime.now(),  # 简化处理
                            last_access=datetime.now(),  # 简化处理
                            access_pattern=access_pattern,
                            cost_per_gb_month=cost_per_gb_month
                        )
                        mounts.append(mount)
        
        except Exception as e:
            logger.error(f"分析系统挂载点失败: {e}")
        
        return mounts
    
    async def _analyze_pvc_mounts(self) -> List[StorageMount]:
        """分析Kubernetes PVC挂载点"""
        mounts = []
        
        try:
            # 获取所有命名空间的PVC
            namespaces = ['default', 'train-platform', 'kube-system']
            
            for namespace in namespaces:
                try:
                    pvcs = self.core_v1_api.list_namespaced_persistent_volume_claim(namespace)
                    
                    for pvc in pvcs.items:
                        # 获取PVC状态
                        pvc_status = pvc.status
                        if pvc_status.phase == 'Bound':
                            # 获取PV信息
                            pv_name = pvc_status.volume_name
                            try:
                                pv = self.core_v1_api.read_persistent_volume(pv_name)
                                
                                # 解析存储大小
                                capacity = pv.spec.capacity.get('storage', '0Gi')
                                total_gb = self._parse_size_to_gb(capacity)
                                
                                # 估算使用量（简化处理）
                                used_gb = total_gb * 0.6  # 假设60%使用率
                                available_gb = total_gb - used_gb
                                usage_percentage = (used_gb / total_gb) * 100
                                
                                # 确定存储类型和成本
                                storage_class = pv.spec.storage_class_name or 'standard'
                                cost_per_gb_month = self._get_cost_per_gb_month(StorageType.PVC, storage_class)
                                
                                # 构建挂载路径
                                mount_path = f"/pvc/{namespace}/{pvc.metadata.name}"
                                
                                mount = StorageMount(
                                    mount_path=mount_path,
                                    storage_type=StorageType.PVC,
                                    total_size=total_gb,
                                    used_size=used_gb,
                                    available_size=available_gb,
                                    usage_percentage=usage_percentage,
                                    mount_time=datetime.now(),
                                    last_access=datetime.now(),
                                    access_pattern={},
                                    cost_per_gb_month=cost_per_gb_month
                                )
                                mounts.append(mount)
                                
                            except ApiException as e:
                                logger.warning(f"无法获取PV {pv_name} 信息: {e}")
                
                except ApiException as e:
                    logger.warning(f"无法获取命名空间 {namespace} 的PVC: {e}")
        
        except Exception as e:
            logger.error(f"分析PVC挂载点失败: {e}")
        
        return mounts
    
    async def _analyze_docker_mounts(self) -> List[StorageMount]:
        """分析Docker卷挂载点"""
        mounts = []
        
        try:
            # 获取Docker卷信息
            result = subprocess.run(['docker', 'volume', 'ls', '--format', 'json'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if line:
                        try:
                            volume_info = json.loads(line)
                            volume_name = volume_info.get('Name', '')
                            
                            # 获取卷详细信息
                            inspect_result = subprocess.run(
                                ['docker', 'volume', 'inspect', volume_name],
                                capture_output=True, text=True
                            )
                            
                            if inspect_result.returncode == 0:
                                inspect_data = json.loads(inspect_result.stdout)
                                if inspect_data:
                                    volume_data = inspect_data[0]
                                    mountpoint = volume_data.get('Mountpoint', '')
                                    
                                    # 获取卷大小
                                    du_result = subprocess.run(
                                        ['du', '-sh', mountpoint],
                                        capture_output=True, text=True
                                    )
                                    
                                    if du_result.returncode == 0:
                                        size_str = du_result.stdout.strip().split()[0]
                                        used_gb = self._parse_size_to_gb(size_str)
                                        
                                        # 估算总大小（简化处理）
                                        total_gb = used_gb * 1.5  # 假设1.5倍使用量
                                        available_gb = total_gb - used_gb
                                        usage_percentage = (used_gb / total_gb) * 100
                                        
                                        mount = StorageMount(
                                            mount_path=mountpoint,
                                            storage_type=StorageType.HOST_PATH,
                                            total_size=total_gb,
                                            used_size=used_gb,
                                            available_size=available_gb,
                                            usage_percentage=usage_percentage,
                                            mount_time=datetime.now(),
                                            last_access=datetime.now(),
                                            access_pattern={},
                                            cost_per_gb_month=self._get_cost_per_gb_month(StorageType.HOST_PATH, "default")
                                        )
                                        mounts.append(mount)
                        
                        except json.JSONDecodeError:
                            continue
        
        except Exception as e:
            logger.error(f"分析Docker挂载点失败: {e}")
        
        return mounts
    
    def _determine_storage_type(self, filesystem: str, mount_path: str) -> StorageType:
        """确定存储类型"""
        if 'nfs' in filesystem.lower():
            return StorageType.NFS
        elif 'tmpfs' in filesystem.lower():
            return StorageType.MEMORY
        elif '/dev/' in filesystem:
            return StorageType.HOST_PATH
        elif 'minio' in mount_path.lower():
            return StorageType.MINIO
        else:
            return StorageType.HOST_PATH
    
    async def _analyze_access_pattern(self, mount_path: str) -> Dict[str, Any]:
        """分析访问模式"""
        try:
            # 检查文件访问时间
            if os.path.exists(mount_path):
                stat = os.stat(mount_path)
                return {
                    "last_access": datetime.fromtimestamp(stat.st_atime),
                    "last_modify": datetime.fromtimestamp(stat.st_mtime),
                    "file_count": self._count_files(mount_path),
                    "access_frequency": "medium"  # 简化处理
                }
        except Exception as e:
            logger.warning(f"分析访问模式失败 {mount_path}: {e}")
        
        return {}
    
    def _count_files(self, path: str) -> int:
        """统计文件数量"""
        try:
            count = 0
            for root, dirs, files in os.walk(path):
                count += len(files)
            return count
        except Exception:
            return 0
    
    def _parse_size_to_gb(self, size_str: str) -> float:
        """解析大小字符串为GB"""
        try:
            size_str = size_str.upper()
            if 'T' in size_str:
                return float(size_str.replace('T', '')) * 1024
            elif 'G' in size_str:
                return float(size_str.replace('G', ''))
            elif 'M' in size_str:
                return float(size_str.replace('M', '')) / 1024
            elif 'K' in size_str:
                return float(size_str.replace('K', '')) / (1024 * 1024)
            else:
                return float(size_str) / (1024 * 1024 * 1024)
        except Exception:
            return 0.0
    
    def _get_cost_per_gb_month(self, storage_type: StorageType, storage_class: str) -> float:
        """获取每GB每月的成本"""
        return self.cost_config.get(storage_type, {}).get(storage_class, 0.10)
    
    async def calculate_costs(self, mounts: List[StorageMount]) -> List[CostAnalysis]:
        """
        计算存储成本
        
        Args:
            mounts: 存储挂载点列表
            
        Returns:
            成本分析结果列表
        """
        cost_analyses = []
        
        for mount in mounts:
            # 计算月度成本
            monthly_cost = mount.total_size * mount.cost_per_gb_month
            
            # 计算成本效率
            cost_efficiency = mount.usage_percentage / (monthly_cost + 0.01)  # 避免除零
            
            # 生成优化建议
            suggestions = self._generate_optimization_suggestions(mount)
            
            # 计算潜在节省
            potential_savings = self._calculate_potential_savings(mount, suggestions)
            
            analysis = CostAnalysis(
                mount_path=mount.mount_path,
                storage_type=mount.storage_type,
                monthly_cost=monthly_cost,
                cost_efficiency=cost_efficiency,
                optimization_suggestions=suggestions,
                potential_savings=potential_savings
            )
            cost_analyses.append(analysis)
        
        return cost_analyses
    
    def _generate_optimization_suggestions(self, mount: StorageMount) -> List[str]:
        """生成优化建议"""
        suggestions = []
        
        # 低使用率建议
        if mount.usage_percentage < 20:
            suggestions.append(self.optimization_templates["low_usage"])
        
        # 高成本建议
        if mount.cost_per_gb_month > 0.20:  # 成本阈值
            suggestions.append(self.optimization_templates["high_cost"])
        
        # 未使用存储建议
        if mount.usage_percentage < 5:
            suggestions.append(self.optimization_templates["unused_storage"])
        
        # 过度配置建议
        if mount.available_size > mount.used_size * 3:
            suggestions.append(self.optimization_templates["over_provisioned"])
        
        # 存储类型优化建议
        if mount.storage_type == StorageType.PVC and mount.cost_per_gb_month > 0.15:
            suggestions.append("建议迁移到S3或NFS以降低成本")
        
        return suggestions
    
    def _calculate_potential_savings(self, mount: StorageMount, suggestions: List[str]) -> float:
        """计算潜在节省"""
        potential_savings = 0.0
        
        for suggestion in suggestions:
            if "降级" in suggestion or "迁移" in suggestion:
                # 假设可以节省30%的成本
                potential_savings += mount.monthly_cost * 0.3
            elif "删除" in suggestion:
                # 假设可以节省100%的成本
                potential_savings += mount.monthly_cost
            elif "减少容量" in suggestion:
                # 假设可以减少50%的容量
                potential_savings += mount.monthly_cost * 0.5
        
        return potential_savings
    
    async def generate_optimization_report(self) -> Dict[str, Any]:
        """
        生成优化报告
        
        Returns:
            优化报告
        """
        # 分析所有挂载点
        mounts = await self.analyze_all_mounts()
        
        # 计算成本
        cost_analyses = await self.calculate_costs(mounts)
        
        # 计算总体统计
        total_monthly_cost = sum(analysis.monthly_cost for analysis in cost_analyses)
        total_potential_savings = sum(analysis.potential_savings for analysis in cost_analyses)
        total_storage_gb = sum(mount.total_size for mount in mounts)
        total_used_gb = sum(mount.used_size for mount in mounts)
        average_usage_percentage = (total_used_gb / total_storage_gb * 100) if total_storage_gb > 0 else 0
        
        # 按存储类型分组
        storage_type_stats = {}
        for mount in mounts:
            storage_type = mount.storage_type.value
            if storage_type not in storage_type_stats:
                storage_type_stats[storage_type] = {
                    'count': 0,
                    'total_size': 0,
                    'total_cost': 0,
                    'average_usage': 0
                }
            
            storage_type_stats[storage_type]['count'] += 1
            storage_type_stats[storage_type]['total_size'] += mount.total_size
            storage_type_stats[storage_type]['total_cost'] += mount.total_size * mount.cost_per_gb_month
        
        # 计算平均使用率
        for storage_type in storage_type_stats:
            total_size = storage_type_stats[storage_type]['total_size']
            if total_size > 0:
                total_used = sum(mount.used_size for mount in mounts if mount.storage_type.value == storage_type)
                storage_type_stats[storage_type]['average_usage'] = (total_used / total_size) * 100
        
        # 生成报告
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_mounts': len(mounts),
                'total_storage_gb': total_storage_gb,
                'total_used_gb': total_used_gb,
                'average_usage_percentage': average_usage_percentage,
                'total_monthly_cost': total_monthly_cost,
                'total_potential_savings': total_potential_savings,
                'savings_percentage': (total_potential_savings / total_monthly_cost * 100) if total_monthly_cost > 0 else 0
            },
            'storage_type_analysis': storage_type_stats,
            'mount_details': [
                {
                    'mount_path': mount.mount_path,
                    'storage_type': mount.storage_type.value,
                    'total_size_gb': mount.total_size,
                    'used_size_gb': mount.used_size,
                    'usage_percentage': mount.usage_percentage,
                    'monthly_cost': mount.cost_per_gb_month * mount.total_size,
                    'cost_per_gb_month': mount.cost_per_gb_month
                }
                for mount in mounts
            ],
            'cost_analysis': [
                {
                    'mount_path': analysis.mount_path,
                    'storage_type': analysis.storage_type.value,
                    'monthly_cost': analysis.monthly_cost,
                    'cost_efficiency': analysis.cost_efficiency,
                    'optimization_suggestions': analysis.optimization_suggestions,
                    'potential_savings': analysis.potential_savings
                }
                for analysis in cost_analyses
            ],
            'recommendations': self._generate_global_recommendations(cost_analyses, storage_type_stats)
        }
        
        return report
    
    def _generate_global_recommendations(self, cost_analyses: List[CostAnalysis], 
                                       storage_type_stats: Dict[str, Any]) -> List[str]:
        """生成全局优化建议"""
        recommendations = []
        
        # 计算总体成本效率
        total_cost = sum(analysis.monthly_cost for analysis in cost_analyses)
        total_savings = sum(analysis.potential_savings for analysis in cost_analyses)
        
        if total_savings > total_cost * 0.2:  # 如果潜在节省超过20%
            recommendations.append(f"总体优化潜力巨大，预计可节省 ${total_savings:.2f}/月")
        
        # 存储类型优化建议
        if 'pvc' in storage_type_stats and storage_type_stats['pvc']['total_cost'] > total_cost * 0.5:
            recommendations.append("PVC存储成本占比过高，建议迁移部分数据到S3或NFS")
        
        if 'memory' in storage_type_stats and storage_type_stats['memory']['total_cost'] > total_cost * 0.3:
            recommendations.append("内存存储成本较高，建议优化内存使用或迁移到磁盘存储")
        
        # 使用率优化建议
        low_usage_mounts = [analysis for analysis in cost_analyses if '低使用率' in str(analysis.optimization_suggestions)]
        if len(low_usage_mounts) > len(cost_analyses) * 0.3:
            recommendations.append("超过30%的存储挂载点使用率较低，建议进行存储整合")
        
        return recommendations
    
    async def save_report(self, report: Dict[str, Any], filepath: str = "cost_optimization_report.json"):
        """保存优化报告"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)
            logger.info(f"优化报告已保存到: {filepath}")
        except Exception as e:
            logger.error(f"保存报告失败: {e}")
    
    async def export_to_csv(self, report: Dict[str, Any], filepath: str = "cost_optimization_data.csv"):
        """导出数据到CSV"""
        try:
            # 创建挂载点详情DataFrame
            mount_data = []
            for mount_detail in report['mount_details']:
                mount_data.append({
                    'Mount Path': mount_detail['mount_path'],
                    'Storage Type': mount_detail['storage_type'],
                    'Total Size (GB)': mount_detail['total_size_gb'],
                    'Used Size (GB)': mount_detail['used_size_gb'],
                    'Usage Percentage': mount_detail['usage_percentage'],
                    'Monthly Cost ($)': mount_detail['monthly_cost'],
                    'Cost per GB/Month ($)': mount_detail['cost_per_gb_month']
                })
            
            df = pd.DataFrame(mount_data)
            df.to_csv(filepath, index=False, encoding='utf-8')
            logger.info(f"数据已导出到: {filepath}")
            
        except Exception as e:
            logger.error(f"导出CSV失败: {e}")


async def main():
    """主函数"""
    print("🔍 开始成本优化分析...")
    
    # 创建成本优化器
    optimizer = CostOptimizer()
    
    # 生成优化报告
    report = await optimizer.generate_optimization_report()
    
    # 打印摘要
    summary = report['summary']
    print(f"\n📊 成本优化分析摘要:")
    print(f"   总挂载点数量: {summary['total_mounts']}")
    print(f"   总存储容量: {summary['total_storage_gb']:.2f} GB")
    print(f"   已使用存储: {summary['total_used_gb']:.2f} GB")
    print(f"   平均使用率: {summary['average_usage_percentage']:.1f}%")
    print(f"   月度总成本: ${summary['total_monthly_cost']:.2f}")
    print(f"   潜在节省: ${summary['total_potential_savings']:.2f}")
    print(f"   节省比例: {summary['savings_percentage']:.1f}%")
    
    # 打印存储类型分析
    print(f"\n📈 存储类型分析:")
    for storage_type, stats in report['storage_type_analysis'].items():
        print(f"   {storage_type.upper()}:")
        print(f"     挂载点数量: {stats['count']}")
        print(f"     总容量: {stats['total_size']:.2f} GB")
        print(f"     总成本: ${stats['total_cost']:.2f}/月")
        print(f"     平均使用率: {stats['average_usage']:.1f}%")
    
    # 打印优化建议
    print(f"\n💡 全局优化建议:")
    for recommendation in report['recommendations']:
        print(f"   • {recommendation}")
    
    # 打印详细成本分析
    print(f"\n💰 详细成本分析:")
    for analysis in report['cost_analysis'][:5]:  # 只显示前5个
        print(f"   {analysis['mount_path']}:")
        print(f"     存储类型: {analysis['storage_type']}")
        print(f"     月度成本: ${analysis['monthly_cost']:.2f}")
        print(f"     潜在节省: ${analysis['potential_savings']:.2f}")
        
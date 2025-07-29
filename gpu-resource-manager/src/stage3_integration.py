import logging
import threading
from typing import Dict, Any, Optional
from gpu_exporter import GPUExporter
from monitoring_aggregator import MonitoringAggregator
from web_api import MonitoringWebAPI

class Stage3ResourceManager:
    """第三阶段资源管理器集成"""
    
    def __init__(self, k8s_client, stage2_manager, prometheus_url: str = "http://localhost:9090"):
        self.k8s_client = k8s_client
        self.stage2_manager = stage2_manager
        self.prometheus_url = prometheus_url
        self.logger = logging.getLogger(__name__)
        
        # 初始化第三阶段组件
        self.gpu_exporter = GPUExporter(k8s_client, port=9090)
        self.aggregator = MonitoringAggregator(self._create_prometheus_client(), k8s_client)
        self.web_api = MonitoringWebAPI(self.gpu_exporter, self.aggregator, stage2_manager)
        
        # 服务状态
        self.services_running = False
        self.web_server_thread = None
    
    def _create_prometheus_client(self):
        """创建Prometheus客户端（模拟实现）"""
        class MockPrometheusClient:
            def query_metric(self, query):
                # 模拟Prometheus查询结果
                import random
                return [type('Metric', (), {'value': random.uniform(0, 100)})() for _ in range(5)]
        
        return MockPrometheusClient()
    
    def start_all_services(self):
        """启动所有第三阶段服务"""
        try:
            # 启动GPU指标导出器
            self.gpu_exporter.start()
            self.logger.info("GPU指标导出器已启动")
            
            # 启动数据聚合器
            self.aggregator.start_aggregation()
            self.logger.info("监控数据聚合器已启动")
            
            # 启动Web API服务器
            self.web_server_thread = threading.Thread(
                target=self.web_api.run,
                kwargs={'host': '0.0.0.0', 'port': 8080, 'debug': False},
                daemon=True
            )
            self.web_server_thread.start()
            self.logger.info("Web API服务器已启动")
            
            self.services_running = True
            self.logger.info("第三阶段所有服务已启动")
            
        except Exception as e:
            self.logger.error(f"启动第三阶段服务失败: {e}")
    
    def stop_all_services(self):
        """停止所有第三阶段服务"""
        try:
            # 停止GPU指标导出器
            self.gpu_exporter.stop()
            
            # 停止数据聚合器
            self.aggregator.stop_aggregation()
            
            # Web服务器会在主线程结束时自动停止
            
            self.services_running = False
            self.logger.info("第三阶段所有服务已停止")
            
        except Exception as e:
            self.logger.error(f"停止第三阶段服务失败: {e}")
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """获取监控系统状态"""
        return {
            'gpu_exporter_running': self.gpu_exporter.running,
            'aggregator_running': self.aggregator.running,
            'web_api_running': self.services_running,
            'exporter_port': self.gpu_exporter.port,
            'web_api_port': 8080
        }
    
    def get_cluster_summary(self) -> Dict[str, Any]:
        """获取集群摘要信息"""
        return self.aggregator.get_cluster_summary()
    
    def get_resource_trends(self) -> Dict[str, Any]:
        """获取资源趋势数据"""
        return self.aggregator.get_resource_trends()
    
    def get_node_performance(self) -> Dict[str, Any]:
        """获取节点性能排名"""
        return {
            'nodes': self.aggregator.get_node_performance_ranking()
        }
    
    def export_dashboard_config(self, dashboard_type: str = "overview", filename: str = None):
        """导出仪表板配置"""
        try:
            from grafana_dashboard_generator import GrafanaDashboardGenerator
            generator = GrafanaDashboardGenerator()
            
            if dashboard_type == "overview":
                dashboard = generator.generate_gpu_overview_dashboard()
            else:
                dashboard = generator.generate_gpu_resource_dashboard()
            
            if filename:
                generator.save_dashboard_to_file(dashboard, filename)
                self.logger.info(f"仪表板配置已保存到: {filename}")
            
            return dashboard
            
        except Exception as e:
            self.logger.error(f"导出仪表板配置失败: {e}")
            return None
    
    def record_custom_metric(self, metric_name: str, value: float, labels: Dict[str, str] = None):
        """记录自定义指标"""
        try:
            # 这里可以添加自定义指标记录逻辑
            self.logger.info(f"记录自定义指标: {metric_name} = {value}, labels: {labels}")
        except Exception as e:
            self.logger.error(f"记录自定义指标失败: {e}")
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """获取告警摘要"""
        try:
            # 模拟告警数据
            alerts = self.web_api._get_simulated_alerts()
            
            # 按严重程度分组
            alert_summary = {
                'critical': len([a for a in alerts if a['severity'] == 'critical']),
                'warning': len([a for a in alerts if a['severity'] == 'warning']),
                'info': len([a for a in alerts if a['severity'] == 'info']),
                'total': len(alerts),
                'recent_alerts': alerts[:5]  # 最近5个告警
            }
            
            return alert_summary
            
        except Exception as e:
            self.logger.error(f"获取告警摘要失败: {e}")
            return {}
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        try:
            # 获取系统性能指标
            cluster_summary = self.get_cluster_summary()
            trends = self.get_resource_trends()
            node_performance = self.get_node_performance()
            alert_summary = self.get_alert_summary()
            
            return {
                'cluster_summary': cluster_summary,
                'resource_trends': trends,
                'node_performance': node_performance,
                'alert_summary': alert_summary,
                'monitoring_status': self.get_monitoring_status()
            }
            
        except Exception as e:
            self.logger.error(f"获取性能指标失败: {e}")
            return {}
    
    def generate_monitoring_report(self) -> Dict[str, Any]:
        """生成监控报告"""
        try:
            performance_metrics = self.get_performance_metrics()
            
            # 生成报告摘要
            report = {
                'timestamp': performance_metrics.get('cluster_summary', {}).get('timestamp'),
                'summary': {
                    'total_gpus': performance_metrics.get('cluster_summary', {}).get('current', {}).get('total_gpus', 0),
                    'allocated_gpus': performance_metrics.get('cluster_summary', {}).get('current', {}).get('allocated_gpus', 0),
                    'avg_utilization': performance_metrics.get('cluster_summary', {}).get('current', {}).get('avg_utilization', 0),
                    'active_pods': performance_metrics.get('cluster_summary', {}).get('current', {}).get('active_pods', 0),
                    'pending_pods': performance_metrics.get('cluster_summary', {}).get('current', {}).get('pending_pods', 0),
                    'total_alerts': performance_metrics.get('alert_summary', {}).get('total', 0)
                },
                'trends': performance_metrics.get('cluster_summary', {}).get('trends', {}),
                'top_performing_nodes': performance_metrics.get('node_performance', {}).get('nodes', [])[:5],
                'recent_alerts': performance_metrics.get('alert_summary', {}).get('recent_alerts', []),
                'system_status': performance_metrics.get('monitoring_status', {})
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"生成监控报告失败: {e}")
            return {} 
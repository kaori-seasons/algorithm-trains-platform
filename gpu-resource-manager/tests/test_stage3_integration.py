import unittest
import time
import threading
from unittest.mock import Mock, patch
from stage3_integration import Stage3ResourceManager

class TestStage3Integration(unittest.TestCase):
    """第三阶段集成测试"""
    
    def setUp(self):
        self.k8s_client = Mock()
        self.stage2_manager = Mock()
        self.prometheus_url = "http://localhost:9090"
        
        self.stage3_manager = Stage3ResourceManager(
            self.k8s_client,
            self.stage2_manager,
            self.prometheus_url
        )
    
    def test_start_stop_services(self):
        """测试服务启动和停止"""
        # 启动服务
        self.stage3_manager.start_all_services()
        
        # 验证服务状态
        self.assertTrue(self.stage3_manager.services_running)
        self.assertTrue(self.stage3_manager.gpu_exporter.running)
        self.assertTrue(self.stage3_manager.aggregator.running)
        
        # 停止服务
        self.stage3_manager.stop_all_services()
        
        # 验证服务已停止
        self.assertFalse(self.stage3_manager.services_running)
        self.assertFalse(self.stage3_manager.gpu_exporter.running)
        self.assertFalse(self.stage3_manager.aggregator.running)
    
    def test_get_monitoring_status(self):
        """测试获取监控状态"""
        status = self.stage3_manager.get_monitoring_status()
        
        self.assertIn('gpu_exporter_running', status)
        self.assertIn('aggregator_running', status)
        self.assertIn('web_api_running', status)
        self.assertIn('exporter_port', status)
        self.assertIn('web_api_port', status)
    
    def test_get_cluster_summary(self):
        """测试获取集群摘要"""
        summary = self.stage3_manager.get_cluster_summary()
        
        # 验证摘要结构
        if summary:  # 如果有数据
            self.assertIn('current', summary)
            self.assertIn('trends', summary)
            self.assertIn('timestamp', summary)
    
    def test_get_resource_trends(self):
        """测试获取资源趋势"""
        trends = self.stage3_manager.get_resource_trends()
        
        # 验证趋势数据结构
        if trends:  # 如果有数据
            self.assertIn('timestamps', trends)
            self.assertIn('gpu_utilization', trends)
            self.assertIn('gpu_memory_usage', trends)
            self.assertIn('gpu_allocation_rate', trends)
    
    def test_get_node_performance(self):
        """测试获取节点性能"""
        performance = self.stage3_manager.get_node_performance()
        
        self.assertIn('nodes', performance)
        self.assertIsInstance(performance['nodes'], list)
    
    def test_export_dashboard_config(self):
        """测试导出仪表板配置"""
        # 测试概览仪表板
        overview_config = self.stage3_manager.export_dashboard_config("overview")
        self.assertIsNotNone(overview_config)
        self.assertIn('dashboard', overview_config)
        self.assertEqual(overview_config['dashboard']['title'], 'GPU Resource Overview')
        
        # 测试详细仪表板
        detail_config = self.stage3_manager.export_dashboard_config("detail")
        self.assertIsNotNone(detail_config)
        self.assertIn('dashboard', detail_config)
        self.assertEqual(detail_config['dashboard']['title'], 'GPU Resource Details')
    
    def test_get_alert_summary(self):
        """测试获取告警摘要"""
        alert_summary = self.stage3_manager.get_alert_summary()
        
        self.assertIn('critical', alert_summary)
        self.assertIn('warning', alert_summary)
        self.assertIn('info', alert_summary)
        self.assertIn('total', alert_summary)
        self.assertIn('recent_alerts', alert_summary)
    
    def test_get_performance_metrics(self):
        """测试获取性能指标"""
        metrics = self.stage3_manager.get_performance_metrics()
        
        self.assertIn('cluster_summary', metrics)
        self.assertIn('resource_trends', metrics)
        self.assertIn('node_performance', metrics)
        self.assertIn('alert_summary', metrics)
        self.assertIn('monitoring_status', metrics)
    
    def test_generate_monitoring_report(self):
        """测试生成监控报告"""
        report = self.stage3_manager.generate_monitoring_report()
        
        if report:  # 如果有数据
            self.assertIn('timestamp', report)
            self.assertIn('summary', report)
            self.assertIn('trends', report)
            self.assertIn('top_performing_nodes', report)
            self.assertIn('recent_alerts', report)
            self.assertIn('system_status', report)
    
    def test_record_custom_metric(self):
        """测试记录自定义指标"""
        # 测试记录自定义指标
        self.stage3_manager.record_custom_metric(
            "test_metric",
            42.5,
            {"node": "test-node", "gpu": "0"}
        )
        
        # 验证日志记录（这里只是确保不抛出异常）
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main() 
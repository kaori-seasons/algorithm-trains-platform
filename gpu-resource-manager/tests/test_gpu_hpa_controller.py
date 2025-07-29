import unittest
from unittest.mock import Mock, patch
from src.gpu_hpa_controller import GPUHPAController, HPAMetric, MetricType

class TestGPUHPAController(unittest.TestCase):
    
    def setUp(self):
        self.k8s_client = Mock()
        self.hpa_controller = GPUHPAController(self.k8s_client)
    
    def test_parse_metrics(self):
        """测试指标解析"""
        metrics = ["cpu:70%", "mem:80%", "gpu:60%"]
        parsed = self.hpa_controller._parse_metrics(metrics)
        
        self.assertEqual(len(parsed), 3)
        
        # 检查CPU指标
        cpu_metric = next(m for m in parsed if m.name == 'cpu')
        self.assertEqual(cpu_metric.metric_type, MetricType.RESOURCE)
        self.assertEqual(cpu_metric.target_value, 70.0)
        
        # 检查GPU指标
        gpu_metric = next(m for m in parsed if m.name == 'container_gpu_usage')
        self.assertEqual(gpu_metric.metric_type, MetricType.PODS)
        self.assertEqual(gpu_metric.target_value, 0.6)
    
    def test_build_metric_config_resource(self):
        """测试资源指标配置构建"""
        metric = HPAMetric(
            metric_type=MetricType.RESOURCE,
            name='cpu',
            target_value=50.0,
            target_type='Utilization'
        )
        
        config = self.hpa_controller._build_metric_config(metric)
        
        self.assertIsNotNone(config)
        self.assertEqual(config['type'], 'Resource')
        self.assertEqual(config['resource']['name'], 'cpu')
        self.assertEqual(config['resource']['target']['averageUtilization'], 50)
    
    def test_build_metric_config_pods(self):
        """测试Pod指标配置构建"""
        metric = HPAMetric(
            metric_type=MetricType.PODS,
            name='container_gpu_usage',
            target_value=0.7,
            target_type='AverageValue'
        )
        
        config = self.hpa_controller._build_metric_config(metric)
        
        self.assertIsNotNone(config)
        self.assertEqual(config['type'], 'Pods')
        self.assertEqual(config['pods']['metric']['name'], 'container_gpu_usage')
        self.assertEqual(config['pods']['target']['averageValue'], '0.7')
    
    def test_build_hpa_spec(self):
        """测试HPA规格构建"""
        metrics = [
            HPAMetric(MetricType.RESOURCE, 'cpu', 50.0),
            HPAMetric(MetricType.PODS, 'container_gpu_usage', 0.7)
        ]
        
        spec = self.hpa_controller._build_hpa_spec(
            name='test-app',
            namespace='default',
            min_replicas=1,
            max_replicas=10,
            metrics=metrics
        )
        
        self.assertEqual(spec['apiVersion'], 'autoscaling/v2')
        self.assertEqual(spec['kind'], 'HorizontalPodAutoscaler')
        self.assertEqual(spec['metadata']['name'], 'test-app')
        self.assertEqual(spec['metadata']['namespace'], 'default')
        self.assertEqual(spec['spec']['minReplicas'], 1)
        self.assertEqual(spec['spec']['maxReplicas'], 10)
        self.assertEqual(len(spec['spec']['metrics']), 2)
    
    @patch('src.gpu_hpa_controller.client.AutoscalingV2Api')
    @patch('src.gpu_hpa_controller.client.AutoscalingV1Api')
    def test_create_hpa_success(self, mock_v1_api, mock_v2_api):
        """测试成功创建HPA"""
        # 完全模拟API调用
        mock_v2_instance = Mock()
        mock_v1_instance = Mock()
        mock_v2_api.return_value = mock_v2_instance
        mock_v1_api.return_value = mock_v1_instance
    
        # 模拟删除操作成功（不抛出异常）
        mock_v2_instance.delete_namespaced_horizontal_pod_autoscaler.return_value = None
        mock_v1_instance.delete_namespaced_horizontal_pod_autoscaler.return_value = None
        # 模拟创建操作成功
        mock_v2_instance.create_namespaced_horizontal_pod_autoscaler.return_value = None
        
        # 直接模拟hpa_controller的autoscaling_v2属性
        self.hpa_controller.autoscaling_v2 = mock_v2_instance
    
        metrics = ["cpu:50%", "gpu:70%"]
        result = self.hpa_controller.create_hpa(
            namespace='default',
            name='test-app',
            min_replicas=1,
            max_replicas=10,
            metrics=metrics
        )
    
        self.assertTrue(result)
        mock_v2_instance.create_namespaced_horizontal_pod_autoscaler.assert_called_once()
    
    @patch('src.gpu_hpa_controller.client.AutoscalingV2Api')
    def test_create_hpa_failure(self, mock_api):
        """测试创建HPA失败"""
        mock_api_instance = Mock()
        mock_api_instance.create_namespaced_horizontal_pod_autoscaler.side_effect = Exception("API Error")
        mock_api.return_value = mock_api_instance
        
        # 直接模拟hpa_controller的autoscaling_v2属性
        self.hpa_controller.autoscaling_v2 = mock_api_instance
        
        metrics = ["cpu:50%"]
        result = self.hpa_controller.create_hpa(
            namespace='default',
            name='test-app',
            min_replicas=1,
            max_replicas=10,
            metrics=metrics
        )
        
        self.assertFalse(result)

if __name__ == '__main__':
    unittest.main() 
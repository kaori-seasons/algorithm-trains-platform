import unittest
from unittest.mock import Mock, patch
from src.gpu_scheduler import GPUScheduler, SchedulingRequest, SchedulingPolicy
from src.gpu_parser import GPUResourceParser

class TestGPUScheduler(unittest.TestCase):
    
    def setUp(self):
        self.k8s_client = Mock()
        self.gpu_parser = GPUResourceParser()
        self.memory_guard = Mock()
        self.scheduler = GPUScheduler(self.k8s_client, self.gpu_parser, self.memory_guard)
    
    def test_schedule_pod_success(self):
        """测试成功调度Pod"""
        # 模拟内存验证通过
        self.memory_guard.validate_memory_requirement.return_value = True
        
        # 模拟节点数据
        mock_nodes = [
            {
                'name': 'gpu-node-1',
                'ready': True,
                'schedulable': True,
                'labels': {'gpu-type': 'V100'},
                'gpu_info': {'nvidia': {'allocatable': 4, 'available': 2}},
                'cpu_allocatable': 8.0,
                'memory_allocatable': 32.0,
                'used_cpu': 2.0,
                'used_memory': 8.0
            }
        ]
        self.k8s_client.get_nodes.return_value = mock_nodes
        
        # 模拟Pod数据
        self.k8s_client.get_pods.return_value = []
        
        request = SchedulingRequest(
            pod_name='test-pod',
            namespace='default',
            gpu_requirement='1(V100)',
            memory_requirement=16.0
        )
        
        result = self.scheduler.schedule_pod(request)
        
        self.assertIsNotNone(result)
        self.assertEqual(result, 'gpu-node-1')
    
    def test_schedule_pod_memory_validation_failed(self):
        """测试内存验证失败"""
        # 模拟内存验证失败
        self.memory_guard.validate_memory_requirement.return_value = False
        
        request = SchedulingRequest(
            pod_name='test-pod',
            namespace='default',
            gpu_requirement='1(V100)',
            memory_requirement=64.0  # 超出显存
        )
        
        result = self.scheduler.schedule_pod(request)
        
        self.assertIsNone(result)
    
    def test_schedule_pod_no_suitable_nodes(self):
        """测试没有合适的节点"""
        # 模拟内存验证通过
        self.memory_guard.validate_memory_requirement.return_value = True
        
        # 模拟没有合适的节点
        self.k8s_client.get_nodes.return_value = []
        
        request = SchedulingRequest(
            pod_name='test-pod',
            namespace='default',
            gpu_requirement='1(V100)',
            memory_requirement=16.0
        )
        
        result = self.scheduler.schedule_pod(request)
        
        self.assertIsNone(result)
    
    def test_calculate_node_score(self):
        """测试节点评分计算"""
        node = {
            'name': 'test-node',
            'cpu_allocatable': 8.0,
            'memory_allocatable': 32.0,
            'used_cpu': 2.0,
            'used_memory': 8.0,
            'gpu_info': {'nvidia': {'allocatable': 4, 'available': 2}}
        }
        
        # 模拟Pod数据
        self.k8s_client.get_pods.return_value = []
        
        gpu_info = self.gpu_parser.parse_gpu_resource('1(V100)')
        score = self.scheduler._calculate_node_score(node, gpu_info)
        
        self.assertIsInstance(score, float)
        self.assertGreater(score, 0)
    
    def test_get_available_gpus(self):
        """测试获取可用GPU数量"""
        node = {
            'name': 'test-node',
            'gpu_info': {
                'nvidia': {'allocatable': 4, 'available': 2}
            }
        }
        
        # 模拟Pod数据
        self.k8s_client.get_pods.return_value = [
            {
                'node_name': 'test-node',
                'gpu_resources': {'nvidia.com/gpu': 1}
            }
        ]
        
        available_gpus = self.scheduler._get_available_gpus(node)
        
        self.assertEqual(available_gpus, 3.0)  # 4 - 1 = 3
    
    def test_add_scheduling_request(self):
        """测试添加调度请求"""
        request = SchedulingRequest(
            pod_name='test-pod',
            namespace='default',
            gpu_requirement='1(V100)',
            memory_requirement=16.0,
            priority=2
        )
        
        self.scheduler.add_scheduling_request(request)
        
        self.assertEqual(len(self.scheduler.pending_requests), 1)
        self.assertEqual(self.scheduler.pending_requests[0].pod_name, 'test-pod')
    
    def test_process_pending_requests(self):
        """测试处理待调度请求"""
        # 添加一个调度请求
        request = SchedulingRequest(
            pod_name='test-pod',
            namespace='default',
            gpu_requirement='1(V100)',
            memory_requirement=16.0
        )
        self.scheduler.add_scheduling_request(request)
        
        # 模拟调度成功
        with patch.object(self.scheduler, 'schedule_pod', return_value='gpu-node-1'):
            self.scheduler.process_pending_requests()
        
        # 请求应该被处理并从队列中移除
        self.assertEqual(len(self.scheduler.pending_requests), 0)
    
    def test_node_selector_filtering(self):
        """测试节点选择器过滤"""
        # 模拟内存验证通过
        self.memory_guard.validate_memory_requirement.return_value = True
        
        # 模拟节点数据
        mock_nodes = [
            {
                'name': 'gpu-node-1',
                'ready': True,
                'schedulable': True,
                'labels': {'gpu-type': 'V100', 'zone': 'zone-a'},
                'gpu_info': {'nvidia': {'allocatable': 4, 'available': 2}},
                'cpu_allocatable': 8.0,
                'memory_allocatable': 32.0,
                'used_cpu': 2.0,
                'used_memory': 8.0
            },
            {
                'name': 'gpu-node-2',
                'ready': True,
                'schedulable': True,
                'labels': {'gpu-type': 'V100', 'zone': 'zone-b'},
                'gpu_info': {'nvidia': {'allocatable': 4, 'available': 2}},
                'cpu_allocatable': 8.0,
                'memory_allocatable': 32.0,
                'used_cpu': 2.0,
                'used_memory': 8.0
            }
        ]
        self.k8s_client.get_nodes.return_value = mock_nodes
        self.k8s_client.get_pods.return_value = []
        
        request = SchedulingRequest(
            pod_name='test-pod',
            namespace='default',
            gpu_requirement='1(V100)',
            memory_requirement=16.0,
            node_selector={'zone': 'zone-a'}
        )
        
        gpu_info = self.gpu_parser.parse_gpu_resource('1(V100)')
        suitable_nodes = self.scheduler._find_suitable_nodes(request, gpu_info)
        
        # 应该只找到zone-a的节点
        self.assertEqual(len(suitable_nodes), 1)
        self.assertEqual(suitable_nodes[0]['node']['name'], 'gpu-node-1')

if __name__ == '__main__':
    unittest.main() 
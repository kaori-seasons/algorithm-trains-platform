import unittest
from src.gpu_parser import GPUResourceParser, GPUResource

class TestGPUResourceParser(unittest.TestCase):
    
    def setUp(self):
        self.parser = GPUResourceParser()
    
    def test_parse_simple_gpu_count(self):
        """测试简单GPU数量解析"""
        result = self.parser.parse_gpu_resource("2")
        self.assertEqual(result.gpu_num, 2)
        self.assertIsNone(result.gpu_type)
        self.assertEqual(result.resource_name, "nvidia.com/gpu")
    
    def test_parse_fractional_gpu(self):
        """测试小数GPU解析"""
        result = self.parser.parse_gpu_resource("0.5")
        self.assertEqual(result.gpu_num, 0.5)
        self.assertIsNone(result.gpu_type)
    
    def test_parse_gpu_with_type(self):
        """测试带GPU型号的解析"""
        result = self.parser.parse_gpu_resource("2(V100)")
        self.assertEqual(result.gpu_num, 2)
        self.assertEqual(result.gpu_type, "V100")
    
    def test_parse_gpu_with_vendor_and_type(self):
        """测试带厂商和型号的解析"""
        result = self.parser.parse_gpu_resource("1(nvidia,V100)")
        self.assertEqual(result.gpu_num, 1)
        self.assertEqual(result.gpu_type, "V100")
        self.assertEqual(result.resource_name, "nvidia.com/gpu")
    
    def test_parse_memory_and_compute_ratio(self):
        """测试显存和算力比例解析"""
        result = self.parser.parse_gpu_resource("8G,0.5")
        self.assertEqual(result.gpu_num, 0.5)
        self.assertEqual(result.memory_gb, 8.0)
        self.assertEqual(result.compute_ratio, 0.5)
    
    def test_validate_gpu_memory(self):
        """测试GPU显存验证"""
        gpu_resource = GPUResource(
            gpu_num=1,
            gpu_type="V100",
            resource_name="nvidia.com/gpu"
        )
        
        # V100有32GB显存，应该满足16GB需求
        self.assertTrue(self.parser.validate_gpu_memory(gpu_resource, 16.0))
        
        # 不应该满足64GB需求
        self.assertFalse(self.parser.validate_gpu_memory(gpu_resource, 64.0))
    
    def test_chinese_brackets(self):
        """测试中文括号解析"""
        result = self.parser.parse_gpu_resource("2（V100）")
        self.assertEqual(result.gpu_num, 2)
        self.assertEqual(result.gpu_type, "V100")
    
    def test_empty_resource(self):
        """测试空资源解析"""
        result = self.parser.parse_gpu_resource("")
        self.assertEqual(result.gpu_num, 0)
        self.assertIsNone(result.gpu_type)
    
    def test_invalid_format(self):
        """测试无效格式解析"""
        result = self.parser.parse_gpu_resource("invalid")
        self.assertEqual(result.gpu_num, 0)

if __name__ == '__main__':
    unittest.main() 
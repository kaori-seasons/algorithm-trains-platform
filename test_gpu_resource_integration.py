#!/usr/bin/env python3
"""
GPU资源管理集成测试脚本
验证GPU资源分配、监控和训练功能
"""
import asyncio
import logging
import sys
import os
from typing import Dict, Any
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from backend.algorithm_engine.gpu_resource_integration import (
    get_gpu_resource_manager,
    get_tensorflow_gpu_integration,
    get_pytorch_gpu_integration,
    TrainingGPUConfig
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GPUResourceIntegrationTest:
    """GPU资源管理集成测试"""
    
    def __init__(self):
        self.gpu_manager = get_gpu_resource_manager()
        self.tensorflow_gpu = get_tensorflow_gpu_integration()
        self.pytorch_gpu = get_pytorch_gpu_integration()
    
    def test_gpu_manager_initialization(self):
        """测试GPU管理器初始化"""
        logger.info("=== 测试GPU管理器初始化 ===")
        
        try:
            if self.gpu_manager.initialized:
                logger.info("✅ GPU资源管理器初始化成功")
                return True
            else:
                logger.warning("⚠️ GPU资源管理器未初始化，将使用模拟模式")
                return False
        except Exception as e:
            logger.error(f"❌ GPU管理器初始化失败: {e}")
            return False
    
    def test_gpu_resource_status(self):
        """测试GPU资源状态查询"""
        logger.info("=== 测试GPU资源状态查询 ===")
        
        try:
            status = self.gpu_manager.get_gpu_monitoring_data()
            logger.info(f"GPU监控数据: {status}")
            
            if status.get('error'):
                logger.warning(f"⚠️ GPU监控数据获取失败: {status['error']}")
                return False
            else:
                logger.info("✅ GPU资源状态查询成功")
                return True
                
        except Exception as e:
            logger.error(f"❌ GPU资源状态查询失败: {e}")
            return False
    
    def test_gpu_node_discovery(self):
        """测试GPU节点发现"""
        logger.info("=== 测试GPU节点发现 ===")
        
        try:
            # 测试不同GPU类型的节点发现
            gpu_types = ['V100', 'A100', 'T4', None]
            
            for gpu_type in gpu_types:
                nodes = self.gpu_manager.get_available_gpu_nodes(
                    gpu_type=gpu_type,
                    min_memory_gb=16.0
                )
                
                logger.info(f"GPU类型 {gpu_type}: 发现 {len(nodes)} 个可用节点")
                
                for node in nodes:
                    logger.info(f"  - 节点: {node.node_name}, "
                              f"GPU类型: {node.gpu_type}, "
                              f"可用GPU: {node.available_gpus}, "
                              f"显存: {node.memory_per_gpu}GB")
            
            logger.info("✅ GPU节点发现测试成功")
            return True
            
        except Exception as e:
            logger.error(f"❌ GPU节点发现测试失败: {e}")
            return False
    
    def test_gpu_resource_validation(self):
        """测试GPU资源需求验证"""
        logger.info("=== 测试GPU资源需求验证 ===")
        
        try:
            # 测试不同的GPU配置
            test_configs = [
                TrainingGPUConfig(gpu_count=1, gpu_type='V100', memory_gb=16.0),
                TrainingGPUConfig(gpu_count=2, gpu_type='A100', memory_gb=32.0),
                TrainingGPUConfig(gpu_count=4, gpu_type='T4', memory_gb=8.0),
                TrainingGPUConfig(gpu_count=1, gpu_type='V100', memory_gb=64.0),  # 可能不满足
            ]
            
            for i, config in enumerate(test_configs):
                is_valid = self.gpu_manager.validate_gpu_requirements(config)
                logger.info(f"配置 {i+1}: GPU={config.gpu_count}x{config.gpu_type}, "
                          f"显存={config.memory_gb}GB, 验证结果={'✅通过' if is_valid else '❌不满足'}")
            
            logger.info("✅ GPU资源需求验证测试成功")
            return True
            
        except Exception as e:
            logger.error(f"❌ GPU资源需求验证测试失败: {e}")
            return False
    
    def test_gpu_resource_allocation(self):
        """测试GPU资源分配"""
        logger.info("=== 测试GPU资源分配 ===")
        
        try:
            # 测试GPU资源分配
            gpu_config = TrainingGPUConfig(
                gpu_count=1,
                gpu_type='V100',
                memory_gb=16.0,
                distributed_training=False
            )
            
            allocated_node = self.gpu_manager.allocate_gpu_resources(gpu_config)
            
            if allocated_node:
                logger.info(f"✅ GPU资源分配成功: {allocated_node}")
                
                # 清理资源
                self.gpu_manager.cleanup_gpu_resources(allocated_node, gpu_config.gpu_count)
                logger.info(f"✅ GPU资源清理成功: {allocated_node}")
                return True
            else:
                logger.warning("⚠️ GPU资源分配失败，可能是资源不足")
                return False
                
        except Exception as e:
            logger.error(f"❌ GPU资源分配测试失败: {e}")
            return False
    
    def test_tensorflow_gpu_setup(self):
        """测试TensorFlow GPU设置"""
        logger.info("=== 测试TensorFlow GPU设置 ===")
        
        try:
            gpu_config = TrainingGPUConfig(
                gpu_count=1,
                gpu_type='V100',
                memory_gb=16.0,
                distributed_training=False
            )
            
            tf_config = self.tensorflow_gpu.setup_tensorflow_gpu(gpu_config)
            logger.info(f"TensorFlow GPU配置: {tf_config}")
            
            if tf_config:
                logger.info("✅ TensorFlow GPU设置成功")
                return True
            else:
                logger.warning("⚠️ TensorFlow GPU设置失败，可能是TensorFlow未安装")
                return False
                
        except Exception as e:
            logger.error(f"❌ TensorFlow GPU设置测试失败: {e}")
            return False
    
    def test_pytorch_gpu_setup(self):
        """测试PyTorch GPU设置"""
        logger.info("=== 测试PyTorch GPU设置 ===")
        
        try:
            gpu_config = TrainingGPUConfig(
                gpu_count=1,
                gpu_type='V100',
                memory_gb=16.0,
                distributed_training=False
            )
            
            pytorch_config = self.pytorch_gpu.setup_pytorch_gpu(gpu_config)
            logger.info(f"PyTorch GPU配置: {pytorch_config}")
            
            if pytorch_config:
                logger.info("✅ PyTorch GPU设置成功")
                return True
            else:
                logger.warning("⚠️ PyTorch GPU设置失败，可能是PyTorch未安装")
                return False
                
        except Exception as e:
            logger.error(f"❌ PyTorch GPU设置测试失败: {e}")
            return False
    
    def test_distributed_training_setup(self):
        """测试分布式训练设置"""
        logger.info("=== 测试分布式训练设置 ===")
        
        try:
            gpu_config = TrainingGPUConfig(
                gpu_count=2,
                gpu_type='V100',
                memory_gb=32.0,
                distributed_training=True
            )
            
            # 测试分布式训练配置
            distributed_config = self.gpu_manager.setup_distributed_training(gpu_config)
            logger.info(f"分布式训练配置: {distributed_config}")
            
            if distributed_config:
                logger.info("✅ 分布式训练设置成功")
                return True
            else:
                logger.warning("⚠️ 分布式训练设置失败，可能是GPU资源不足")
                return False
                
        except Exception as e:
            logger.error(f"❌ 分布式训练设置测试失败: {e}")
            return False
    
    def test_gpu_monitoring(self):
        """测试GPU监控功能"""
        logger.info("=== 测试GPU监控功能 ===")
        
        try:
            # 获取GPU监控数据
            monitoring_data = self.gpu_manager.get_gpu_monitoring_data()
            
            logger.info(f"GPU监控数据: {monitoring_data}")
            
            # 检查监控数据的关键字段
            required_fields = ['utilization', 'available_nodes', 'total_nodes', 'timestamp']
            missing_fields = [field for field in required_fields if field not in monitoring_data]
            
            if missing_fields:
                logger.warning(f"⚠️ 监控数据缺少字段: {missing_fields}")
                return False
            else:
                logger.info("✅ GPU监控功能正常")
                return True
                
        except Exception as e:
            logger.error(f"❌ GPU监控功能测试失败: {e}")
            return False
    
    def run_all_tests(self):
        """运行所有测试"""
        logger.info("🚀 开始GPU资源管理集成测试")
        logger.info("=" * 50)
        
        test_results = []
        
        # 运行各项测试
        tests = [
            ("GPU管理器初始化", self.test_gpu_manager_initialization),
            ("GPU资源状态查询", self.test_gpu_resource_status),
            ("GPU节点发现", self.test_gpu_node_discovery),
            ("GPU资源需求验证", self.test_gpu_resource_validation),
            ("GPU资源分配", self.test_gpu_resource_allocation),
            ("TensorFlow GPU设置", self.test_tensorflow_gpu_setup),
            ("PyTorch GPU设置", self.test_pytorch_gpu_setup),
            ("分布式训练设置", self.test_distributed_training_setup),
            ("GPU监控功能", self.test_gpu_monitoring),
        ]
        
        for test_name, test_func in tests:
            try:
                result = test_func()
                test_results.append((test_name, result))
                logger.info(f"{'✅' if result else '❌'} {test_name}: {'通过' if result else '失败'}")
            except Exception as e:
                logger.error(f"❌ {test_name}: 测试异常 - {e}")
                test_results.append((test_name, False))
        
        # 统计测试结果
        passed_tests = sum(1 for _, result in test_results if result)
        total_tests = len(test_results)
        
        logger.info("=" * 50)
        logger.info(f"📊 测试结果统计:")
        logger.info(f"  总测试数: {total_tests}")
        logger.info(f"  通过测试: {passed_tests}")
        logger.info(f"  失败测试: {total_tests - passed_tests}")
        logger.info(f"  通过率: {passed_tests/total_tests*100:.1f}%")
        
        if passed_tests == total_tests:
            logger.info("🎉 所有测试通过！GPU资源管理集成成功")
        else:
            logger.warning("⚠️ 部分测试失败，请检查GPU资源管理器配置")
        
        return passed_tests == total_tests


async def main():
    """主函数"""
    logger.info("开始GPU资源管理集成测试")
    
    # 创建测试实例
    test = GPUResourceIntegrationTest()
    
    # 运行所有测试
    success = test.run_all_tests()
    
    if success:
        logger.info("✅ GPU资源管理集成测试完成")
        return 0
    else:
        logger.error("❌ GPU资源管理集成测试失败")
        return 1


if __name__ == "__main__":
    # 运行测试
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 
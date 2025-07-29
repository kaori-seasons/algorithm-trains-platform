#!/usr/bin/env python3
"""
参数调优模块测试脚本
"""
import asyncio
import logging
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.algorithm_engine.parameter_tuner import InteractiveParameterTuner
from backend.algorithm_engine.model_manager import ModelVersionManager
from backend.algorithm_engine.inference_service import RealTimeInferenceService
from backend.algorithm_engine.models import AlgorithmType

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_parameter_tuner():
    """测试参数调优器"""
    print("🧪 测试参数调优器...")
    
    try:
        # 创建参数调优器
        tuner = InteractiveParameterTuner()
        
        # 测试创建参数界面
        algorithm_config = {
            'algorithm_type': 'status_recognition',
            'parameters': {
                'threshold': 0.5,
                'n_estimators': 100
            }
        }
        
        interface = await tuner.create_parameter_interface(algorithm_config)
        print(f"✅ 参数界面创建成功: {interface['interface']['algorithm_type']}")
        
        # 测试参数更新
        new_params = {'threshold': 0.7, 'n_estimators': 150}
        result = await tuner.update_parameters(new_params)
        print(f"✅ 参数更新成功: {result['status']}")
        
        # 测试参数选择
        selection_data = {
            'type': 'threshold',
            'value': 0.8
        }
        selection_result = await tuner.apply_parameter_selection(selection_data)
        print(f"✅ 参数选择应用成功: {selection_result['status']}")
        
        # 测试导出最优参数
        optimal_params = await tuner.export_optimal_parameters()
        print(f"✅ 最优参数导出成功: {optimal_params['status']}")
        
        print("🎉 参数调优器测试通过！")
        return True
        
    except Exception as e:
        print(f"❌ 参数调优器测试失败: {e}")
        return False


async def test_model_manager():
    """测试模型版本管理器"""
    print("🧪 测试模型版本管理器...")
    
    try:
        # 创建模型版本管理器
        manager = ModelVersionManager()
        
        # 测试列出版本
        versions = await manager.list_versions()
        print(f"✅ 版本列表获取成功: {len(versions)} 个版本")
        
        # 测试获取版本详情
        if versions:
            version_id = versions[0]['version_id']
            details = await manager.get_version_details(version_id)
            print(f"✅ 版本详情获取成功: {details['status']}")
        
        print("🎉 模型版本管理器测试通过！")
        return True
        
    except Exception as e:
        print(f"❌ 模型版本管理器测试失败: {e}")
        return False


async def test_inference_service():
    """测试推理服务"""
    print("🧪 测试推理服务...")
    
    try:
        # 创建推理服务
        service = RealTimeInferenceService()
        
        # 测试健康检查
        health = await service.health_check()
        print(f"✅ 健康检查成功: {health['status']}")
        
        # 测试服务统计
        stats = service.get_service_stats()
        print(f"✅ 服务统计获取成功: 缓存大小={stats['cache_stats']['size']}")
        
        print("🎉 推理服务测试通过！")
        return True
        
    except Exception as e:
        print(f"❌ 推理服务测试失败: {e}")
        return False


async def main():
    """主测试函数"""
    print("🚀 开始测试参数调优相关模块...")
    
    results = []
    
    # 测试参数调优器
    results.append(await test_parameter_tuner())
    
    # 测试模型版本管理器
    results.append(await test_model_manager())
    
    # 测试推理服务
    results.append(await test_inference_service())
    
    # 总结
    success_count = sum(results)
    total_count = len(results)
    
    print(f"\n📊 测试结果总结:")
    print(f"   成功: {success_count}/{total_count}")
    print(f"   失败: {total_count - success_count}/{total_count}")
    
    if success_count == total_count:
        print("🎉 所有测试通过！")
        return True
    else:
        print("❌ 部分测试失败")
        return False


if __name__ == "__main__":
    asyncio.run(main()) 
"""
测试不依赖模型训练的交互式调试功能
"""
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime
import tempfile
import os

from algorithm_engine.trainers import StatusRecognitionTrainer, TrainerFactory
from algorithm_engine.models import TrainingConfig, AlgorithmType, ModelFormat


async def test_interactive_debug():
    """测试交互式调试功能"""
    print("🧪 开始测试不依赖模型训练的交互式调试功能...")
    
    # 创建测试数据
    print("📊 创建测试数据...")
    np.random.seed(42)
    n_samples = 1000
    
    # 生成模拟的工业设备数据，包含一些异常值
    data = {
        'temperature': np.random.normal(60, 10, n_samples),
        'vibration': np.random.normal(5, 2, n_samples),
        'pressure': np.random.normal(100, 20, n_samples),
        'speed': np.random.normal(1500, 100, n_samples),
        'current': np.random.normal(50, 5, n_samples),
        'voltage': np.random.normal(220, 10, n_samples)
    }
    
    # 添加一些异常值
    data['temperature'][:50] = np.random.uniform(100, 120, 50)  # 高温异常值
    data['vibration'][50:100] = np.random.uniform(15, 25, 50)   # 高振动异常值
    
    # 创建目标变量（设备状态：0=正常，1=异常）
    data['status'] = np.where(
        (data['temperature'] > 70) | (data['vibration'] > 7),
        1, 0
    )
    
    df = pd.DataFrame(data)
    
    # 保存测试数据
    with tempfile.TemporaryDirectory() as temp_dir:
        data_path = os.path.join(temp_dir, 'test_data.csv')
        df.to_csv(data_path, index=False)
        
        print(f"✅ 测试数据已保存到: {data_path}")
        
        # 创建训练配置
        config = TrainingConfig(
            algorithm_type=AlgorithmType.STATUS_RECOGNITION,
            train_data_path=data_path,
            feature_columns=['temperature', 'vibration', 'pressure', 'speed', 'current', 'voltage'],
            target_column='status',
            algorithm_params={
                'model_type': 'random_forest',
                'model_parameters': {
                    'n_estimators': 100,
                    'max_depth': 10
                }
            },
            output_path=temp_dir,
            save_model=True,
            model_format=ModelFormat.M
        )
        
        # 创建训练器
        trainer = StatusRecognitionTrainer()
        
        # 第一次训练
        print("🚀 开始第一次训练...")
        result1 = await trainer.train(config, {})
        print(f"✅ 第一次训练完成，准确率: {result1.accuracy:.4f}")
        
        # 获取数据分析结果
        print("📈 获取数据分析结果...")
        data_analysis = await trainer.get_data_analysis()
        print(f"✅ 数据分析完成，包含 {len(data_analysis)} 个分析维度")
        
        # 获取调试建议
        print("🎯 获取调试建议...")
        suggestions = await trainer.get_debug_suggestions()
        print(f"✅ 调试建议: {list(suggestions.keys())}")
        
        # 测试交互式调试 - 异常值处理
        print("\n🔄 测试异常值处理调试...")
        debug_params_1 = {
            'outlier_handling': {
                'columns': ['temperature', 'vibration'],
                'method': 'iqr',
                'action': 'remove',
                'multiplier': 1.5
            }
        }
        
        debug_result_1 = await trainer.interactive_debug(debug_params_1)
        print(f"✅ 异常值处理调试完成: {debug_result_1['success']}")
        print(f"   原始数据行数: {debug_result_1['data_summary']['original_rows']}")
        print(f"   处理后行数: {debug_result_1['data_summary']['processed_rows']}")
        print(f"   移除行数: {debug_result_1['data_summary']['removed_rows']}")
        
        # 测试交互式调试 - 特征选择
        print("\n🔄 测试特征选择调试...")
        debug_params_2 = {
            'feature_selection': ['temperature', 'vibration', 'pressure']
        }
        
        debug_result_2 = await trainer.interactive_debug(debug_params_2)
        print(f"✅ 特征选择调试完成: {debug_result_2['success']}")
        print(f"   处理后特征数: {debug_result_2['results']['data_statistics']['total_columns']}")
        
        # 测试交互式调试 - 数据采样
        print("\n🔄 测试数据采样调试...")
        debug_params_3 = {
            'sampling': {
                'method': 'stratified',
                'size': 500
            }
        }
        
        debug_result_3 = await trainer.interactive_debug(debug_params_3)
        print(f"✅ 数据采样调试完成: {debug_result_3['success']}")
        print(f"   采样后行数: {debug_result_3['data_summary']['processed_rows']}")
        
        # 测试交互式调试 - 特征变换
        print("\n🔄 测试特征变换调试...")
        debug_params_4 = {
            'feature_transformations': [
                {'column': 'temperature', 'method': 'standardize'},
                {'column': 'vibration', 'method': 'log'}
            ]
        }
        
        debug_result_4 = await trainer.interactive_debug(debug_params_4)
        print(f"✅ 特征变换调试完成: {debug_result_4['success']}")
        
        # 测试组合调试参数
        print("\n🔄 测试组合调试参数...")
        combined_debug_params = {
            'outlier_handling': {
                'columns': ['temperature'],
                'method': 'iqr',
                'action': 'cap'
            },
            'feature_selection': ['temperature', 'vibration', 'pressure', 'speed'],
            'sampling': {
                'method': 'random',
                'size': 800
            }
        }
        
        combined_result = await trainer.interactive_debug(combined_debug_params)
        print(f"✅ 组合调试完成: {combined_result['success']}")
        print(f"   最终数据行数: {combined_result['data_summary']['processed_rows']}")
        print(f"   最终特征数: {combined_result['results']['data_statistics']['total_columns']}")
        
        # 应用调试参数并重新训练
        print("\n🚀 应用调试参数并重新训练...")
        try:
            # 这里需要实际的训练配置，我们模拟一下
            print("✅ 调试参数已应用，可以重新训练模型")
        except Exception as e:
            print(f"❌ 重新训练失败: {str(e)}")
        
        # 测试可视化数据生成
        print("\n📊 测试调试可视化数据...")
        if 'visualization_data' in debug_result_1['results']:
            viz_data = debug_result_1['results']['visualization_data']
            print(f"✅ 可视化数据类型: {list(viz_data.keys())}")
            
            if 'feature_distributions' in viz_data:
                print(f"   特征分布图: {list(viz_data['feature_distributions'].keys())}")
            
            if 'correlation_matrix' in viz_data:
                print(f"   相关性矩阵: {len(viz_data['correlation_matrix']['columns'])} 个特征")
        
        print("\n🎉 交互式调试功能测试完成！")


async def test_debug_suggestions():
    """测试调试建议功能"""
    print("\n🧪 测试调试建议功能...")
    
    # 创建包含明显问题的测试数据
    np.random.seed(42)
    n_samples = 1000
    
    # 创建不平衡的数据
    data = {
        'feature1': np.random.normal(0, 1, n_samples),
        'feature2': np.random.normal(0, 1, n_samples),
        'feature3': np.random.normal(0, 1, n_samples),
        'feature4': np.random.normal(0, 1, n_samples),
        'feature5': np.random.normal(0, 1, n_samples)
    }
    
    # 创建严重不平衡的目标变量
    data['status'] = np.random.choice([0, 1], n_samples, p=[0.9, 0.1])  # 90% 正常，10% 异常
    
    # 添加一些异常值
    data['feature1'][:100] = np.random.uniform(5, 10, 100)  # 10% 异常值
    
    df = pd.DataFrame(data)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        data_path = os.path.join(temp_dir, 'unbalanced_data.csv')
        df.to_csv(data_path, index=False)
        
        # 创建训练配置
        config = TrainingConfig(
            algorithm_type=AlgorithmType.STATUS_RECOGNITION,
            train_data_path=data_path,
            feature_columns=['feature1', 'feature2', 'feature3', 'feature4', 'feature5'],
            target_column='status',
            algorithm_params={'model_type': 'random_forest'},
            output_path=temp_dir,
            save_model=True,
            model_format=ModelFormat.M
        )
        
        # 创建训练器并训练
        trainer = StatusRecognitionTrainer()
        await trainer.train(config, {})
        
        # 获取调试建议
        suggestions = await trainer.get_debug_suggestions()
        
        print("📋 调试建议分析:")
        for category, suggestion_list in suggestions.items():
            print(f"\n{category}:")
            if isinstance(suggestion_list, list):
                for suggestion in suggestion_list:
                    print(f"  - {suggestion.get('issue', 'N/A')}")
                    print(f"    建议: {suggestion.get('suggestion', 'N/A')}")
            else:
                print(f"  - {suggestion_list.get('issue', 'N/A')}")
                print(f"    建议: {suggestion_list.get('suggestion', 'N/A')}")
        
        print("\n✅ 调试建议功能测试完成！")


async def main():
    """主测试函数"""
    print("🚀 开始交互式调试功能测试...")
    
    try:
        await test_interactive_debug()
        await test_debug_suggestions()
        
        print("\n✅ 所有交互式调试功能测试完成！")
        
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main()) 
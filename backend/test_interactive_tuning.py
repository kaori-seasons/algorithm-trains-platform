"""
测试交互式调参功能
"""
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime
import tempfile
import os

from algorithm_engine.trainers import StatusRecognitionTrainer, TrainerFactory
from algorithm_engine.models import TrainingConfig, AlgorithmType, ModelFormat


async def test_interactive_tuning():
    """测试交互式调参功能"""
    print("🧪 开始测试交互式调参功能...")
    
    # 创建测试数据
    print("📊 创建测试数据...")
    np.random.seed(42)
    n_samples = 1000
    
    # 生成模拟的工业设备数据
    data = {
        'temperature': np.random.normal(60, 10, n_samples),
        'vibration': np.random.normal(5, 2, n_samples),
        'pressure': np.random.normal(100, 20, n_samples),
        'speed': np.random.normal(1500, 100, n_samples),
        'current': np.random.normal(50, 5, n_samples),
        'voltage': np.random.normal(220, 10, n_samples)
    }
    
    # 创建目标变量（设备状态：0=正常，1=异常）
    # 基于温度和振动的组合判断异常
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
            model_format=ModelFormat.M  # 使用.m格式保存
        )
        
        # 创建训练器
        trainer = StatusRecognitionTrainer()
        
        # 第一次训练
        print("🚀 开始第一次训练...")
        result1 = await trainer.train(config, {})
        print(f"✅ 第一次训练完成，准确率: {result1.accuracy:.4f}")
        
        # 获取可视化数据
        print("📈 获取可视化数据...")
        viz_data = await trainer.get_visualization_data()
        print(f"✅ 可视化数据类型: {list(viz_data.keys())}")
        
        # 获取最优参数建议
        print("🎯 获取最优参数建议...")
        optimal_params = await trainer.get_optimal_parameters()
        print(f"✅ 最优参数建议: {optimal_params}")
        
        # 交互式调参 - 调整特征选择
        print("🔄 开始交互式调参...")
        interactive_params = {
            'selected_features': [0, 1, 2],  # 只选择前3个特征
            'model_parameters': {
                'n_estimators': 150,
                'max_depth': 15
            }
        }
        
        result2 = await trainer.interactive_tuning(config, interactive_params)
        print(f"✅ 交互式调参完成，准确率: {result2.accuracy:.4f}")
        
        # 比较结果
        print("\n📊 训练结果对比:")
        print(f"第一次训练 - 准确率: {result1.accuracy:.4f}, 精确率: {result1.precision:.4f}")
        print(f"调参后训练 - 准确率: {result2.accuracy:.4f}, 精确率: {result2.precision:.4f}")
        
        # 测试不同模型类型
        print("\n🧪 测试不同模型类型...")
        model_types = ['svm', 'logistic_regression', 'decision_tree']
        
        for model_type in model_types:
            print(f"测试模型: {model_type}")
            config.algorithm_params['model_type'] = model_type
            
            try:
                result = await trainer.train(config, {})
                print(f"  ✅ {model_type} - 准确率: {result.accuracy:.4f}")
            except Exception as e:
                print(f"  ❌ {model_type} - 失败: {str(e)}")
        
        # 测试.m格式模型保存
        print("\n💾 测试.m格式模型保存...")
        if os.path.exists(os.path.join(temp_dir, 'model.m')):
            print("✅ .m格式模型文件保存成功")
            
            # 测试模型加载
            import pickle
            with open(os.path.join(temp_dir, 'model.m'), 'rb') as f:
                loaded_model = pickle.load(f)
            print("✅ .m格式模型文件加载成功")
            
            # 测试预测
            test_data = df[['temperature', 'vibration', 'pressure', 'speed', 'current', 'voltage']].iloc[:5]
            predictions = loaded_model.predict(test_data)
            print(f"✅ 模型预测测试成功，预测结果: {predictions}")
        else:
            print("❌ .m格式模型文件保存失败")
        
        print("\n🎉 交互式调参功能测试完成！")


async def test_parameter_generation():
    """测试参数生成功能"""
    print("\n🧪 测试参数生成功能...")
    
    # 测试自动参数生成
    trainer = StatusRecognitionTrainer()
    params = await trainer.generate_parameters({})
    print(f"✅ 自动参数生成: {params}")
    
    # 测试随机参数生成
    from algorithm_engine.trainers import ParameterGenerator
    random_params = ParameterGenerator.generate_random_parameters(AlgorithmType.STATUS_RECOGNITION)
    print(f"✅ 随机参数生成: {random_params}")


async def main():
    """主测试函数"""
    print("🚀 开始算法引擎测试...")
    
    try:
        await test_interactive_tuning()
        await test_parameter_generation()
        
        print("\n✅ 所有测试完成！")
        
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main()) 
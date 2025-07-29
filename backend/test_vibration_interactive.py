"""
测试振动算法的交互式训练功能
"""
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import os

from algorithm_engine.trainers import VibrationTrainer, TrainerFactory
from algorithm_engine.models import TrainingConfig, AlgorithmType, ModelFormat


async def test_vibration_interactive_training():
    """测试振动算法交互式训练功能"""
    print("🧪 开始测试振动算法交互式训练功能...")
    
    # 创建模拟振动数据
    print("📊 创建模拟振动数据...")
    np.random.seed(42)
    n_samples = 5000
    sampling_rate = 1000  # Hz
    
    # 生成时间戳
    start_time = datetime.now() - timedelta(hours=1)
    timestamps = [start_time + timedelta(seconds=i/sampling_rate) for i in range(n_samples)]
    
    # 生成模拟振动信号
    time = np.arange(n_samples) / sampling_rate
    
    # 正常振动信号（包含一些噪声）
    normal_vibration = 0.5 * np.sin(2 * np.pi * 50 * time) + 0.1 * np.random.normal(0, 1, n_samples)
    
    # 添加一些异常振动（模拟故障）
    anomaly_start = int(n_samples * 0.7)
    anomaly_vibration = normal_vibration.copy()
    anomaly_vibration[anomaly_start:] += 2.0 * np.sin(2 * np.pi * 100 * time[anomaly_start:])  # 高频异常
    
    # 生成转速数据
    speed = np.ones(n_samples) * 1500  # 1500 RPM
    speed[anomaly_start:] += np.random.normal(0, 50, n_samples - anomaly_start)  # 转速波动
    
    # 创建数据框
    data = {
        'timestamp': timestamps,
        'x_accel': anomaly_vibration,
        'y_accel': 0.3 * np.sin(2 * np.pi * 30 * time) + 0.05 * np.random.normal(0, 1, n_samples),
        'z_accel': 0.2 * np.cos(2 * np.pi * 40 * time) + 0.05 * np.random.normal(0, 1, n_samples),
        'speed': speed,
        'signal_quality': np.random.uniform(0.8, 1.0, n_samples)
    }
    
    # 添加目标变量（异常检测标签）
    data['status'] = np.zeros(n_samples)
    data['status'][anomaly_start:] = 1  # 1表示异常
    
    df = pd.DataFrame(data)
    
    # 保存测试数据
    with tempfile.TemporaryDirectory() as temp_dir:
        data_path = os.path.join(temp_dir, 'vibration_data.csv')
        df.to_csv(data_path, index=False)
        
        print(f"✅ 振动数据已保存到: {data_path}")
        print(f"   数据形状: {df.shape}")
        print(f"   异常样本比例: {np.mean(data['status']):.2%}")
        
        # 创建振动算法配置
        vibration_config = {
            'sampling_rate': sampling_rate,
            'data_type': 'float32',
            'model_type': 'isolation_forest',
            'contamination': 0.1,
            'frequency_range': [0, 500],
            'time_domain': {
                'rms_enabled': True,
                'peak_enabled': True,
                'crest_factor_enabled': True,
                'kurtosis_enabled': True
            },
            'frequency_domain': {
                'spectrum_enabled': True,
                'harmonic_analysis': True,
                'sideband_analysis': True,
                'envelope_analysis': True
            }
        }
        
        # 创建训练配置
        config = TrainingConfig(
            algorithm_type=AlgorithmType.VIBRATION,
            train_data_path=data_path,
            feature_columns=['x_accel', 'y_accel', 'z_accel', 'speed'],
            target_column='status',
            algorithm_params={'vibration_config': vibration_config},
            output_path=temp_dir,
            save_model=True,
            model_format=ModelFormat.M
        )
        
        # 创建振动训练器
        trainer = VibrationTrainer()
        
        # 第一次训练
        print("\n🚀 开始第一次振动模型训练...")
        result1 = await trainer.train(config, {})
        print(f"✅ 第一次训练完成，任务ID: {result1.task_id}")
        print(f"   训练时长: {result1.duration:.2f}秒")
        
        # 获取振动分析结果
        print("\n📈 获取振动分析结果...")
        vibration_analysis = await trainer.get_vibration_analysis()
        print(f"✅ 振动分析完成，包含 {len(vibration_analysis)} 个分析维度")
        
        # 获取振动可视化数据
        print("\n📊 获取振动可视化数据...")
        visualization_data = await trainer.get_vibration_visualization()
        print(f"✅ 可视化数据生成完成，包含 {len(visualization_data)} 种图表类型")
        
        # 测试交互式训练 - 频率过滤
        print("\n🔄 测试频率过滤交互式训练...")
        frequency_filter_params = {
            'frequency_filtering': {
                'enabled': True,
                'low_freq_cutoff': 10,
                'high_freq_cutoff': 200,
                'sampling_rate': sampling_rate,
                'bandpass_filters': [
                    {'name': 'bearing_freq', 'center': 50, 'bandwidth': 10},
                    {'name': 'gear_freq', 'center': 100, 'bandwidth': 20}
                ]
            }
        }
        
        filter_result = await trainer.interactive_training(frequency_filter_params)
        print(f"✅ 频率过滤训练完成: {filter_result['success']}")
        if filter_result['success']:
            print(f"   处理后数据形状: {filter_result['result']['processed_data_shape']}")
            print(f"   特征形状: {filter_result['result']['features_shape']}")
        
        # 测试交互式训练 - 振幅阈值
        print("\n🔄 测试振幅阈值交互式训练...")
        amplitude_params = {
            'amplitude_thresholds': {
                'warning_level': 0.3,
                'alarm_level': 0.8,
                'critical_level': 1.5
            }
        }
        
        amplitude_result = await trainer.interactive_training(amplitude_params)
        print(f"✅ 振幅阈值训练完成: {amplitude_result['success']}")
        
        # 测试交互式训练 - 数据选择
        print("\n🔄 测试数据选择交互式训练...")
        data_selection_params = {
            'data_selection': {
                'time_range': {
                    'start_time': timestamps[1000].strftime('%Y-%m-%d %H:%M:%S'),
                    'end_time': timestamps[4000].strftime('%Y-%m-%d %H:%M:%S')
                },
                'speed_range': {
                    'min_speed': 1400,
                    'max_speed': 1600
                },
                'quality_filters': {
                    'min_signal_quality': 0.85
                }
            }
        }
        
        selection_result = await trainer.interactive_training(data_selection_params)
        print(f"✅ 数据选择训练完成: {selection_result['success']}")
        if selection_result['success']:
            print(f"   选择后数据形状: {selection_result['result']['processed_data_shape']}")
        
        # 测试交互式训练 - 实时配置
        print("\n🔄 测试实时配置交互式训练...")
        real_time_params = {
            'real_time_adjustment': {
                'adaptive_thresholds': {
                    'enabled': True,
                    'learning_rate': 0.1,
                    'update_frequency': 'hourly'
                },
                'dynamic_filtering': {
                    'enabled': True,
                    'auto_adjust_bandwidth': True,
                    'noise_adaptation': True
                },
                'feature_weights': {
                    'rms_weight': 0.4,
                    'peak_weight': 0.3,
                    'crest_factor_weight': 0.2,
                    'kurtosis_weight': 0.1
                }
            }
        }
        
        real_time_result = await trainer.interactive_training(real_time_params)
        print(f"✅ 实时配置训练完成: {real_time_result['success']}")
        
        # 获取实时配置
        real_time_config = await trainer.get_real_time_config()
        print(f"   实时配置: {list(real_time_config.keys())}")
        
        # 测试组合参数训练
        print("\n🔄 测试组合参数交互式训练...")
        combined_params = {
            'frequency_filtering': {
                'enabled': True,
                'low_freq_cutoff': 20,
                'high_freq_cutoff': 300
            },
            'amplitude_thresholds': {
                'warning_level': 0.4,
                'alarm_level': 1.0,
                'critical_level': 2.0
            },
            'data_selection': {
                'time_range': {
                    'start_time': timestamps[500].strftime('%Y-%m-%d %H:%M:%S'),
                    'end_time': timestamps[4500].strftime('%Y-%m-%d %H:%M:%S')
                }
            },
            'real_time_adjustment': {
                'adaptive_thresholds': {
                    'enabled': True,
                    'learning_rate': 0.05
                }
            }
        }
        
        combined_result = await trainer.interactive_training(combined_params)
        print(f"✅ 组合参数训练完成: {combined_result['success']}")
        
        # 测试可视化数据更新
        print("\n📊 测试可视化数据更新...")
        updated_visualization = await trainer.get_vibration_visualization()
        print(f"✅ 可视化数据更新完成")
        print(f"   时域波形图: {len(updated_visualization.get('time_domain_waveforms', {}))} 个通道")
        print(f"   频谱图: {len(updated_visualization.get('frequency_spectrums', {}))} 个通道")
        print(f"   特征分布图: {len(updated_visualization.get('feature_distributions', {}))} 个特征")
        
        # 测试振动分析更新
        print("\n📈 测试振动分析更新...")
        updated_analysis = await trainer.get_vibration_analysis()
        print(f"✅ 振动分析更新完成")
        print(f"   信号质量分析: {len(updated_analysis.get('signal_quality', {}))} 个指标")
        print(f"   频率分析: {len(updated_analysis.get('frequency_analysis', {}))} 个通道")
        print(f"   趋势分析: {len(updated_analysis.get('trend_analysis', {}))} 个趋势")
        print(f"   异常检测: {len(updated_analysis.get('anomaly_detection', {}))} 个异常指标")
        
        print("\n🎉 振动算法交互式训练功能测试完成！")


async def test_vibration_parameter_generation():
    """测试振动算法参数生成"""
    print("\n🧪 测试振动算法参数生成...")
    
    trainer = VibrationTrainer()
    
    # 测试默认参数生成
    default_params = await trainer.generate_parameters({})
    print("📋 默认参数:")
    print(f"   算法类型: {default_params['algorithm_type']}")
    print(f"   采样率: {default_params['vibration_config']['sampling_rate']} Hz")
    print(f"   模型类型: {default_params['vibration_config']['model_type']}")
    print(f"   时域特征: {list(default_params['vibration_config']['time_domain'].keys())}")
    print(f"   频域特征: {list(default_params['vibration_config']['frequency_domain'].keys())}")
    
    # 测试自定义参数生成
    custom_config = {
        'sampling_rate': 2000,
        'frequency_range': [0, 1000],
        'model_type': 'one_class_svm'
    }
    
    custom_params = await trainer.generate_parameters(custom_config)
    print("\n📋 自定义参数:")
    print(f"   采样率: {custom_params['vibration_config']['sampling_rate']} Hz")
    print(f"   频率范围: {custom_params['vibration_config']['frequency_range']}")
    print(f"   模型类型: {custom_params['vibration_config']['model_type']}")
    
    print("✅ 振动算法参数生成测试完成！")


async def main():
    """主测试函数"""
    print("🚀 开始振动算法交互式训练测试...")
    
    try:
        await test_vibration_interactive_training()
        await test_vibration_parameter_generation()
        
        print("\n✅ 所有振动算法测试完成！")
        
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main()) 
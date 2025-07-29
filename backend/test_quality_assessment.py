"""
测试质量评估和重新预处理功能
"""
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import os

from quality_assessment.service import QualityAssessmentService, QualityIssue, QualityIssueType
from preprocessing.pipeline import PreprocessingPipeline
from annotation.service import AnnotationService, AnnotationSegment, AnnotationType, SpeedLevel


async def test_quality_assessment():
    """测试质量评估功能"""
    print("🧪 开始测试质量评估功能...")
    
    # 创建模拟振动数据（包含质量问题）
    print("📊 创建包含质量问题的模拟振动数据...")
    np.random.seed(42)
    n_samples = 5000
    sampling_rate = 1000  # Hz
    
    # 生成时间戳
    start_time = datetime.now() - timedelta(hours=1)
    timestamps = [start_time + timedelta(seconds=i/sampling_rate) for i in range(n_samples)]
    
    # 生成时间序列
    time = np.arange(n_samples) / sampling_rate
    
    # 正常振动信号
    normal_vibration = 0.5 * np.sin(2 * np.pi * 50 * time) + 0.1 * np.random.normal(0, 1, n_samples)
    
    # 添加质量问题
    # 1. 波形不连续（跳跃）
    jump_start = int(n_samples * 0.3)
    jump_end = int(n_samples * 0.35)
    normal_vibration[jump_start:jump_end] += 3.0  # 添加跳跃
    
    # 2. 噪声污染
    noise_start = int(n_samples * 0.6)
    noise_end = int(n_samples * 0.7)
    normal_vibration[noise_start:noise_end] += 2.0 * np.random.normal(0, 1, noise_end - noise_start)
    
    # 3. 数据缺失
    missing_start = int(n_samples * 0.8)
    missing_end = int(n_samples * 0.85)
    normal_vibration[missing_start:missing_end] = np.nan
    
    # 4. 异常振幅
    anomaly_start = int(n_samples * 0.4)
    anomaly_end = int(n_samples * 0.45)
    normal_vibration[anomaly_start:anomaly_end] *= 5.0  # 放大振幅
    
    # 创建数据框
    data = {
        'timestamp': timestamps,
        'x_accel': normal_vibration,
        'y_accel': 0.3 * np.sin(2 * np.pi * 30 * time) + 0.05 * np.random.normal(0, 1, n_samples),
        'z_accel': 0.2 * np.cos(2 * np.pi * 40 * time) + 0.05 * np.random.normal(0, 1, n_samples),
        'speed': np.ones(n_samples) * 1500 + np.random.normal(0, 50, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # 保存测试数据
    with tempfile.TemporaryDirectory() as temp_dir:
        data_path = os.path.join(temp_dir, 'quality_test_data.csv')
        df.to_csv(data_path, index=False)
        
        print(f"✅ 测试数据已保存到: {data_path}")
        print(f"   数据形状: {df.shape}")
        
        # 创建质量评估服务
        quality_service = QualityAssessmentService()
        
        # 配置评估参数
        assessment_config = {
            'sampling_rate': sampling_rate,
            'jump_threshold': 2.0,
            'min_snr': 15.0,
            'max_missing_ratio': 0.1,
            'frequency_window_size': 500
        }
        
        # 执行质量评估
        print("\n🔍 执行质量评估...")
        assessment = await quality_service.assess_vibration_quality(df, assessment_config)
        
        print(f"✅ 质量评估完成")
        print(f"   总体质量分数: {assessment.overall_score:.2f}")
        print(f"   质量等级: {assessment.quality_level.value}")
        print(f"   发现问题数量: {len(assessment.issues)}")
        
        # 显示各通道质量分数
        print("\n📊 各通道质量分数:")
        for channel, score in assessment.channel_scores.items():
            print(f"   {channel}: {score:.2f}")
        
        # 显示问题详情
        print("\n⚠️ 发现的质量问题:")
        for i, issue in enumerate(assessment.issues):
            print(f"   {i+1}. {issue.issue_type.value}: {issue.description}")
            print(f"      严重程度: {issue.severity:.2f}")
            print(f"      建议操作: {issue.suggested_action}")
            print(f"      影响通道: {issue.affected_channels}")
        
        print("\n🎉 质量评估功能测试完成！")


async def test_preprocessing_pipeline():
    """测试预处理管道功能"""
    print("\n🧪 开始测试预处理管道功能...")
    
    # 创建模拟数据（包含质量问题）
    np.random.seed(42)
    n_samples = 3000
    sampling_rate = 1000
    
    time = np.arange(n_samples) / sampling_rate
    
    # 创建包含问题的信号
    signal = 0.5 * np.sin(2 * np.pi * 50 * time) + 0.1 * np.random.normal(0, 1, n_samples)
    
    # 添加问题
    # 波形跳跃
    signal[int(n_samples * 0.3):int(n_samples * 0.35)] += 2.0
    
    # 噪声
    signal[int(n_samples * 0.6):int(n_samples * 0.7)] += 1.5 * np.random.normal(0, 1, 100)
    
    # 缺失数据
    signal[int(n_samples * 0.8):int(n_samples * 0.85)] = np.nan
    
    # 异常振幅
    signal[int(n_samples * 0.4):int(n_samples * 0.45)] *= 4.0
    
    data = {
        'timestamp': pd.date_range(start=datetime.now(), periods=n_samples, freq='1ms'),
        'x_accel': signal,
        'y_accel': 0.3 * np.sin(2 * np.pi * 30 * time) + 0.05 * np.random.normal(0, 1, n_samples),
        'speed': np.ones(n_samples) * 1500
    }
    
    df = pd.DataFrame(data)
    
    # 创建质量评估服务
    quality_service = QualityAssessmentService()
    
    # 执行质量评估
    print("🔍 执行质量评估...")
    assessment = await quality_service.assess_vibration_quality(df, {'sampling_rate': sampling_rate})
    
    print(f"   发现问题数量: {len(assessment.issues)}")
    
    # 创建预处理管道
    preprocessing_pipeline = PreprocessingPipeline()
    
    # 配置预处理参数
    preprocessing_config = {
        'sampling_rate': sampling_rate,
        'discontinuity_cutoff_freq': 100,
        'noise_low_cutoff': 10,
        'noise_high_cutoff': 500,
        'frequency_correction_window': 100
    }
    
    # 执行预处理
    print("\n🔄 执行数据预处理...")
    result = await preprocessing_pipeline.process_data(df, assessment.issues, preprocessing_config)
    
    print(f"✅ 预处理完成")
    print(f"   原始数据形状: {result.original_data.shape}")
    print(f"   处理后数据形状: {result.processed_data.shape}")
    print(f"   处理步骤数量: {len(result.steps)}")
    print(f"   处理耗时: {result.processing_time:.2f}秒")
    
    # 显示处理步骤
    print("\n📋 处理步骤:")
    for i, step in enumerate(result.steps):
        print(f"   {i+1}. {step.step_type.value}: {step.description}")
        print(f"      应用通道: {step.applied_channels}")
        print(f"      参数: {step.parameters}")
    
    # 显示质量改善
    print("\n📈 质量改善情况:")
    for channel, improvement in result.quality_improvement.items():
        print(f"   {channel}: {improvement:.2f}%")
    
    print("\n🎉 预处理管道功能测试完成！")


async def test_annotation_service():
    """测试标注服务功能"""
    print("\n🧪 开始测试标注服务功能...")
    
    # 创建模拟数据
    np.random.seed(42)
    n_samples = 2000
    sampling_rate = 1000
    
    time = np.arange(n_samples) / sampling_rate
    signal = 0.5 * np.sin(2 * np.pi * 50 * time) + 0.1 * np.random.normal(0, 1, n_samples)
    
    data = {
        'timestamp': pd.date_range(start=datetime.now(), periods=n_samples, freq='1ms'),
        'x_accel': signal,
        'y_accel': 0.3 * np.sin(2 * np.pi * 30 * time) + 0.05 * np.random.normal(0, 1, n_samples),
        'speed': np.ones(n_samples) * 1500
    }
    
    df = pd.DataFrame(data)
    
    # 保存测试数据
    with tempfile.TemporaryDirectory() as temp_dir:
        data_path = os.path.join(temp_dir, 'annotation_test_data.csv')
        df.to_csv(data_path, index=False)
        
        # 创建标注服务
        annotation_service = AnnotationService()
        
        # 创建标注任务
        print("📝 创建标注任务...")
        task_config = {
            'assigned_user': 'test_user',
            'description': '测试标注任务'
        }
        
        task_id = await annotation_service.create_annotation_task(data_path, task_config)
        print(f"   任务ID: {task_id}")
        
        # 添加转速标注
        print("\n🏷️ 添加转速标注...")
        speed_segment = AnnotationSegment(
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(minutes=10),
            annotation_type=AnnotationType.SPEED_RANGE,
            label=SpeedLevel.MEDIUM_SPEED.value,
            confidence=0.9,
            metadata={'speed_value': 1500}
        )
        
        success = await annotation_service.add_annotation_segment(task_id, speed_segment)
        print(f"   转速标注添加: {'成功' if success else '失败'}")
        
        # 添加质量标注
        print("\n🏷️ 添加质量标注...")
        quality_segment = AnnotationSegment(
            start_time=datetime.now() + timedelta(minutes=5),
            end_time=datetime.now() + timedelta(minutes=8),
            annotation_type=AnnotationType.QUALITY_MARK,
            label='good',
            confidence=0.8,
            metadata={'quality_score': 85}
        )
        
        success = await annotation_service.add_annotation_segment(task_id, quality_segment)
        print(f"   质量标注添加: {'成功' if success else '失败'}")
        
        # 获取标注数据
        print("\n📊 获取标注数据...")
        annotation_data = await annotation_service.get_annotation_data(task_id)
        print(f"   标注段数量: {len(annotation_data['segments'])}")
        print(f"   时间序列通道: {list(annotation_data['time_series_data'].keys())}")
        
        # 获取转速标注
        speed_annotations = await annotation_service.get_speed_annotations(task_id)
        print(f"   转速标注数量: {len(speed_annotations)}")
        
        # 获取质量标注
        quality_annotations = await annotation_service.get_quality_annotations(task_id)
        print(f"   质量标注数量: {len(quality_annotations)}")
        
        # 获取统计信息
        statistics = await annotation_service.get_annotation_statistics(task_id)
        print(f"\n📈 标注统计:")
        print(f"   总标注段: {statistics['total_segments']}")
        print(f"   类型分布: {statistics['type_distribution']}")
        print(f"   标签分布: {statistics['label_distribution']}")
        print(f"   平均置信度: {statistics['average_confidence']:.2f}")
        
        # 导出标注结果
        print("\n📤 导出标注结果...")
        export_data = await annotation_service.export_annotations(task_id, 'json')
        print(f"   导出数据长度: {len(export_data)} 字符")
        
        print("\n🎉 标注服务功能测试完成！")


async def test_integrated_workflow():
    """测试集成工作流"""
    print("\n🧪 开始测试集成工作流...")
    
    # 1. 创建包含质量问题的数据
    print("📊 步骤1: 创建测试数据...")
    np.random.seed(42)
    n_samples = 4000
    sampling_rate = 1000
    
    time = np.arange(n_samples) / sampling_rate
    signal = 0.5 * np.sin(2 * np.pi * 50 * time) + 0.1 * np.random.normal(0, 1, n_samples)
    
    # 添加质量问题
    signal[int(n_samples * 0.3):int(n_samples * 0.35)] += 2.0  # 跳跃
    signal[int(n_samples * 0.6):int(n_samples * 0.7)] += 1.5 * np.random.normal(0, 1, 100)  # 噪声
    signal[int(n_samples * 0.8):int(n_samples * 0.85)] = np.nan  # 缺失
    
    data = {
        'timestamp': pd.date_range(start=datetime.now(), periods=n_samples, freq='1ms'),
        'x_accel': signal,
        'y_accel': 0.3 * np.sin(2 * np.pi * 30 * time) + 0.05 * np.random.normal(0, 1, n_samples),
        'speed': np.ones(n_samples) * 1500
    }
    
    df = pd.DataFrame(data)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        data_path = os.path.join(temp_dir, 'workflow_test_data.csv')
        df.to_csv(data_path, index=False)
        
        # 2. 质量评估
        print("\n🔍 步骤2: 质量评估...")
        quality_service = QualityAssessmentService()
        assessment = await quality_service.assess_vibration_quality(df, {'sampling_rate': sampling_rate})
        
        print(f"   质量分数: {assessment.overall_score:.2f}")
        print(f"   问题数量: {len(assessment.issues)}")
        
        # 3. 数据预处理
        print("\n🔄 步骤3: 数据预处理...")
        preprocessing_pipeline = PreprocessingPipeline()
        result = await preprocessing_pipeline.process_data(df, assessment.issues, {'sampling_rate': sampling_rate})
        
        print(f"   处理步骤: {len(result.steps)}")
        print(f"   质量改善: {np.mean(list(result.quality_improvement.values())):.2f}%")
        
        # 4. 创建标注任务
        print("\n📝 步骤4: 创建标注任务...")
        annotation_service = AnnotationService()
        task_id = await annotation_service.create_annotation_task(data_path, {})
        
        # 5. 添加标注
        print("\n🏷️ 步骤5: 添加标注...")
        segment = AnnotationSegment(
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(minutes=5),
            annotation_type=AnnotationType.SPEED_RANGE,
            label=SpeedLevel.MEDIUM_SPEED.value,
            confidence=0.9
        )
        
        await annotation_service.add_annotation_segment(task_id, segment)
        
        # 6. 获取结果
        print("\n📊 步骤6: 获取结果...")
        annotation_data = await annotation_service.get_annotation_data(task_id)
        statistics = await annotation_service.get_annotation_statistics(task_id)
        
        print(f"   标注段: {statistics['total_segments']}")
        print(f"   时间序列通道: {len(annotation_data['time_series_data'])}")
        
        print("\n🎉 集成工作流测试完成！")


async def main():
    """主测试函数"""
    print("🚀 开始质量评估系统测试...")
    
    try:
        await test_quality_assessment()
        await test_preprocessing_pipeline()
        await test_annotation_service()
        await test_integrated_workflow()
        
        print("\n✅ 所有质量评估功能测试完成！")
        
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main()) 
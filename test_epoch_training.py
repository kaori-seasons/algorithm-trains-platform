"""
Epoch训练功能测试脚本
测试TensorFlow/PyTorch epoch轮次训练功能
"""
import asyncio
import logging
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.algorithm_engine.epoch_trainers import EpochDeepLearningTrainer, epoch_training_manager

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_epoch_training():
    """测试epoch训练功能"""
    print("🧪 开始测试Epoch训练功能...")
    
    try:
        # 创建训练器
        trainer = EpochDeepLearningTrainer()
        
        # 配置训练参数
        config = {
            "name": "测试深度学习模型",
            "algorithm_type": "deep_learning",
            "model_type": "mlp",
            "epochs": 10,  # 测试用较少的epoch
            "batch_size": 32,
            "learning_rate": 0.001,
            "hidden_units": [64, 32],
            "dropout_rate": 0.2,
            "early_stopping_patience": 5,
            "learning_rate_scheduler": "step",
            "feature_columns": ["feature1", "feature2", "feature3"],
            "target_column": "target",
            "output_path": "./test_models"
        }
        
        # 模拟数据
        data = {
            "train_data": "模拟训练数据",
            "train_labels": "模拟训练标签",
            "val_data": "模拟验证数据",
            "val_labels": "模拟验证标签"
        }
        
        print(f"📋 训练配置: {config}")
        print(f"📊 数据配置: {list(data.keys())}")
        
        # 启动训练
        print("🚀 启动Epoch训练...")
        result = await trainer.train_with_epochs(config, data, task_id="test_epoch_training")
        
        print(f"✅ 训练完成!")
        print(f"📈 训练结果: {result}")
        
        # 检查训练历史
        if trainer.epoch_history:
            print(f"📊 训练历史记录数: {len(trainer.epoch_history)}")
            print(f"🏆 最佳epoch: {trainer.early_stopping.get_best_epoch()}")
            
            # 显示最后几个epoch的指标
            print("\n📈 最后5个epoch的指标:")
            for metrics in trainer.epoch_history[-5:]:
                print(f"  Epoch {metrics.epoch}: "
                      f"Train Loss={metrics.train_loss:.4f}, "
                      f"Train Acc={metrics.train_accuracy:.4f}, "
                      f"Val Loss={metrics.val_loss:.4f}, "
                      f"Val Acc={metrics.val_accuracy:.4f}")
        
        return result
        
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


async def test_training_progress():
    """测试训练进度监控"""
    print("\n🧪 测试训练进度监控...")
    
    try:
        # 获取训练进度
        progress = epoch_training_manager.get_training_progress()
        print(f"📊 当前训练进度: {progress}")
        
        # 测试暂停和恢复
        print("⏸️  测试暂停训练...")
        epoch_training_manager.pause_training()
        
        progress = epoch_training_manager.get_training_progress()
        print(f"📊 暂停后进度: {progress}")
        
        print("▶️  测试恢复训练...")
        epoch_training_manager.resume_training()
        
        progress = epoch_training_manager.get_training_progress()
        print(f"📊 恢复后进度: {progress}")
        
        return True
        
    except Exception as e:
        print(f"❌ 进度监控测试失败: {str(e)}")
        return False


async def test_model_creation():
    """测试模型创建功能"""
    print("\n🧪 测试模型创建功能...")
    
    try:
        trainer = EpochDeepLearningTrainer()
        
        # 测试TensorFlow模型创建
        print("🔧 测试TensorFlow模型创建...")
        config = {
            "model_type": "mlp",
            "hidden_units": [64, 32],
            "dropout_rate": 0.2,
            "learning_rate": 0.001,
            "feature_columns": ["feature1", "feature2", "feature3"]
        }
        
        trainer._initialize_training(config, {})
        
        if trainer.tf_model:
            print("✅ TensorFlow模型创建成功")
        elif trainer.pytorch_model:
            print("✅ PyTorch模型创建成功")
        else:
            print("⚠️  模型创建失败，可能需要安装TensorFlow或PyTorch")
        
        return True
        
    except Exception as e:
        print(f"❌ 模型创建测试失败: {str(e)}")
        return False


async def test_early_stopping():
    """测试早停机制"""
    print("\n🧪 测试早停机制...")
    
    try:
        from backend.algorithm_engine.epoch_trainers import EarlyStopping
        
        early_stopping = EarlyStopping(patience=3, min_delta=0.001)
        
        # 模拟验证损失
        val_losses = [0.5, 0.4, 0.35, 0.34, 0.33, 0.32, 0.31, 0.30]
        
        for epoch, loss in enumerate(val_losses):
            should_stop = early_stopping.should_stop(loss, epoch)
            print(f"  Epoch {epoch}: Loss={loss:.3f}, Should Stop={should_stop}")
            
            if should_stop:
                print(f"🛑 早停触发于第{epoch}轮")
                break
        
        best_epoch = early_stopping.get_best_epoch()
        print(f"🏆 最佳epoch: {best_epoch}")
        
        return True
        
    except Exception as e:
        print(f"❌ 早停测试失败: {str(e)}")
        return False


async def test_learning_rate_scheduler():
    """测试学习率调度器"""
    print("\n🧪 测试学习率调度器...")
    
    try:
        from backend.algorithm_engine.epoch_trainers import LearningRateScheduler
        
        # 测试步进调度器
        scheduler = LearningRateScheduler(initial_lr=0.001, scheduler_type='step')
        
        print("📈 学习率变化:")
        for epoch in range(10):
            lr = scheduler.get_current_lr()
            scheduler.step(epoch, {'val_loss': 0.5})
            print(f"  Epoch {epoch}: LR={lr:.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 学习率调度器测试失败: {str(e)}")
        return False


async def main():
    """主测试函数"""
    print("🚀 开始Epoch训练功能测试...")
    
    try:
        # 测试模型创建
        await test_model_creation()
        
        # 测试早停机制
        await test_early_stopping()
        
        # 测试学习率调度器
        await test_learning_rate_scheduler()
        
        # 测试训练进度监控
        await test_training_progress()
        
        # 测试完整训练流程
        await test_epoch_training()
        
        print("\n✅ 所有Epoch训练测试完成！")
        
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main()) 
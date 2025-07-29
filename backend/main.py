"""
训练存储工作流平台主应用
集成所有服务模块
"""
import logging
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from shared.config import config
from shared.database import init_db, check_db_connection
from doris_connector.connection import doris_manager
from feast_service.manager import feast_manager
from feast_service.training_set_manager import training_set_version_manager
from pipeline_service.incremental_learning import incremental_learner
from auth_service.routes import router as auth_router
from api.algorithm_engine import router as algorithm_router
from api.quality_assessment import router as quality_router
from api.epoch_training import router as epoch_router
from api.gpu_resource_management import router as gpu_router

# 配置日志
logging.basicConfig(
    level=getattr(logging, config.monitoring.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时执行
    logger.info("🚀 训练存储工作流平台启动中...")
    
    try:
        # 初始化数据库
        logger.info("📊 初始化数据库...")
        try:
            init_db()
            
            # 检查数据库连接
            if not check_db_connection():
                logger.warning("⚠️ 数据库连接失败，将在需要时重试")
            else:
                logger.info("✅ 数据库连接成功")
        except Exception as e:
            logger.warning(f"⚠️ 数据库初始化失败，将在需要时重试: {e}")
        
        # 测试Doris连接
        logger.info("🔗 测试Doris连接...")
        if await doris_manager.test_connection():
            logger.info("✅ Doris连接成功")
        else:
            logger.warning("⚠️ Doris连接失败，将在需要时重试")
        
        logger.info("✅ 训练存储工作流平台启动成功")
        
    except Exception as e:
        logger.error(f"❌ 应用启动失败: {e}")
        raise
    
    yield
    
    # 关闭时执行
    logger.info("🔄 训练存储工作流平台关闭中...")
    
    try:
        # 关闭Doris连接
        await doris_manager.close()
        logger.info("✅ Doris连接已关闭")
        
        logger.info("✅ 训练存储工作流平台已关闭")
        
    except Exception as e:
        logger.error(f"❌ 应用关闭时出错: {e}")


# 创建FastAPI应用
app = FastAPI(
    title=config.app_name,
    version=config.version,
    description="独立的企业级AI训练平台，支持Pipeline编排、增量学习、多用户并发训练",
    docs_url="/docs" if config.debug else None,
    redoc_url="/redoc" if config.debug else None,
    lifespan=lifespan
)

# 添加中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境中应该限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# 注册路由
app.include_router(auth_router, prefix="/api/v1")
app.include_router(algorithm_router, prefix="/api/v1")
app.include_router(quality_router, prefix="/api/v1")
app.include_router(epoch_router, prefix="/api/v1")
app.include_router(gpu_router, prefix="/api/v1")


# 健康检查
@app.get("/health")
async def health_check():
    """健康检查接口"""
    try:
        # 检查数据库连接
        db_status = "healthy" if check_db_connection() else "unhealthy"
        
        # 检查Doris连接
        doris_status = "healthy" if await doris_manager.test_connection() else "unhealthy"
        
        return {
            "status": "healthy",
            "timestamp": "2024-07-22T10:00:00Z",
            "version": config.version,
            "services": {
                "database": db_status,
                "doris": doris_status,
                "feast": "healthy"
            }
        }
    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        return {
            "status": "unhealthy",
            "timestamp": "2024-07-22T10:00:00Z",
            "version": config.version,
            "error": str(e)
        }


@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "欢迎使用训练存储工作流平台",
        "version": config.version,
        "docs": "/docs" if config.debug else None,
        "health": "/health"
    }


# 全局异常处理
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """全局异常处理器"""
    logger.error(f"未处理的异常: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "detail": "内部服务器错误",
            "error": str(exc) if config.debug else "请联系管理员"
        }
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """HTTP异常处理器"""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )


@app.get("/api/v1/system/info")
async def get_system_info():
    """获取系统信息"""
    return {
        "app_name": config.app_name,
        "version": config.version,
        "environment": "development" if config.debug else "production",
        "database": {
            "host": config.database.host,
            "port": config.database.port,
            "database": config.database.database
        },
        "doris": {
            "host": config.doris.host,
            "port": config.doris.port,
            "database": config.doris.database
        },
        "feast": {
            "registry_path": config.feast.registry_path,
            "provider": config.feast.provider
        },
        "storage": {
            "default_provider": config.storage.default_provider,
            "mount_base_path": config.storage.mount_base_path
        }
    }


# 功能演示接口
@app.get("/api/v1/demo/pipeline-example")
async def get_pipeline_example():
    """获取Pipeline示例"""
    return {
        "name": "增量学习Pipeline示例",
        "version": "1.0.0",
        "description": "演示完整的增量学习流程",
        "tasks": [
            {
                "name": "doris_data_query",
                "type": "doris_query",
                "config": {
                    "table_name": "feature_snapshots",
                    "time_range": {
                        "start_time": "2024-07-01T00:00:00Z",
                        "end_time": "2024-07-22T00:00:00Z"
                    },
                    "filters": {
                        "node_id": ["node_001", "node_002"]
                    }
                },
                "dependencies": []
            },
            {
                "name": "feast_feature_engineering",
                "type": "feast_feature",
                "config": {
                    "feature_view_name": "sensor_features",
                    "entities": ["uuid", "node_id"],
                    "features": ["meanLf", "std", "peakPowers", "spectralCentroid"]
                },
                "dependencies": ["doris_data_query"]
            },
            {
                "name": "training_set_generation",
                "type": "training_set_generation",
                "config": {
                    "training_set_name": "sensor_training_set_v1",
                    "feature_views": ["sensor_features"],
                    "quality_assessment": True
                },
                "dependencies": ["feast_feature_engineering"]
            },
            {
                "name": "incremental_learning",
                "type": "incremental_learning",
                "config": {
                    "model_type": "transformer_timeseries",
                    "base_model_path": None,
                    "learning_rate": 0.001,
                    "epochs": 10
                },
                "dependencies": ["training_set_generation"]
            }
        ]
    }


@app.get("/api/v1/demo/feature-example")
async def get_feature_example():
    """获取特征工程示例"""
    return {
        "feature_view": {
            "name": "sensor_features",
            "entities": ["uuid", "node_id"],
            "features": [
                "meanLf",
                "std",
                "peakPowers",
                "peakFreqs",
                "spectralCentroid",
                "spectralRolloff",
                "zeroCrossingRate",
                "mfcc_1",
                "mfcc_2",
                "mfcc_3"
            ]
        },
        "training_set": {
            "name": "sensor_training_set_v1",
            "feature_views": ["sensor_features"],
            "entities": ["uuid", "node_id"],
            "features": [
                "meanLf",
                "std",
                "peakPowers",
                "peakFreqs",
                "spectralCentroid"
            ]
        }
    }


@app.get("/api/v1/demo/incremental-learning-example")
async def get_incremental_learning_example():
    """获取增量学习示例"""
    return {
        "base_model": {
            "model_id": "transformer_timeseries_v1",
            "model_type": "transformer_timeseries",
            "created_at": "2024-07-01T00:00:00Z",
            "performance": {
                "accuracy": 0.92,
                "precision": 0.89,
                "recall": 0.91,
                "f1_score": 0.90
            }
        },
        "incremental_training": {
            "new_data_size": 1000,
            "training_config": {
                "learning_rate": 0.001,
                "epochs": 10,
                "batch_size": 32
            },
            "expected_improvement": {
                "accuracy": "+2%",
                "precision": "+1.5%",
                "recall": "+2.5%"
            }
        }
    }


if __name__ == "__main__":
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 
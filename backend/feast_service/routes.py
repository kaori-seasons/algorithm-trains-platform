"""
Feast特征工程服务路由
提供特征视图和训练集管理的REST API接口
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any

from feast_service.manager import feast_manager, FeatureViewConfig, TrainingSetConfig
from shared.database import get_db
from training_set_manager import training_set_version_manager

router = APIRouter()


@router.get("/feature-views")
async def list_feature_views():
    """列出特征视图"""
    try:
        feature_views = await feast_manager.list_feature_views()
        return {"feature_views": feature_views}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/feature-views")
async def create_feature_view(feature_view_config: Dict[str, Any]):
    """创建特征视图"""
    try:
        config = FeatureViewConfig(
            name=feature_view_config["name"],
            entities=feature_view_config["entities"],
            features=feature_view_config["features"],
            description=feature_view_config.get("description", "")
        )
        
        result = await feast_manager.create_feature_view(config)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/feature-views/{name}")
async def get_feature_view(name: str):
    """获取特征视图"""
    try:
        feature_view = await feast_manager.get_feature_view(name)
        if not feature_view:
            raise HTTPException(status_code=404, detail="Feature view not found")
        return feature_view
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/feature-views/{name}")
async def delete_feature_view(name: str):
    """删除特征视图"""
    try:
        success = await feast_manager.delete_feature_view(name)
        if not success:
            raise HTTPException(status_code=404, detail="Feature view not found")
        return {"message": "Feature view deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/training-sets")
async def list_training_sets():
    """列出训练集"""
    try:
        training_sets = await feast_manager.list_training_sets()
        return {"training_sets": training_sets}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/training-sets")
async def create_training_set(training_set_config: Dict[str, Any]):
    """创建训练集"""
    try:
        # 这里需要实际的DataFrame，暂时使用模拟数据
        import pandas as pd
        import numpy as np
        
        # 创建模拟实体数据
        entity_data = pd.DataFrame({
            'uuid': [f'uuid_{i}' for i in range(100)],
            'node_id': [f'node_{i % 10}' for i in range(100)],
            'timestamp': pd.date_range('2024-07-01', periods=100, freq='H')
        })
        
        config = TrainingSetConfig(
            name=training_set_config["name"],
            feature_views=training_set_config["feature_views"],
            entity_df=entity_data,
            description=training_set_config.get("description", "")
        )
        
        result = await feast_manager.generate_training_set(config)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/training-sets/{name}")
async def get_training_set(name: str):
    """获取训练集"""
    try:
        training_set = await feast_manager.get_training_set(name)
        if not training_set:
            raise HTTPException(status_code=404, detail="Training set not found")
        return training_set
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/training-sets/{name}")
async def delete_training_set(name: str):
    """删除训练集"""
    try:
        success = await feast_manager.delete_training_set(name)
        if not success:
            raise HTTPException(status_code=404, detail="Training set not found")
        return {"message": "Training set deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/training-set-versions")
async def list_training_set_versions(
    user_id: Optional[int] = None,
    status: Optional[str] = None
):
    """列出训练集版本"""
    try:
        versions = await training_set_version_manager.list_versions(user_id, status)
        return {"versions": [vars(v) for v in versions]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/training-set-versions")
async def create_training_set_version(version_config: Dict[str, Any]):
    """创建训练集版本"""
    try:
        user_id = version_config.get("user_id", 1)
        parent_versions = version_config.get("parent_versions", [])
        
        version = await training_set_version_manager.create_version(
            version_config, user_id, parent_versions
        )
        return vars(version)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/training-set-versions/{version_id}")
async def get_training_set_version(version_id: str):
    """获取训练集版本"""
    try:
        version = await training_set_version_manager.get_version(version_id)
        if not version:
            raise HTTPException(status_code=404, detail="Version not found")
        return vars(version)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/training-set-versions/{version_id}/assess-quality")
async def assess_version_quality(version_id: str, assessment_data: Dict[str, Any]):
    """评估版本质量"""
    try:
        # 这里需要实际的DataFrame，暂时使用模拟数据
        import pandas as pd
        import numpy as np
        
        # 创建模拟训练集数据
        data = pd.DataFrame({
            'feature_1': np.random.randn(1000),
            'feature_2': np.random.randn(1000),
            'feature_3': np.random.randn(1000),
            'target': np.random.randint(0, 2, 1000)
        })
        
        result = await training_set_version_manager.assess_version_quality(
            version_id, data
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/training-set-versions/select-quality")
async def select_quality_version(criteria: Dict[str, Any]):
    """选择优质版本"""
    try:
        version = await training_set_version_manager.select_quality_version(criteria)
        if not version:
            raise HTTPException(status_code=404, detail="No qualified version found")
        return vars(version)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/training-set-versions/{version_id}/lineage")
async def get_version_lineage(version_id: str):
    """获取版本血缘信息"""
    try:
        lineage = await training_set_version_manager.get_version_lineage(version_id)
        return lineage
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 
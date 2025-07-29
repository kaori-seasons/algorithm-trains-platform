"""
应用配置管理
"""
import os
from typing import Optional
from pydantic_settings import BaseSettings


class DatabaseSettings(BaseSettings):
    """数据库配置"""
    host: str = "localhost"
    port: int = 5432
    database: str = "train_platform"
    username: str = "postgres"
    password: str = "password"
    url: Optional[str] = None
    
    class Config:
        env_prefix = "DB_"


class DorisSettings(BaseSettings):
    """Doris配置"""
    host: str = "localhost"
    port: int = 9030
    database: str = "train_platform"
    username: str = "root"
    password: str = ""
    charset: str = "utf8"
    max_connections: int = 10
    connection_timeout: int = 30
    read_timeout: int = 60
    
    class Config:
        env_prefix = "DORIS_"


class FeastSettings(BaseSettings):
    """Feast配置"""
    registry_path: str = "./feast_registry"
    provider: str = "local"
    online_store_type: str = "redis"
    offline_store_type: str = "file"
    redis_host: str = "localhost"
    redis_port: int = 6379
    
    class Config:
        env_prefix = "FEAST_"


class StorageSettings(BaseSettings):
    """存储配置"""
    default_provider: str = "pvc"
    mount_base_path: str = "/mnt"
    temp_path: str = "/tmp"
    
    # S3配置
    s3_endpoint_url: Optional[str] = None
    s3_access_key: Optional[str] = None
    s3_secret_key: Optional[str] = None
    s3_bucket: str = "train-platform"
    
    # NFS配置
    nfs_server: Optional[str] = None
    nfs_path: Optional[str] = None
    
    class Config:
        env_prefix = "STORAGE_"


class MonitoringSettings(BaseSettings):
    """监控配置"""
    log_level: str = "INFO"
    enable_metrics: bool = True
    metrics_port: int = 9090
    prometheus_port: int = 9090
    grafana_port: int = 3000
    enable_tracing: bool = True
    
    class Config:
        env_prefix = "MONITOR_"


class SecuritySettings(BaseSettings):
    """安全配置"""
    secret_key: str = "your-secret-key-here-change-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    
    class Config:
        env_prefix = "SECURITY_"


class Settings(BaseSettings):
    """应用主配置"""
    # 应用基本信息
    app_name: str = "训练存储工作流平台"
    version: str = "1.0.0"
    debug: bool = True
    
    # 服务器配置
    host: str = "0.0.0.0"
    port: int = 8000
    
    # 子配置
    database: DatabaseSettings = DatabaseSettings()
    doris: DorisSettings = DorisSettings()
    feast: FeastSettings = FeastSettings()
    storage: StorageSettings = StorageSettings()
    monitoring: MonitoringSettings = MonitoringSettings()
    security: SecuritySettings = SecuritySettings()
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# 创建全局配置实例
config = Settings()

# 设置数据库URL
if not config.database.url:
    if config.database.password:
        config.database.url = f"postgresql://{config.database.username}:{config.database.password}@{config.database.host}:{config.database.port}/{config.database.database}"
    else:
        config.database.url = f"postgresql://{config.database.username}@{config.database.host}:{config.database.port}/{config.database.database}" 
-- 训练平台数据库初始化脚本
-- 创建数据库表结构

-- 用户表
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(100) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    full_name VARCHAR(255),
    is_active BOOLEAN DEFAULT TRUE,
    is_admin BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Pipeline表
CREATE TABLE pipelines (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    version VARCHAR(50) NOT NULL,
    user_id INTEGER NOT NULL REFERENCES users(id),
    config JSONB NOT NULL,
    status VARCHAR(50) DEFAULT 'created',
    description TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(name, version)
);

-- 任务表
CREATE TABLE tasks (
    id SERIAL PRIMARY KEY,
    pipeline_id INTEGER NOT NULL REFERENCES pipelines(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    type VARCHAR(100) NOT NULL,
    config JSONB NOT NULL,
    status VARCHAR(50) DEFAULT 'pending',
    dependencies JSONB DEFAULT '[]',
    resources JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- 任务执行记录表
CREATE TABLE task_executions (
    id SERIAL PRIMARY KEY,
    task_id INTEGER NOT NULL REFERENCES tasks(id),
    pipeline_execution_id VARCHAR(255) NOT NULL,
    status VARCHAR(50) NOT NULL,
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    logs TEXT,
    result JSONB,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- 训练集版本表
CREATE TABLE training_set_versions (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    version_id VARCHAR(255) UNIQUE NOT NULL,
    user_id INTEGER NOT NULL REFERENCES users(id),
    doris_query_config JSONB NOT NULL,
    feast_config JSONB NOT NULL,
    quality_score DECIMAL(3,2),
    status VARCHAR(50) DEFAULT 'created',
    data_path TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- 特征快照表
CREATE TABLE feature_snapshots (
    id SERIAL PRIMARY KEY,
    uuid VARCHAR(255) NOT NULL,
    node_id VARCHAR(255) NOT NULL,
    time TIMESTAMP NOT NULL,
    features JSONB NOT NULL,
    is_tag BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW(),
    INDEX idx_uuid (uuid),
    INDEX idx_node_id (node_id),
    INDEX idx_time (time)
);

-- 存储挂载表
CREATE TABLE storage_mounts (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id),
    mount_path VARCHAR(500) NOT NULL,
    storage_type VARCHAR(50) NOT NULL,
    config JSONB NOT NULL,
    status VARCHAR(50) DEFAULT 'mounted',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- 模型版本表
CREATE TABLE model_versions (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    version VARCHAR(50) NOT NULL,
    user_id INTEGER NOT NULL REFERENCES users(id),
    model_path TEXT NOT NULL,
    config JSONB NOT NULL,
    metrics JSONB DEFAULT '{}',
    status VARCHAR(50) DEFAULT 'created',
    training_set_version_id INTEGER REFERENCES training_set_versions(id),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(name, version)
);

-- 增量学习记录表
CREATE TABLE incremental_learning_records (
    id SERIAL PRIMARY KEY,
    base_model_version_id INTEGER NOT NULL REFERENCES model_versions(id),
    new_model_version_id INTEGER NOT NULL REFERENCES model_versions(id),
    training_data_path TEXT NOT NULL,
    learning_config JSONB NOT NULL,
    performance_improvement JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- 系统配置表
CREATE TABLE system_configs (
    id SERIAL PRIMARY KEY,
    key VARCHAR(255) UNIQUE NOT NULL,
    value JSONB NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- 审计日志表
CREATE TABLE audit_logs (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(100) NOT NULL,
    resource_id VARCHAR(255),
    details JSONB,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- 创建索引
CREATE INDEX idx_pipelines_user_id ON pipelines(user_id);
CREATE INDEX idx_pipelines_status ON pipelines(status);
CREATE INDEX idx_tasks_pipeline_id ON tasks(pipeline_id);
CREATE INDEX idx_tasks_status ON tasks(status);
CREATE INDEX idx_task_executions_pipeline_execution_id ON task_executions(pipeline_execution_id);
CREATE INDEX idx_training_set_versions_user_id ON training_set_versions(user_id);
CREATE INDEX idx_training_set_versions_status ON training_set_versions(status);
CREATE INDEX idx_storage_mounts_user_id ON storage_mounts(user_id);
CREATE INDEX idx_model_versions_user_id ON model_versions(user_id);
CREATE INDEX idx_audit_logs_user_id ON audit_logs(user_id);
CREATE INDEX idx_audit_logs_created_at ON audit_logs(created_at);

-- 创建触发器函数
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- 为需要自动更新updated_at的表创建触发器
CREATE TRIGGER update_pipelines_updated_at BEFORE UPDATE ON pipelines
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_tasks_updated_at BEFORE UPDATE ON tasks
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_training_set_versions_updated_at BEFORE UPDATE ON training_set_versions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_storage_mounts_updated_at BEFORE UPDATE ON storage_mounts
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_model_versions_updated_at BEFORE UPDATE ON model_versions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_system_configs_updated_at BEFORE UPDATE ON system_configs
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- 插入默认系统配置
INSERT INTO system_configs (key, value, description) VALUES
('storage.default_provider', '"pvc"', '默认存储提供者类型'),
('doris.connection_pool_size', '10', 'Doris连接池大小'),
('feast.feature_store_path', '"/opt/feast/feature_store"', 'Feast特征存储路径'),
('monitoring.enabled', 'true', '是否启用监控'),
('security.jwt_secret', '"your-secret-key-here"', 'JWT密钥'),
('security.jwt_expiration', '3600', 'JWT过期时间（秒）');

-- 创建默认管理员用户（密码需要在应用层处理）
INSERT INTO users (username, email, password_hash, full_name, is_admin) VALUES
('admin', 'admin@train-platform.com', 'placeholder_hash', '系统管理员', TRUE); 
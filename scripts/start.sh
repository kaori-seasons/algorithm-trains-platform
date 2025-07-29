#!/bin/bash

# 训练平台启动脚本

set -e

echo "🚀 启动训练存储工作流平台..."

# 检查Python版本
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.9"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ 错误: 需要Python $required_version或更高版本，当前版本: $python_version"
    exit 1
fi

echo "✅ Python版本检查通过: $python_version"

# 检查依赖
echo "📦 检查依赖包..."
if ! python3 -c "import fastapi, sqlalchemy, kubernetes" 2>/dev/null; then
    echo "⚠️  检测到缺少依赖包，正在安装..."
    pip3 install -r requirements.txt
fi

# 检查环境变量
echo "🔧 检查环境配置..."
if [ -z "$DB_HOST" ]; then
    export DB_HOST="localhost"
    echo "设置默认数据库主机: $DB_HOST"
fi

if [ -z "$DB_PASSWORD" ]; then
    export DB_PASSWORD="train_password"
    echo "设置默认数据库密码: $DB_PASSWORD"
fi

# 检查数据库连接
echo "🔍 检查数据库连接..."
if ! python3 -c "
import asyncio
import sys
sys.path.append('.')
from backend.shared.database import test_database_connection
result = asyncio.run(test_database_connection())
if not result:
    sys.exit(1)
" 2>/dev/null; then
    echo "⚠️  数据库连接失败，请检查数据库配置"
    echo "💡 提示: 可以使用 docker-compose up -d 启动本地数据库"
fi

# 启动应用
echo "🌟 启动训练平台服务..."
cd backend
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload 
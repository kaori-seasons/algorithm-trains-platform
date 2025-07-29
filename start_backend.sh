#!/bin/bash

# 训练存储工作流平台后端启动脚本

echo "🚀 启动训练存储工作流平台后端..."

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 未安装，请先安装Python3"
    exit 1
fi

# 检查虚拟环境
if [ ! -d "venv" ]; then
    echo "📦 创建虚拟环境..."
    python3 -m venv venv
fi

# 激活虚拟环境
echo "🔧 激活虚拟环境..."
source venv/bin/activate

# 安装依赖
echo "📥 安装依赖包..."
pip install -r requirements.txt

# 检查数据库
echo "🗄️ 检查数据库配置..."
if [ ! -f "train_platform.db" ]; then
    echo "📊 初始化数据库..."
    python3 -c "
import sys
sys.path.append('.')
from backend.shared.database import init_db
init_db()
"
fi

# 启动后端服务
echo "🌐 启动后端服务..."
echo "📖 API文档地址: http://localhost:8000/docs"
echo "🔍 健康检查: http://localhost:8000/health"
echo ""

python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload 
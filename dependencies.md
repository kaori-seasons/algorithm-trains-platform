# 依赖包说明

## 核心依赖

### 1. kubernetes>=28.1.0
**用途**：Kubernetes Python客户端库
**功能**：
- 与Kubernetes API交互
- 创建和管理PVC、Pod、Deployment等资源
- 支持集群内和集群外访问

**安装**：
```bash
pip install kubernetes>=28.1.0
```

**使用场景**：
- PVCProvider：创建和管理PVC
- ConfigMapProvider：管理ConfigMap
- SecretProvider：管理Secret
- Pod和Deployment的创建和管理

### 2. PyYAML>=6.0
**用途**：YAML配置文件处理
**功能**：
- 解析和生成YAML格式的配置文件
- 支持Kubernetes资源定义文件
- 配置文件读写

**安装**：
```bash
pip install PyYAML>=6.0
```

**使用场景**：
- 解析Kubernetes配置文件
- 生成Deployment和Pod的YAML定义
- 配置文件管理

### 3. boto3>=1.34.0
**用途**：AWS SDK for Python
**功能**：
- AWS S3存储操作
- 文件上传、下载、删除
- 存储桶管理

**安装**：
```bash
pip install boto3>=1.34.0
```

**使用场景**：
- S3Provider：AWS S3存储支持
- 对象存储操作
- 云存储集成

## 可选依赖

### 云存储提供商

#### MinIO S3兼容存储
```bash
pip install minio>=7.2.0
```
**用途**：MinIO对象存储支持
**适用场景**：私有云存储、S3兼容存储

#### 阿里云OSS
```bash
pip install oss2>=2.18.0
```
**用途**：阿里云对象存储服务
**适用场景**：阿里云环境

#### 腾讯云COS
```bash
pip install cos-python-sdk-v5>=1.9.0
```
**用途**：腾讯云对象存储
**适用场景**：腾讯云环境

#### 华为云OBS
```bash
pip install obs-python-sdk>=3.23.0
```
**用途**：华为云对象存储
**适用场景**：华为云环境

## 开发依赖

### 代码质量工具

#### Black - 代码格式化
```bash
pip install black>=23.0.0
```
**用途**：自动代码格式化
**使用**：
```bash
black storage-providers.py
```

#### Flake8 - 代码检查
```bash
pip install flake8>=6.0.0
```
**用途**：代码风格和错误检查
**使用**：
```bash
flake8 storage-providers.py
```

#### MyPy - 类型检查
```bash
pip install mypy>=1.0.0
```
**用途**：静态类型检查
**使用**：
```bash
mypy storage-providers.py
```

### 测试框架

#### Pytest - 测试框架
```bash
pip install pytest>=7.0.0
```
**用途**：单元测试和集成测试

#### Pytest-cov - 测试覆盖率
```bash
pip install pytest-cov>=4.0.0
```
**用途**：测试覆盖率报告

### 文档生成

#### Sphinx - 文档生成
```bash
pip install sphinx>=6.0.0
```
**用途**：生成项目文档

#### Sphinx RTD Theme - 文档主题
```bash
pip install sphinx-rtd-theme>=1.2.0
```
**用途**：ReadTheDocs风格的文档主题

### 监控和日志

#### Psutil - 系统监控
```bash
pip install psutil>=5.9.0
```
**用途**：系统资源监控
**功能**：CPU、内存、磁盘使用率监控

#### Structlog - 结构化日志
```bash
pip install structlog>=23.0.0
```
**用途**：结构化日志输出
**功能**：JSON格式日志、日志级别管理

### 配置管理

#### Python-dotenv - 环境变量
```bash
pip install python-dotenv>=1.0.0
```
**用途**：环境变量管理
**功能**：从.env文件加载配置

### 类型检查

#### Types-PyYAML - YAML类型
```bash
pip install types-PyYAML>=6.0.0
```
**用途**：PyYAML的类型提示

#### Types-requests - Requests类型
```bash
pip install types-requests>=2.31.0
```
**用途**：Requests库的类型提示

## 安装指南

### 1. 最小安装（仅核心功能）
```bash
pip install -r requirements.txt
```

### 2. 完整安装（包含开发工具）
```bash
# 取消注释requirements.txt中的开发依赖
pip install -r requirements.txt
```

### 3. 按需安装
```bash
# 仅安装核心依赖
pip install kubernetes>=28.1.0 PyYAML>=6.0 boto3>=1.34.0

# 如果需要MinIO支持
pip install minio>=7.2.0

# 如果需要开发工具
pip install black>=23.0.0 flake8>=6.0.0 mypy>=1.0.0
```

## 环境要求

### Python版本
- Python 3.8+
- 推荐使用Python 3.9或更高版本

### 操作系统
- Linux（推荐）
- macOS
- Windows（部分功能可能受限）

### Kubernetes环境
- Kubernetes 1.20+
- kubectl命令行工具
- 适当的RBAC权限

## 配置说明

### Kubernetes配置
```python
# 集群外访问
from kubernetes import config
config.load_kube_config(config_file="/path/to/kubeconfig")

# 集群内访问
config.load_incluster_config()
```

### AWS S3配置
```python
# 环境变量方式
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=us-west-2

# 配置文件方式
# ~/.aws/credentials
# ~/.aws/config
```

### MinIO配置
```python
# 环境变量
export MINIO_ENDPOINT=minio.example.com
export MINIO_ACCESS_KEY=your_access_key
export MINIO_SECRET_KEY=your_secret_key
```

## 故障排除

### 常见问题

#### 1. Kubernetes连接失败
```bash
# 检查kubectl配置
kubectl config view

# 检查集群连接
kubectl cluster-info
```

#### 2. S3连接失败
```bash
# 检查AWS凭证
aws sts get-caller-identity

# 检查S3访问权限
aws s3 ls
```

#### 3. 权限问题
```bash
# 检查Python包权限
pip install --user package_name

# 使用虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

## 版本兼容性

### Kubernetes客户端版本
- kubernetes 28.x：支持Kubernetes 1.20+
- kubernetes 27.x：支持Kubernetes 1.19+
- kubernetes 26.x：支持Kubernetes 1.18+

### Python版本兼容性
- Python 3.8：所有依赖包支持
- Python 3.9：推荐版本
- Python 3.10：完全兼容
- Python 3.11：完全兼容

## 更新和维护

### 定期更新
```bash
# 更新所有依赖
pip install --upgrade -r requirements.txt

# 检查过时的包
pip list --outdated

# 更新特定包
pip install --upgrade package_name
```

### 安全更新
```bash
# 检查安全漏洞
pip-audit

# 安装安全更新
pip install --upgrade --security-only package_name
``` 
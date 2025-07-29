# 训练存储工作流平台前端

基于Vue2 + Element UI + Ant Design Vue构建的企业级AI训练平台前端系统。

## 技术栈

- **Vue 2.6.14** - 渐进式JavaScript框架
- **Element UI 2.15.13** - 基于Vue的组件库
- **Ant Design Vue 1.7.8** - 企业级UI设计语言和React组件库
- **Vue Router 3.5.3** - Vue.js官方路由管理器
- **Vuex 3.6.2** - Vue.js的状态管理模式
- **Axios 0.27.2** - 基于Promise的HTTP客户端
- **ECharts 5.4.3** - 数据可视化图表库

## 项目特性

- 🎨 **现代化UI设计** - 采用Element UI和Ant Design的设计语言
- 📱 **响应式布局** - 支持桌面端和移动端自适应
- 🔧 **Flex布局** - 使用CSS Flexbox实现灵活的布局
- 🚀 **模块化架构** - 基于Vuex的状态管理和模块化组件
- 📊 **数据可视化** - 集成ECharts图表库
- 🔐 **权限管理** - 基于JWT的身份认证和授权
- 🌐 **国际化支持** - 支持多语言切换

## 快速开始

### 环境要求

- Node.js >= 14.0.0
- npm >= 6.0.0

### 安装依赖

```bash
npm install
```

### 开发环境启动

```bash
npm run serve
```

### 生产环境构建

```bash
npm run build
```

### 代码检查

```bash
npm run lint
```

## 项目结构

```
src/
├── api/                 # API接口
├── assets/              # 静态资源
├── components/          # 全局组件
├── layout/              # 布局组件
├── router/              # 路由配置
├── store/               # Vuex状态管理
├── styles/              # 全局样式
├── utils/               # 工具函数
├── views/               # 页面组件
├── App.vue              # 根组件
└── main.js              # 入口文件
```

## 主要功能模块

### 1. 仪表盘
- 系统概览统计
- Pipeline执行趋势图表
- 存储使用情况
- 最近活动列表
- 快速操作入口

### 2. Pipeline管理
- Pipeline列表展示
- Pipeline创建和编辑
- Pipeline执行监控
- 任务状态跟踪

### 3. Feast特征平台
- 特征管理
- 训练集管理
- 在线服务监控

### 4. Doris数据源
- 数据查询界面
- 表管理
- 监控面板

### 5. 增量学习
- 增量任务管理
- 模型版本控制
- 训练历史记录

### 6. 存储管理
- 挂载点管理
- 存储提供者配置
- 存储监控

### 7. 系统监控
- 监控概览
- 系统日志
- 告警管理

## 布局设计

### 响应式布局
- 桌面端：侧边栏 + 主内容区
- 平板端：可折叠侧边栏
- 移动端：抽屉式侧边栏

### Flex布局特性
- 使用CSS Flexbox实现灵活的布局
- 支持不同屏幕尺寸的自适应
- 组件间距和排列的精确控制

## 开发规范

### 组件命名
- 使用PascalCase命名组件
- 文件名与组件名保持一致

### 样式规范
- 使用SCSS预处理器
- 采用BEM命名规范
- 优先使用flex布局

### 代码规范
- 遵循ESLint规则
- 使用Prettier格式化代码
- 编写组件注释和文档

## 部署说明

### 开发环境
```bash
# 启动开发服务器
npm run serve

# 访问地址
http://localhost:3000
```

### 生产环境
```bash
# 构建生产版本
npm run build

# 部署dist目录到Web服务器
```

## 浏览器支持

- Chrome >= 60
- Firefox >= 60
- Safari >= 12
- Edge >= 79

## 贡献指南

1. Fork项目
2. 创建功能分支
3. 提交代码
4. 创建Pull Request

## 许可证

MIT License 
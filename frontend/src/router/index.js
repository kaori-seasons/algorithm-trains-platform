import Vue from 'vue'
import VueRouter from 'vue-router'
import Layout from '@/layout/index.vue'

Vue.use(VueRouter)

const routes = [
  {
    path: '/login',
    name: 'Login',
    component: () => import('@/views/login/index.vue'),
    meta: { title: '登录', hidden: true }
  },
  {
    path: '/',
    component: Layout,
    redirect: '/dashboard',
    children: [
      {
        path: 'dashboard',
        name: 'Dashboard',
        component: () => import('@/views/dashboard/index.vue'),
        meta: { title: '仪表盘', icon: 'dashboard' }
      }
    ]
  },
  {
    path: '/pipeline',
    component: Layout,
    redirect: '/pipeline/list',
    meta: { title: 'Pipeline管理', icon: 'workflow' },
    children: [
      {
        path: 'list',
        name: 'PipelineList',
        component: () => import('@/views/pipeline/list.vue'),
        meta: { title: 'Pipeline列表' }
      },
      {
        path: 'create',
        name: 'PipelineCreate',
        component: () => import('@/views/pipeline/create.vue'),
        meta: { title: '创建Pipeline' }
      },
      {
        path: 'edit/:id',
        name: 'PipelineEdit',
        component: () => import('@/views/pipeline/edit.vue'),
        meta: { title: '编辑Pipeline', hidden: true }
      },
      {
        path: 'detail/:id',
        name: 'PipelineDetail',
        component: () => import('@/views/pipeline/detail.vue'),
        meta: { title: 'Pipeline详情', hidden: true }
      }
    ]
  },
  {
    path: '/feast',
    component: Layout,
    redirect: '/feast/features',
    meta: { title: 'Feast特征平台', icon: 'database' },
    children: [
      {
        path: 'features',
        name: 'Features',
        component: () => import('@/views/feast/features.vue'),
        meta: { title: '特征管理' }
      },
      {
        path: 'training-sets',
        name: 'TrainingSets',
        component: () => import('@/views/feast/training-sets.vue'),
        meta: { title: '训练集管理' }
      },
      {
        path: 'online-serving',
        name: 'OnlineServing',
        component: () => import('@/views/feast/online-serving.vue'),
        meta: { title: '在线服务' }
      }
    ]
  },
  {
    path: '/doris',
    component: Layout,
    redirect: '/doris/data',
    meta: { title: 'Doris数据源', icon: 'table' },
    children: [
      {
        path: 'data',
        name: 'DorisData',
        component: () => import('@/views/doris/data.vue'),
        meta: { title: '数据查询' }
      },
      {
        path: 'tables',
        name: 'DorisTables',
        component: () => import('@/views/doris/tables.vue'),
        meta: { title: '表管理' }
      },
      {
        path: 'monitoring',
        name: 'DorisMonitoring',
        component: () => import('@/views/doris/monitoring.vue'),
        meta: { title: '监控面板' }
      }
    ]
  },
  {
    path: '/incremental-learning',
    component: Layout,
    redirect: '/incremental-learning/tasks',
    meta: { title: '增量学习', icon: 'experiment' },
    children: [
      {
        path: 'tasks',
        name: 'IncrementalTasks',
        component: () => import('@/views/incremental-learning/tasks.vue'),
        meta: { title: '增量任务' }
      },
      {
        path: 'models',
        name: 'IncrementalModels',
        component: () => import('@/views/incremental-learning/models.vue'),
        meta: { title: '模型版本' }
      },
      {
        path: 'history',
        name: 'IncrementalHistory',
        component: () => import('@/views/incremental-learning/history.vue'),
        meta: { title: '训练历史' }
      }
    ]
  },
  {
    path: '/storage',
    component: Layout,
    redirect: '/storage/mounts',
    meta: { title: '存储管理', icon: 'folder' },
    children: [
      {
        path: 'mounts',
        name: 'StorageMounts',
        component: () => import('@/views/storage/mounts.vue'),
        meta: { title: '挂载点管理' }
      },
      {
        path: 'providers',
        name: 'StorageProviders',
        component: () => import('@/views/storage/providers.vue'),
        meta: { title: '存储提供者' }
      },
      {
        path: 'monitoring',
        name: 'StorageMonitoring',
        component: () => import('@/views/storage/monitoring.vue'),
        meta: { title: '存储监控' }
      }
    ]
  },
  {
    path: '/monitor',
    component: Layout,
    redirect: '/monitor/overview',
    meta: { title: '系统监控', icon: 'monitor' },
    children: [
      {
        path: 'overview',
        name: 'MonitorOverview',
        component: () => import('@/views/monitor/overview.vue'),
        meta: { title: '监控概览' }
      },
      {
        path: 'logs',
        name: 'MonitorLogs',
        component: () => import('@/views/monitor/logs.vue'),
        meta: { title: '系统日志' }
      },
      {
        path: 'alerts',
        name: 'MonitorAlerts',
        component: () => import('@/views/monitor/alerts.vue'),
        meta: { title: '告警管理' }
      }
    ]
  },
  {
    path: '/user',
    component: Layout,
    redirect: '/user/profile',
    meta: { title: '用户管理', icon: 'user' },
    children: [
      {
        path: 'profile',
        name: 'UserProfile',
        component: () => import('@/views/user/profile.vue'),
        meta: { title: '个人资料' }
      },
      {
        path: 'settings',
        name: 'UserSettings',
        component: () => import('@/views/user/settings.vue'),
        meta: { title: '系统设置' }
      }
    ]
  }
]

const router = new VueRouter({
  mode: 'history',
  base: process.env.BASE_URL,
  routes
})

// 路由守卫
router.beforeEach((to, from, next) => {
  const token = localStorage.getItem('token')
  
  if (to.path === '/login') {
    if (token) {
      next('/')
    } else {
      next()
    }
  } else {
    if (token) {
      next()
    } else {
      next('/login')
    }
  }
})

export default router 
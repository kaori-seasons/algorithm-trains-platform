import Vue from 'vue'
import App from './App.vue'
import router from './router'
import store from './store'

// Element UI
import ElementUI from 'element-ui'
import 'element-ui/lib/theme-chalk/index.css'

// Ant Design Vue
import Antd from 'ant-design-vue'
import 'ant-design-vue/dist/antd.css'

// 全局样式
import './styles/index.scss'

// 全局组件
import './components'

// 工具库
import axios from 'axios'
import moment from 'moment'
import _ from 'lodash'

// 配置
Vue.use(ElementUI)
Vue.use(Antd)

// 全局属性
Vue.prototype.$http = axios
Vue.prototype.$moment = moment
Vue.prototype.$_ = _

// 配置axios
axios.defaults.baseURL = process.env.VUE_APP_API_BASE_URL || '/api'
axios.defaults.timeout = 10000

// 请求拦截器
axios.interceptors.request.use(
  config => {
    // 添加token
    const token = localStorage.getItem('token')
    if (token) {
      config.headers.Authorization = `Bearer ${token}`
    }
    return config
  },
  error => {
    return Promise.reject(error)
  }
)

// 响应拦截器
axios.interceptors.response.use(
  response => {
    return response
  },
  error => {
    if (error.response && error.response.status === 401) {
      // 未授权，跳转到登录页
      router.push('/login')
    }
    return Promise.reject(error)
  }
)

Vue.config.productionTip = false

new Vue({
  router,
  store,
  render: h => h(App)
}).$mount('#app') 
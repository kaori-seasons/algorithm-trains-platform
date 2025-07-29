import { login, register, getCurrentUser, updateCurrentUser, refreshToken } from '@/api/auth'
import { getToken, setToken, removeToken, setRefreshToken, getRefreshToken } from '@/utils/auth'

const state = {
  token: getToken(),
  refreshToken: getRefreshToken(),
  userInfo: null,
  isLoggedIn: false,
  loading: false
}

const mutations = {
  SET_TOKEN: (state, token) => {
    state.token = token
  },
  SET_REFRESH_TOKEN: (state, refreshToken) => {
    state.refreshToken = refreshToken
  },
  SET_USER_INFO: (state, userInfo) => {
    state.userInfo = userInfo
  },
  SET_LOGGED_IN: (state, status) => {
    state.isLoggedIn = status
  },
  SET_LOADING: (state, loading) => {
    state.loading = loading
  },
  CLEAR_USER_DATA: (state) => {
    state.token = null
    state.refreshToken = null
    state.userInfo = null
    state.isLoggedIn = false
  }
}

const actions = {
  // 用户登录
  async login({ commit, dispatch }, userInfo) {
    try {
      commit('SET_LOADING', true)
      const response = await login(userInfo)
      const { access_token, refresh_token } = response
      
      commit('SET_TOKEN', access_token)
      commit('SET_REFRESH_TOKEN', refresh_token)
      commit('SET_LOGGED_IN', true)
      
      setToken(access_token)
      setRefreshToken(refresh_token)
      
      // 获取用户信息
      await dispatch('getUserInfo')
      
      return response
    } catch (error) {
      console.error('登录失败:', error)
      throw error
    } finally {
      commit('SET_LOADING', false)
    }
  },

  // 用户注册
  async register({ commit }, userInfo) {
    try {
      commit('SET_LOADING', true)
      const response = await register(userInfo)
      return response
    } catch (error) {
      console.error('注册失败:', error)
      throw error
    } finally {
      commit('SET_LOADING', false)
    }
  },

  // 获取用户信息
  async getUserInfo({ commit, state, dispatch }) {
    try {
      if (!state.token) {
        throw new Error('未找到访问令牌')
      }
      
      const response = await getCurrentUser()
      const userInfo = response
      
      commit('SET_USER_INFO', userInfo)
      commit('SET_LOGGED_IN', true)
      
      return userInfo
    } catch (error) {
      console.error('获取用户信息失败:', error)
      // 如果获取用户信息失败，可能是token过期，尝试刷新
      if (error.response && error.response.status === 401) {
        await dispatch('refreshUserToken')
      }
      throw error
    }
  },

  // 更新用户信息
  async updateUserInfo({ commit }, userInfo) {
    try {
      commit('SET_LOADING', true)
      const response = await updateCurrentUser(userInfo)
      const updatedUserInfo = response
      
      commit('SET_USER_INFO', updatedUserInfo)
      return updatedUserInfo
    } catch (error) {
      console.error('更新用户信息失败:', error)
      throw error
    } finally {
      commit('SET_LOADING', false)
    }
  },

  // 刷新访问令牌
  async refreshUserToken({ commit, state, dispatch }) {
    try {
      if (!state.refreshToken) {
        throw new Error('未找到刷新令牌')
      }
      
      const response = await refreshToken(state.refreshToken)
      const { access_token } = response
      
      commit('SET_TOKEN', access_token)
      setToken(access_token)
      
      return access_token
    } catch (error) {
      console.error('刷新令牌失败:', error)
      // 刷新失败，清除所有认证信息
      await dispatch('logout')
      throw error
    }
  },

  // 用户登出
  async logout({ commit }) {
    try {
      commit('CLEAR_USER_DATA')
      removeToken()
      // 这里可以调用后端的登出接口（如果有的话）
      // await logout()
    } catch (error) {
      console.error('登出失败:', error)
    }
  },

  // 检查认证状态
  async checkAuth({ state, dispatch }) {
    try {
      if (!state.token) {
        return false
      }
      
      // 尝试获取用户信息来验证token是否有效
      await dispatch('getUserInfo')
      return true
    } catch (error) {
      console.error('认证检查失败:', error)
      return false
    }
  }
}

const getters = {
  token: state => state.token,
  refreshToken: state => state.refreshToken,
  userInfo: state => state.userInfo,
  isLoggedIn: state => state.isLoggedIn,
  loading: state => state.loading,
  isAdmin: state => state.userInfo ? state.userInfo.is_admin : false,
  username: state => state.userInfo ? state.userInfo.username : '',
  email: state => state.userInfo ? state.userInfo.email : '',
  fullName: state => state.userInfo ? state.userInfo.full_name : ''
}

export default {
  namespaced: true,
  state,
  mutations,
  actions,
  getters
} 
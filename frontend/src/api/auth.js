import request from '@/utils/request'

/**
 * 用户登录
 * @param {Object} data - 登录信息
 * @param {string} data.username - 用户名
 * @param {string} data.password - 密码
 * @returns {Promise} 登录结果
 */
export function login(data) {
  return request({
    url: '/api/v1/auth/login',
    method: 'post',
    data
  })
}

/**
 * 用户注册
 * @param {Object} data - 注册信息
 * @param {string} data.username - 用户名
 * @param {string} data.email - 邮箱
 * @param {string} data.password - 密码
 * @param {string} data.full_name - 全名（可选）
 * @returns {Promise} 注册结果
 */
export function register(data) {
  return request({
    url: '/api/v1/auth/register',
    method: 'post',
    data
  })
}

/**
 * 刷新访问令牌
 * @param {string} refresh_token - 刷新令牌
 * @returns {Promise} 新的访问令牌
 */
export function refreshToken(refresh_token) {
  return request({
    url: '/api/v1/auth/refresh',
    method: 'post',
    data: { refresh_token }
  })
}

/**
 * 获取当前用户信息
 * @returns {Promise} 用户信息
 */
export function getCurrentUser() {
  return request({
    url: '/api/v1/auth/me',
    method: 'get'
  })
}

/**
 * 更新当前用户信息
 * @param {Object} data - 用户信息
 * @returns {Promise} 更新结果
 */
export function updateCurrentUser(data) {
  return request({
    url: '/api/v1/auth/me',
    method: 'put',
    data
  })
}

/**
 * 获取用户列表（管理员权限）
 * @param {Object} params - 查询参数
 * @param {number} params.skip - 跳过数量
 * @param {number} params.limit - 限制数量
 * @returns {Promise} 用户列表
 */
export function getUserList(params) {
  return request({
    url: '/api/v1/auth/users',
    method: 'get',
    params
  })
}

/**
 * 获取指定用户信息
 * @param {number} userId - 用户ID
 * @returns {Promise} 用户信息
 */
export function getUserById(userId) {
  return request({
    url: `/api/v1/auth/users/${userId}`,
    method: 'get'
  })
}

/**
 * 更新指定用户信息（管理员权限）
 * @param {number} userId - 用户ID
 * @param {Object} data - 用户信息
 * @returns {Promise} 更新结果
 */
export function updateUserById(userId, data) {
  return request({
    url: `/api/v1/auth/users/${userId}`,
    method: 'put',
    data
  })
}

/**
 * 删除用户（管理员权限）
 * @param {number} userId - 用户ID
 * @returns {Promise} 删除结果
 */
export function deleteUserById(userId) {
  return request({
    url: `/api/v1/auth/users/${userId}`,
    method: 'delete'
  })
} 
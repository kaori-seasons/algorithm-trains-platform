/**
 * 认证相关工具函数
 */

const TOKEN_KEY = 'access_token'
const REFRESH_TOKEN_KEY = 'refresh_token'

/**
 * 获取访问令牌
 * @returns {string|null} 访问令牌
 */
export function getToken() {
  return localStorage.getItem(TOKEN_KEY)
}

/**
 * 设置访问令牌
 * @param {string} token - 访问令牌
 */
export function setToken(token) {
  localStorage.setItem(TOKEN_KEY, token)
}

/**
 * 移除访问令牌
 */
export function removeToken() {
  localStorage.removeItem(TOKEN_KEY)
}

/**
 * 获取刷新令牌
 * @returns {string|null} 刷新令牌
 */
export function getRefreshToken() {
  return localStorage.getItem(REFRESH_TOKEN_KEY)
}

/**
 * 设置刷新令牌
 * @param {string} refreshToken - 刷新令牌
 */
export function setRefreshToken(refreshToken) {
  localStorage.setItem(REFRESH_TOKEN_KEY, refreshToken)
}

/**
 * 移除刷新令牌
 */
export function removeRefreshToken() {
  localStorage.removeItem(REFRESH_TOKEN_KEY)
}

/**
 * 清除所有认证信息
 */
export function clearAuth() {
  removeToken()
  removeRefreshToken()
}

/**
 * 检查令牌是否过期
 * @param {string} token - JWT令牌
 * @returns {boolean} 是否过期
 */
export function isTokenExpired(token) {
  if (!token) return true
  
  try {
    const payload = JSON.parse(atob(token.split('.')[1]))
    const currentTime = Date.now() / 1000
    
    return payload.exp < currentTime
  } catch (error) {
    console.error('解析令牌失败:', error)
    return true
  }
}

/**
 * 从令牌中获取用户信息
 * @param {string} token - JWT令牌
 * @returns {Object|null} 用户信息
 */
export function getUserFromToken(token) {
  if (!token) return null
  
  try {
    const payload = JSON.parse(atob(token.split('.')[1]))
    return {
      username: payload.sub,
      exp: payload.exp,
      type: payload.type
    }
  } catch (error) {
    console.error('从令牌获取用户信息失败:', error)
    return null
  }
} 
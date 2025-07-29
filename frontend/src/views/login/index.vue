<template>
  <div class="login-container">
    <div class="login-background">
      <div class="login-content">
        <div class="login-header">
          <img
            src="@/assets/logo.svg"
            alt="Logo"
            class="logo"
          >
          <h1 class="title">
            训练存储工作流平台
          </h1>
          <p class="subtitle">
            企业级AI训练平台，支持Pipeline编排、增量学习、多用户并发训练
          </p>
        </div>
        
        <div class="login-form-container">
          <el-tabs
            v-model="activeTab"
            class="login-tabs"
          >
            <el-tab-pane
              label="登录"
              name="login"
            >
              <el-form
                ref="loginForm"
                :model="loginForm"
                :rules="loginRules"
                class="login-form"
                @submit.native.prevent="handleLogin"
              >
                <el-form-item prop="username">
                  <el-input
                    v-model="loginForm.username"
                    placeholder="用户名"
                    prefix-icon="el-icon-user"
                    size="large"
                  />
                </el-form-item>
                
                <el-form-item prop="password">
                  <el-input
                    v-model="loginForm.password"
                    type="password"
                    placeholder="密码"
                    prefix-icon="el-icon-lock"
                    size="large"
                    show-password
                    @keyup.enter.native="handleLogin"
                  />
                </el-form-item>
                
                <el-form-item>
                  <el-button
                    type="primary"
                    size="large"
                    class="login-button"
                    :loading="loading"
                    @click="handleLogin"
                  >
                    登录
                  </el-button>
                </el-form-item>
              </el-form>
            </el-tab-pane>
            
            <el-tab-pane
              label="注册"
              name="register"
            >
              <el-form
                ref="registerForm"
                :model="registerForm"
                :rules="registerRules"
                class="register-form"
                @submit.native.prevent="handleRegister"
              >
                <el-form-item prop="username">
                  <el-input
                    v-model="registerForm.username"
                    placeholder="用户名"
                    prefix-icon="el-icon-user"
                    size="large"
                  />
                </el-form-item>
                
                <el-form-item prop="email">
                  <el-input
                    v-model="registerForm.email"
                    placeholder="邮箱"
                    prefix-icon="el-icon-message"
                    size="large"
                  />
                </el-form-item>
                
                <el-form-item prop="full_name">
                  <el-input
                    v-model="registerForm.full_name"
                    placeholder="全名（可选）"
                    prefix-icon="el-icon-user"
                    size="large"
                  />
                </el-form-item>
                
                <el-form-item prop="password">
                  <el-input
                    v-model="registerForm.password"
                    type="password"
                    placeholder="密码"
                    prefix-icon="el-icon-lock"
                    size="large"
                    show-password
                  />
                </el-form-item>
                
                <el-form-item prop="confirm_password">
                  <el-input
                    v-model="registerForm.confirm_password"
                    type="password"
                    placeholder="确认密码"
                    prefix-icon="el-icon-lock"
                    size="large"
                    show-password
                    @keyup.enter.native="handleRegister"
                  />
                </el-form-item>
                
                <el-form-item>
                  <el-button
                    type="primary"
                    size="large"
                    class="register-button"
                    :loading="loading"
                    @click="handleRegister"
                  >
                    注册
                  </el-button>
                </el-form-item>
              </el-form>
            </el-tab-pane>
          </el-tabs>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { mapActions } from 'vuex'

export default {
  name: 'Login',
  data() {
    // 密码确认验证
    const validateConfirmPassword = (rule, value, callback) => {
      if (value !== this.registerForm.password) {
        callback(new Error('两次输入的密码不一致'))
      } else {
        callback()
      }
    }
    
    return {
      activeTab: 'login',
      loading: false,
      loginForm: {
        username: '',
        password: ''
      },
      registerForm: {
        username: '',
        email: '',
        full_name: '',
        password: '',
        confirm_password: ''
      },
      loginRules: {
        username: [
          { required: true, message: '请输入用户名', trigger: 'blur' },
          { min: 3, max: 20, message: '用户名长度在 3 到 20 个字符', trigger: 'blur' }
        ],
        password: [
          { required: true, message: '请输入密码', trigger: 'blur' },
          { min: 6, message: '密码长度不能少于 6 个字符', trigger: 'blur' }
        ]
      },
      registerRules: {
        username: [
          { required: true, message: '请输入用户名', trigger: 'blur' },
          { min: 3, max: 20, message: '用户名长度在 3 到 20 个字符', trigger: 'blur' },
          { pattern: /^[a-zA-Z0-9_]+$/, message: '用户名只能包含字母、数字和下划线', trigger: 'blur' }
        ],
        email: [
          { required: true, message: '请输入邮箱', trigger: 'blur' },
          { type: 'email', message: '请输入正确的邮箱格式', trigger: 'blur' }
        ],
        full_name: [
          { max: 50, message: '全名长度不能超过 50 个字符', trigger: 'blur' }
        ],
        password: [
          { required: true, message: '请输入密码', trigger: 'blur' },
          { min: 6, message: '密码长度不能少于 6 个字符', trigger: 'blur' }
        ],
        confirm_password: [
          { required: true, message: '请确认密码', trigger: 'blur' },
          { validator: validateConfirmPassword, trigger: 'blur' }
        ]
      }
    }
  },
  methods: {
    ...mapActions('user', ['login', 'register']),
    
    async handleLogin() {
      try {
        await this.$refs.loginForm.validate()
        this.loading = true
        
        await this.login(this.loginForm)
        
        this.$message.success('登录成功')
        this.$router.push('/')
      } catch (error) {
        console.error('登录失败:', error)
        const message = error.response?.data?.detail || error.message || '登录失败，请重试'
        this.$message.error(message)
      } finally {
        this.loading = false
      }
    },
    
    async handleRegister() {
      try {
        await this.$refs.registerForm.validate()
        this.loading = true
        
        const registerData = {
          username: this.registerForm.username,
          email: this.registerForm.email,
          password: this.registerForm.password,
          full_name: this.registerForm.full_name || undefined
        }
        
        await this.register(registerData)
        
        this.$message.success('注册成功，请登录')
        this.activeTab = 'login'
        this.loginForm.username = this.registerForm.username
        this.$refs.registerForm.resetFields()
      } catch (error) {
        console.error('注册失败:', error)
        const message = error.response?.data?.detail || error.message || '注册失败，请重试'
        this.$message.error(message)
      } finally {
        this.loading = false
      }
    }
  }
}
</script>

<style lang="scss" scoped>
.login-container {
  height: 100vh;
  display: flex;
  align-items: center;
  justify-content: center;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}

.login-background {
  width: 100%;
  max-width: 1200px;
  margin: 0 auto;
  padding: 20px;
}

.login-content {
  display: flex;
  background: rgba(255, 255, 255, 0.95);
  border-radius: 20px;
  box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
  overflow: hidden;
  backdrop-filter: blur(10px);
}

.login-header {
  flex: 1;
  padding: 60px 40px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  text-align: center;
  
  .logo {
    width: 80px;
    height: 80px;
    margin-bottom: 20px;
    filter: brightness(0) invert(1);
  }
  
  .title {
    font-size: 2.5rem;
    font-weight: 700;
    margin-bottom: 15px;
    text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
  }
  
  .subtitle {
    font-size: 1.1rem;
    opacity: 0.9;
    line-height: 1.6;
    max-width: 400px;
  }
}

.login-form-container {
  flex: 1;
  padding: 60px 40px;
  background: white;
  
  .login-tabs {
    max-width: 400px;
    margin: 0 auto;
    
    ::v-deep .el-tabs__header {
      margin-bottom: 30px;
    }
    
    ::v-deep .el-tabs__item {
      font-size: 1.1rem;
      font-weight: 600;
    }
  }
  
  .login-form,
  .register-form {
    .el-form-item {
      margin-bottom: 25px;
    }
    
    .el-input {
      ::v-deep .el-input__inner {
        height: 50px;
        border-radius: 8px;
        border: 2px solid #e1e5e9;
        transition: all 0.3s ease;
        
        &:focus {
          border-color: #667eea;
          box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }
      }
    }
    
    .login-button,
    .register-button {
      width: 100%;
      height: 50px;
      border-radius: 8px;
      font-size: 1.1rem;
      font-weight: 600;
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      border: none;
      transition: all 0.3s ease;
      
      &:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3);
      }
      
      &:active {
        transform: translateY(0);
      }
    }
  }
}

// 响应式设计
@media (max-width: 768px) {
  .login-content {
    flex-direction: column;
  }
  
  .login-header {
    padding: 40px 20px;
    
    .title {
      font-size: 2rem;
    }
    
    .subtitle {
      font-size: 1rem;
    }
  }
  
  .login-form-container {
    padding: 40px 20px;
  }
}

@media (max-width: 480px) {
  .login-background {
    padding: 10px;
  }
  
  .login-header {
    padding: 30px 15px;
    
    .title {
      font-size: 1.8rem;
    }
  }
  
  .login-form-container {
    padding: 30px 15px;
  }
}
</style> 
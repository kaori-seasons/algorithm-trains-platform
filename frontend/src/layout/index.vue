<template>
  <div
    class="app-wrapper"
    :class="{ collapsed: !sidebarOpened }"
  >
    <!-- 侧边栏 -->
    <div
      class="sidebar-container"
      :class="{ collapsed: !sidebarOpened }"
    >
      <div class="logo-container">
        <img
          src="@/assets/logo.svg"
          alt="Logo"
          class="logo"
        >
        <span
          v-show="sidebarOpened"
          class="title"
        >训练平台</span>
      </div>
      
      <el-menu
        :default-active="$route.path"
        :collapse="!sidebarOpened"
        :unique-opened="true"
        router
        class="sidebar-menu"
        background-color="#304156"
        text-color="#bfcbd9"
        active-text-color="#409EFF"
      >
        <sidebar-item
          v-for="route in routes"
          :key="route.path"
          :item="route"
          :base-path="route.path"
        />
      </el-menu>
    </div>

    <!-- 主内容区 -->
    <div class="main-container">
      <!-- 顶部导航栏 -->
      <div class="navbar">
        <div class="navbar-left">
          <el-button
            type="text"
            icon="el-icon-menu"
            class="hamburger-btn"
            @click="toggleSidebar"
          />
          <breadcrumb class="breadcrumb" />
        </div>
        
        <div class="navbar-right">
          <el-dropdown
            trigger="click"
            @command="handleCommand"
          >
            <div class="avatar-container">
              <el-avatar
                :size="32"
                :src="userInfo.avatar"
              >
                {{ userInfo.name ? userInfo.name.charAt(0) : 'U' }}
              </el-avatar>
              <span class="username">{{ userInfo.name || '用户' }}</span>
              <i class="el-icon-arrow-down el-icon--right" />
            </div>
            <el-dropdown-menu slot="dropdown">
              <el-dropdown-item command="profile">
                <i class="el-icon-user" /> 个人资料
              </el-dropdown-item>
              <el-dropdown-item command="settings">
                <i class="el-icon-setting" /> 系统设置
              </el-dropdown-item>
              <el-dropdown-item
                divided
                command="logout"
              >
                <i class="el-icon-switch-button" /> 退出登录
              </el-dropdown-item>
            </el-dropdown-menu>
          </el-dropdown>
        </div>
      </div>

      <!-- 内容区域 -->
      <div class="app-main">
        <transition
          name="fade-transform"
          mode="out-in"
        >
          <router-view />
        </transition>
      </div>
    </div>
  </div>
</template>

<script>
import { mapState, mapActions } from 'vuex'
import SidebarItem from './components/SidebarItem.vue'
import Breadcrumb from './components/Breadcrumb.vue'

export default {
  name: 'Layout',
  components: {
    SidebarItem,
    Breadcrumb
  },
  computed: {
    ...mapState({
      sidebarOpened: state => state.sidebar.opened,
      userInfo: state => state.user.userInfo
    }),
    routes() {
      return this.$router.options.routes.filter(route => !route.meta?.hidden)
    }
  },
  methods: {
    ...mapActions(['toggleSidebar']),
    handleCommand(command) {
      switch (command) {
        case 'profile':
          this.$router.push('/user/profile')
          break
        case 'settings':
          this.$router.push('/user/settings')
          break
        case 'logout':
          this.handleLogout()
          break
      }
    },
    async handleLogout() {
      try {
        await this.$store.dispatch('user/logout')
        this.$message.success('退出登录成功')
        this.$router.push('/login')
      } catch (error) {
        this.$message.error('退出登录失败')
      }
    }
  }
}
</script>

<style lang="scss" scoped>
.app-wrapper {
  height: 100vh;
  display: flex;
  
  .sidebar-container {
    width: 210px;
    height: 100%;
    background: #304156;
    transition: width 0.3s;
    display: flex;
    flex-direction: column;
    
    &.collapsed {
      width: 64px;
    }
    
    .logo-container {
      height: 60px;
      display: flex;
      align-items: center;
      padding: 0 16px;
      background: #2b2f3a;
      
      .logo {
        width: 32px;
        height: 32px;
        margin-right: 12px;
      }
      
      .title {
        color: #fff;
        font-size: 16px;
        font-weight: 600;
        white-space: nowrap;
      }
    }
    
    .sidebar-menu {
      flex: 1;
      border: none;
    }
  }
  
  .main-container {
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    
    .navbar {
      height: 60px;
      background: #fff;
      border-bottom: 1px solid #e6e6e6;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 20px;
      
      .navbar-left {
        display: flex;
        align-items: center;
        
        .hamburger-btn {
          margin-right: 16px;
          font-size: 18px;
        }
        
        .breadcrumb {
          flex: 1;
        }
      }
      
      .navbar-right {
        .avatar-container {
          display: flex;
          align-items: center;
          cursor: pointer;
          padding: 8px;
          border-radius: 4px;
          transition: background-color 0.3s;
          
          &:hover {
            background-color: #f5f5f5;
          }
          
          .username {
            margin: 0 8px;
            color: #333;
          }
        }
      }
    }
    
    .app-main {
      flex: 1;
      padding: 20px;
      overflow-y: auto;
      background: #f0f2f5;
    }
  }
}

// 响应式设计
@media (max-width: 768px) {
  .app-wrapper {
    .sidebar-container {
      position: fixed;
      left: 0;
      top: 0;
      z-index: 1000;
      transform: translateX(-100%);
      transition: transform 0.3s;
      
      &.collapsed {
        transform: translateX(0);
      }
    }
    
    .main-container {
      margin-left: 0;
    }
  }
}

// 过渡动画
.fade-transform-enter-active,
.fade-transform-leave-active {
  transition: all 0.3s;
}

.fade-transform-enter {
  opacity: 0;
  transform: translateX(-30px);
}

.fade-transform-leave-to {
  opacity: 0;
  transform: translateX(30px);
}
</style> 
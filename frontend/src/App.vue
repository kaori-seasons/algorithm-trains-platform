<template>
  <div
    id="app"
    class="app-container"
  >
    <router-view />
  </div>
</template>

<script>
export default {
  name: 'App',
  mounted() {
    // 检查用户登录状态
    this.checkAuth()
  },
  methods: {
    checkAuth() {
      const token = localStorage.getItem('token')
      const currentPath = this.$route.path
      
      if (!token && currentPath !== '/login') {
        this.$router.push('/login')
      }
    }
  }
}
</script>

<style lang="scss">
.app-container {
  height: 100vh;
  display: flex;
  flex-direction: column;
  background-color: #f5f5f5;
}

// 全局样式重置
* {
  box-sizing: border-box;
}

html, body {
  height: 100%;
  margin: 0;
  padding: 0;
}

// 滚动条样式
::-webkit-scrollbar {
  width: 6px;
  height: 6px;
}

::-webkit-scrollbar-track {
  background: #f1f1f1;
  border-radius: 3px;
}

::-webkit-scrollbar-thumb {
  background: #c1c1c1;
  border-radius: 3px;
}

::-webkit-scrollbar-thumb:hover {
  background: #a8a8a8;
}

// 通用flex工具类
.flex {
  display: flex;
}

.flex-column {
  flex-direction: column;
}

.flex-row {
  flex-direction: row;
}

.flex-wrap {
  flex-wrap: wrap;
}

.flex-1 {
  flex: 1;
}

.flex-center {
  justify-content: center;
  align-items: center;
}

.flex-between {
  justify-content: space-between;
}

.flex-around {
  justify-content: space-around;
}

.flex-start {
  justify-content: flex-start;
}

.flex-end {
  justify-content: flex-end;
}

.items-center {
  align-items: center;
}

.items-start {
  align-items: flex-start;
}

.items-end {
  align-items: flex-end;
}

// 间距工具类
.m-0 { margin: 0; }
.m-1 { margin: 8px; }
.m-2 { margin: 16px; }
.m-3 { margin: 24px; }

.mt-0 { margin-top: 0; }
.mt-1 { margin-top: 8px; }
.mt-2 { margin-top: 16px; }
.mt-3 { margin-top: 24px; }

.mb-0 { margin-bottom: 0; }
.mb-1 { margin-bottom: 8px; }
.mb-2 { margin-bottom: 16px; }
.mb-3 { margin-bottom: 24px; }

.ml-0 { margin-left: 0; }
.ml-1 { margin-left: 8px; }
.ml-2 { margin-left: 16px; }
.ml-3 { margin-left: 24px; }

.mr-0 { margin-right: 0; }
.mr-1 { margin-right: 8px; }
.mr-2 { margin-right: 16px; }
.mr-3 { margin-right: 24px; }

.p-0 { padding: 0; }
.p-1 { padding: 8px; }
.p-2 { padding: 16px; }
.p-3 { padding: 24px; }

.pt-0 { padding-top: 0; }
.pt-1 { padding-top: 8px; }
.pt-2 { padding-top: 16px; }
.pt-3 { padding-top: 24px; }

.pb-0 { padding-bottom: 0; }
.pb-1 { padding-bottom: 8px; }
.pb-2 { padding-bottom: 16px; }
.pb-3 { padding-bottom: 24px; }

.pl-0 { padding-left: 0; }
.pl-1 { padding-left: 8px; }
.pl-2 { padding-left: 16px; }
.pl-3 { padding-left: 24px; }

.pr-0 { padding-right: 0; }
.pr-1 { padding-right: 8px; }
.pr-2 { padding-right: 16px; }
.pr-3 { padding-right: 24px; }
</style> 
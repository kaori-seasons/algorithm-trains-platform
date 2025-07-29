<template>
  <div v-if="!item.meta || !item.meta.hidden">
    <template v-if="!hasOneShowingChild(item.children, item) || (onlyOneChild.children && !onlyOneChild.noShowingChildren) || item.meta && item.meta.alwaysShow">
      <el-submenu
        :index="resolvePath(item.path)"
        popper-append-to-body
      >
        <template slot="title">
          <i
            v-if="item.meta && item.meta.icon"
            :class="item.meta && item.meta.icon"
          />
          <span v-if="item.meta && item.meta.title">{{ item.meta.title }}</span>
        </template>
        <sidebar-item
          v-for="child in item.children"
          :key="child.path"
          :item="child"
          :base-path="resolvePath(child.path)"
          class="nest-menu"
        />
      </el-submenu>
    </template>
    <template v-else>
      <el-menu-item
        :index="resolvePath(onlyOneChild.path)"
        @click="handleLink(onlyOneChild)"
      >
        <i
          v-if="onlyOneChild.meta && onlyOneChild.meta.icon"
          :class="onlyOneChild.meta && onlyOneChild.meta.icon"
        />
        <span v-if="onlyOneChild.meta && onlyOneChild.meta.title">{{ onlyOneChild.meta.title }}</span>
      </el-menu-item>
    </template>
  </div>
</template>

<script>
import path from 'path-browserify'

export default {
  name: 'SidebarItem',
  props: {
    item: {
      type: Object,
      required: true
    },
    basePath: {
      type: String,
      default: ''
    }
  },
  data() {
    this.onlyOneChild = null
    return {}
  },
  methods: {
    hasOneShowingChild(children = [], parent) {
      const showingChildren = children.filter(item => {
        if (item.meta && item.meta.hidden) {
          return false
        } else {
          this.onlyOneChild = item
          return true
        }
      })

      if (showingChildren.length === 1) {
        return true
      }

      if (showingChildren.length === 0) {
        this.onlyOneChild = { ...parent, path: '', noShowingChildren: true }
        return true
      }

      return false
    },
    resolvePath(routePath) {
      if (this.isExternal(routePath)) {
        return routePath
      }
      if (this.isExternal(this.basePath)) {
        return this.basePath
      }
      return path.resolve(this.basePath, routePath)
    },
    isExternal(path) {
      return /^(https?:|mailto:|tel:)/.test(path)
    },
    handleLink(item) {
      const { redirect, path } = item
      if (redirect) {
        this.$router.push(redirect)
        return
      }
      this.$router.push(path)
    }
  }
}
</script> 
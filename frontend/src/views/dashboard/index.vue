<template>
  <div class="dashboard-container">
    <!-- 统计卡片 -->
    <div class="stats-grid">
      <div
        v-for="stat in stats"
        :key="stat.title"
        class="stat-card"
      >
        <div
          class="stat-icon"
          :style="{ backgroundColor: stat.color }"
        >
          <i :class="stat.icon" />
        </div>
        <div class="stat-content">
          <div class="stat-value">
            {{ stat.value }}
          </div>
          <div class="stat-title">
            {{ stat.title }}
          </div>
          <div
            class="stat-change"
            :class="stat.trend"
          >
            <i :class="stat.trend === 'up' ? 'el-icon-top' : 'el-icon-bottom'" />
            {{ stat.change }}
          </div>
        </div>
      </div>
    </div>

    <!-- 图表区域 -->
    <div class="charts-grid">
      <div class="chart-card">
        <div class="chart-header">
          <h3>Pipeline执行趋势</h3>
          <el-select
            v-model="pipelineTimeRange"
            size="small"
          >
            <el-option
              label="最近7天"
              value="7d"
            />
            <el-option
              label="最近30天"
              value="30d"
            />
            <el-option
              label="最近90天"
              value="90d"
            />
          </el-select>
        </div>
        <div class="chart-content">
          <div
            ref="pipelineChart"
            class="chart"
          />
        </div>
      </div>

      <div class="chart-card">
        <div class="chart-header">
          <h3>存储使用情况</h3>
        </div>
        <div class="chart-content">
          <div
            ref="storageChart"
            class="chart"
          />
        </div>
      </div>
    </div>

    <!-- 最近活动 -->
    <div class="recent-activities">
      <div class="section-header">
        <h3>最近活动</h3>
        <el-button
          type="text"
          @click="viewAllActivities"
        >
          查看全部
        </el-button>
      </div>
      <div class="activity-list">
        <div
          v-for="activity in recentActivities"
          :key="activity.id"
          class="activity-item"
        >
          <div
            class="activity-icon"
            :class="activity.type"
          >
            <i :class="activity.icon" />
          </div>
          <div class="activity-content">
            <div class="activity-title">
              {{ activity.title }}
            </div>
            <div class="activity-desc">
              {{ activity.description }}
            </div>
            <div class="activity-time">
              {{ formatTime(activity.time) }}
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- 快速操作 -->
    <div class="quick-actions">
      <div class="section-header">
        <h3>快速操作</h3>
      </div>
      <div class="action-grid">
        <div
          v-for="action in quickActions"
          :key="action.title"
          class="action-item"
          @click="handleQuickAction(action)"
        >
          <div
            class="action-icon"
            :style="{ backgroundColor: action.color }"
          >
            <i :class="action.icon" />
          </div>
          <div class="action-title">
            {{ action.title }}
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import * as echarts from 'echarts'

export default {
  name: 'Dashboard',
  data() {
    return {
      pipelineTimeRange: '7d',
      stats: [
        {
          title: '活跃Pipeline',
          value: '12',
          change: '+2',
          trend: 'up',
          icon: 'el-icon-s-operation',
          color: '#409EFF'
        },
        {
          title: '运行中任务',
          value: '8',
          change: '-1',
          trend: 'down',
          icon: 'el-icon-loading',
          color: '#67C23A'
        },
        {
          title: '存储使用',
          value: '2.4TB',
          change: '+0.3TB',
          trend: 'up',
          icon: 'el-icon-folder',
          color: '#E6A23C'
        },
        {
          title: '模型版本',
          value: '156',
          change: '+12',
          trend: 'up',
          icon: 'el-icon-data-analysis',
          color: '#F56C6C'
        }
      ],
      recentActivities: [
        {
          id: 1,
          type: 'pipeline',
          icon: 'el-icon-s-operation',
          title: 'Pipeline执行完成',
          description: '图像分类训练Pipeline已成功完成',
          time: new Date(Date.now() - 1000 * 60 * 30)
        },
        {
          id: 2,
          type: 'storage',
          icon: 'el-icon-folder',
          title: '存储挂载成功',
          description: 'S3存储桶已成功挂载到训练环境',
          time: new Date(Date.now() - 1000 * 60 * 60)
        },
        {
          id: 3,
          type: 'model',
          icon: 'el-icon-data-analysis',
          title: '模型版本发布',
          description: '新版本模型v2.1.0已发布到生产环境',
          time: new Date(Date.now() - 1000 * 60 * 120)
        }
      ],
      quickActions: [
        {
          title: '创建Pipeline',
          icon: 'el-icon-plus',
          color: '#409EFF',
          route: '/pipeline/create'
        },
        {
          title: '数据查询',
          icon: 'el-icon-search',
          color: '#67C23A',
          route: '/doris/data'
        },
        {
          title: '特征管理',
          icon: 'el-icon-data-line',
          color: '#E6A23C',
          route: '/feast/features'
        },
        {
          title: '存储管理',
          icon: 'el-icon-folder-opened',
          color: '#F56C6C',
          route: '/storage/mounts'
        }
      ]
    }
  },
  mounted() {
    this.initCharts()
  },
  methods: {
    initCharts() {
      this.initPipelineChart()
      this.initStorageChart()
    },
    initPipelineChart() {
      const chart = echarts.init(this.$refs.pipelineChart)
      const option = {
        tooltip: {
          trigger: 'axis'
        },
        legend: {
          data: ['成功', '失败', '运行中']
        },
        xAxis: {
          type: 'category',
          data: ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
        },
        yAxis: {
          type: 'value'
        },
        series: [
          {
            name: '成功',
            type: 'line',
            data: [12, 15, 18, 14, 16, 20, 22],
            smooth: true,
            itemStyle: { color: '#67C23A' }
          },
          {
            name: '失败',
            type: 'line',
            data: [2, 1, 3, 2, 1, 0, 1],
            smooth: true,
            itemStyle: { color: '#F56C6C' }
          },
          {
            name: '运行中',
            type: 'line',
            data: [5, 8, 6, 9, 7, 4, 6],
            smooth: true,
            itemStyle: { color: '#E6A23C' }
          }
        ]
      }
      chart.setOption(option)
    },
    initStorageChart() {
      const chart = echarts.init(this.$refs.storageChart)
      const option = {
        tooltip: {
          trigger: 'item'
        },
        legend: {
          orient: 'vertical',
          left: 'left'
        },
        series: [
          {
            name: '存储使用',
            type: 'pie',
            radius: '50%',
            data: [
              { value: 1048, name: '训练数据' },
              { value: 735, name: '模型文件' },
              { value: 580, name: '日志文件' },
              { value: 484, name: '临时文件' }
            ],
            emphasis: {
              itemStyle: {
                shadowBlur: 10,
                shadowOffsetX: 0,
                shadowColor: 'rgba(0, 0, 0, 0.5)'
              }
            }
          }
        ]
      }
      chart.setOption(option)
    },
    formatTime(time) {
      const now = new Date()
      const diff = now - time
      const minutes = Math.floor(diff / (1000 * 60))
      const hours = Math.floor(diff / (1000 * 60 * 60))
      const days = Math.floor(diff / (1000 * 60 * 60 * 24))
      
      if (minutes < 60) {
        return `${minutes}分钟前`
      } else if (hours < 24) {
        return `${hours}小时前`
      } else {
        return `${days}天前`
      }
    },
    handleQuickAction(action) {
      this.$router.push(action.route)
    },
    viewAllActivities() {
      this.$router.push('/monitor/logs')
    }
  }
}
</script>

<style lang="scss" scoped>
.dashboard-container {
  .stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 20px;
    margin-bottom: 24px;
    
    .stat-card {
      background: #fff;
      border-radius: 8px;
      padding: 24px;
      display: flex;
      align-items: center;
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
      transition: transform 0.3s;
      
      &:hover {
        transform: translateY(-2px);
      }
      
      .stat-icon {
        width: 60px;
        height: 60px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-right: 16px;
        
        i {
          font-size: 24px;
          color: #fff;
        }
      }
      
      .stat-content {
        flex: 1;
        
        .stat-value {
          font-size: 28px;
          font-weight: 600;
          color: #333;
          margin-bottom: 4px;
        }
        
        .stat-title {
          font-size: 14px;
          color: #666;
          margin-bottom: 8px;
        }
        
        .stat-change {
          font-size: 12px;
          display: flex;
          align-items: center;
          
          &.up {
            color: #67C23A;
          }
          
          &.down {
            color: #F56C6C;
          }
          
          i {
            margin-right: 4px;
          }
        }
      }
    }
  }
  
  .charts-grid {
    display: grid;
    grid-template-columns: 2fr 1fr;
    gap: 20px;
    margin-bottom: 24px;
    
    .chart-card {
      background: #fff;
      border-radius: 8px;
      padding: 20px;
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
      
      .chart-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 16px;
        
        h3 {
          margin: 0;
          font-size: 16px;
          font-weight: 600;
          color: #333;
        }
      }
      
      .chart-content {
        .chart {
          height: 300px;
        }
      }
    }
  }
  
  .recent-activities,
  .quick-actions {
    background: #fff;
    border-radius: 8px;
    padding: 20px;
    margin-bottom: 24px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    
    .section-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 16px;
      
      h3 {
        margin: 0;
        font-size: 16px;
        font-weight: 600;
        color: #333;
      }
    }
  }
  
  .activity-list {
    .activity-item {
      display: flex;
      align-items: flex-start;
      padding: 12px 0;
      border-bottom: 1px solid #f0f0f0;
      
      &:last-child {
        border-bottom: none;
      }
      
      .activity-icon {
        width: 40px;
        height: 40px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-right: 12px;
        flex-shrink: 0;
        
        &.pipeline {
          background-color: #409EFF;
        }
        
        &.storage {
          background-color: #E6A23C;
        }
        
        &.model {
          background-color: #67C23A;
        }
        
        i {
          color: #fff;
          font-size: 16px;
        }
      }
      
      .activity-content {
        flex: 1;
        
        .activity-title {
          font-size: 14px;
          font-weight: 500;
          color: #333;
          margin-bottom: 4px;
        }
        
        .activity-desc {
          font-size: 12px;
          color: #666;
          margin-bottom: 4px;
        }
        
        .activity-time {
          font-size: 12px;
          color: #999;
        }
      }
    }
  }
  
  .action-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
    gap: 16px;
    
    .action-item {
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 20px;
      border-radius: 8px;
      cursor: pointer;
      transition: all 0.3s;
      
      &:hover {
        background-color: #f5f5f5;
        transform: translateY(-2px);
      }
      
      .action-icon {
        width: 48px;
        height: 48px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 8px;
        
        i {
          color: #fff;
          font-size: 20px;
        }
      }
      
      .action-title {
        font-size: 12px;
        color: #333;
        text-align: center;
      }
    }
  }
}

// 响应式设计
@media (max-width: 768px) {
  .dashboard-container {
    .stats-grid {
      grid-template-columns: 1fr;
    }
    
    .charts-grid {
      grid-template-columns: 1fr;
    }
    
    .action-grid {
      grid-template-columns: repeat(2, 1fr);
    }
  }
}
</style> 
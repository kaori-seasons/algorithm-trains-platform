const state = {
  pipelines: [],
  currentPipeline: null,
  loading: false
}

const mutations = {
  SET_PIPELINES: (state, pipelines) => {
    state.pipelines = pipelines
  },
  SET_CURRENT_PIPELINE: (state, pipeline) => {
    state.currentPipeline = pipeline
  },
  SET_LOADING: (state, loading) => {
    state.loading = loading
  }
}

const actions = {
  // 获取Pipeline列表
  getPipelines({ commit }) {
    commit('SET_LOADING', true)
    // 这里应该调用API
    return new Promise((resolve) => {
      setTimeout(() => {
        const pipelines = [
          {
            id: 1,
            name: '图像分类训练',
            status: 'running',
            createTime: '2024-01-15 10:30:00',
            updateTime: '2024-01-15 11:45:00'
          },
          {
            id: 2,
            name: '文本分类Pipeline',
            status: 'completed',
            createTime: '2024-01-14 09:15:00',
            updateTime: '2024-01-14 16:20:00'
          }
        ]
        commit('SET_PIPELINES', pipelines)
        commit('SET_LOADING', false)
        resolve(pipelines)
      }, 1000)
    })
  }
}

export default {
  namespaced: true,
  state,
  mutations,
  actions
} 
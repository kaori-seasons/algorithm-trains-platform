const state = {
  metrics: {},
  logs: [],
  alerts: [],
  loading: false
}

const mutations = {
  SET_METRICS: (state, metrics) => {
    state.metrics = metrics
  },
  SET_LOGS: (state, logs) => {
    state.logs = logs
  },
  SET_ALERTS: (state, alerts) => {
    state.alerts = alerts
  },
  SET_LOADING: (state, loading) => {
    state.loading = loading
  }
}

const actions = {
  getMetrics({ commit }) {
    commit('SET_LOADING', true)
    return new Promise((resolve) => {
      setTimeout(() => {
        const metrics = {}
        commit('SET_METRICS', metrics)
        commit('SET_LOADING', false)
        resolve(metrics)
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
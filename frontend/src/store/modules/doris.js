const state = {
  tables: [],
  data: [],
  loading: false
}

const mutations = {
  SET_TABLES: (state, tables) => {
    state.tables = tables
  },
  SET_DATA: (state, data) => {
    state.data = data
  },
  SET_LOADING: (state, loading) => {
    state.loading = loading
  }
}

const actions = {
  getTables({ commit }) {
    commit('SET_LOADING', true)
    return new Promise((resolve) => {
      setTimeout(() => {
        const tables = []
        commit('SET_TABLES', tables)
        commit('SET_LOADING', false)
        resolve(tables)
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
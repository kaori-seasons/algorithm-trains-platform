const state = {
  mounts: [],
  providers: [],
  loading: false
}

const mutations = {
  SET_MOUNTS: (state, mounts) => {
    state.mounts = mounts
  },
  SET_PROVIDERS: (state, providers) => {
    state.providers = providers
  },
  SET_LOADING: (state, loading) => {
    state.loading = loading
  }
}

const actions = {
  getMounts({ commit }) {
    commit('SET_LOADING', true)
    return new Promise((resolve) => {
      setTimeout(() => {
        const mounts = []
        commit('SET_MOUNTS', mounts)
        commit('SET_LOADING', false)
        resolve(mounts)
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
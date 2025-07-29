const state = {
  features: [],
  trainingSets: [],
  loading: false
}

const mutations = {
  SET_FEATURES: (state, features) => {
    state.features = features
  },
  SET_TRAINING_SETS: (state, trainingSets) => {
    state.trainingSets = trainingSets
  },
  SET_LOADING: (state, loading) => {
    state.loading = loading
  }
}

const actions = {
  getFeatures({ commit }) {
    commit('SET_LOADING', true)
    return new Promise((resolve) => {
      setTimeout(() => {
        const features = []
        commit('SET_FEATURES', features)
        commit('SET_LOADING', false)
        resolve(features)
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
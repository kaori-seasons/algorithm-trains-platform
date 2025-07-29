import Vue from 'vue'
import Vuex from 'vuex'
import user from './modules/user'
import pipeline from './modules/pipeline'
import feast from './modules/feast'
import doris from './modules/doris'
import storage from './modules/storage'
import monitor from './modules/monitor'

Vue.use(Vuex)

export default new Vuex.Store({
  modules: {
    user,
    pipeline,
    feast,
    doris,
    storage,
    monitor
  },
  state: {
    loading: false,
    sidebar: {
      opened: true
    }
  },
  mutations: {
    SET_LOADING(state, loading) {
      state.loading = loading
    },
    TOGGLE_SIDEBAR(state) {
      state.sidebar.opened = !state.sidebar.opened
    }
  },
  actions: {
    setLoading({ commit }, loading) {
      commit('SET_LOADING', loading)
    },
    toggleSidebar({ commit }) {
      commit('TOGGLE_SIDEBAR')
    }
  }
}) 
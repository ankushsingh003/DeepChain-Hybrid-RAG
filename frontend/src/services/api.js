import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export const queryRAG = async (question, method = 'hybrid') => {
  try {
    const response = await axios.post(`${API_BASE_URL}/query`, {
      question,
      method
    })
    return response.data
  } catch (error) {
    console.error('API Error:', error)
    throw error
  }
}

export const triggerIngestion = async () => {
  try {
    const response = await axios.post(`${API_BASE_URL}/ingest`)
    return response.data
  } catch (error) {
    console.error('Ingestion Error:', error)
    throw error
  }
}

export const checkHealth = async () => {
  const response = await axios.get(`${API_BASE_URL}/health`)
  return response.data
}

export const getPortfolioStrategy = async (profile) => {
  try {
    const response = await axios.post(`${API_BASE_URL}/finance/portfolio`, profile)
    return response.data
  } catch (error) {
    console.error('Portfolio API Error:', error)
    throw error
  }
}

export const runTradeTest = async (symbol, strategy, period = '1y') => {
  try {
    const response = await axios.post(`${API_BASE_URL}/finance/trade-test`, {
      symbol,
      strategy,
      period
    })
    return response.data
  } catch (error) {
    console.error('Trade Test API Error:', error)
    throw error
  }
}

export const getStrategyAdvice = async (intent) => {
  try {
    const response = await axios.post(`${API_BASE_URL}/finance/strategy-advisor`, {
      intent
    })
    return response.data
  } catch (error) {
    console.error('Strategy Advisor API Error:', error)
    throw error
  }
}

export const getMarketStrategyAdvice = async (symbol) => {
  try {
    const response = await axios.post(`${API_BASE_URL}/finance/market-advisor`, {
      symbol
    })
    return response.data
  } catch (error) {
    console.error('Market Advisor API Error:', error)
    throw error
  }
}
export const getMLAdvisory = async (symbol) => {
  try {
    const response = await axios.post(`${API_BASE_URL}/finance/ml-stock-advisory`, { symbol })
    return response.data
  } catch (error) {
    console.error('ML Advisory Error:', error)
    throw error
  }
}

export const triggerMLTraining = async (quick = false) => {
  try {
    const response = await axios.post(`${API_BASE_URL}/finance/ml-train`, { quick })
    return response.data
  } catch (error) {
    console.error('ML Training Error:', error)
    throw error
  }
}

export const getMLModelStatus = async () => {
  try {
    const response = await axios.get(`${API_BASE_URL}/finance/ml-model-status`)
    return response.data
  } catch (error) {
    console.error('ML Status Error:', error)
    throw error
  }
}

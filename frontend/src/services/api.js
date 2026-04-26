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

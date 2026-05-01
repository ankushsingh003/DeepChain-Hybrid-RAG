import React, { useState, useEffect } from 'react'
import Landing from './components/Landing'
import DomainSelection from './components/DomainSelection'
import Consultation from './components/Consultation'

function App() {
  const [view, setView] = useState(() => localStorage.getItem('deepchain_view') || 'landing')
  const [selectedDomain, setSelectedDomain] = useState(() => {
    const saved = localStorage.getItem('deepchain_domain')
    return saved ? JSON.parse(saved) : null
  })

  useEffect(() => {
    localStorage.setItem('deepchain_view', view)
  }, [view])

  useEffect(() => {
    if (selectedDomain) {
      localStorage.setItem('deepchain_domain', JSON.stringify(selectedDomain))
    }
  }, [selectedDomain])

  const navigateToDomains = () => setView('domains')
  const navigateToLanding = () => {
    setView('landing')
    localStorage.removeItem('deepchain_domain')
  }
  const startConsultation = (domain) => {
    setSelectedDomain(domain)
    setView('consultation')
  }

  return (
    <div className="relative z-10 min-h-screen">
      {view === 'landing' && <Landing onGetConsulted={navigateToDomains} />}
      {view === 'domains' && (
        <DomainSelection 
          onBack={navigateToLanding} 
          onSelectDomain={startConsultation} 
        />
      )}
      {view === 'consultation' && (
        <Consultation 
          domain={selectedDomain} 
          onBack={navigateToDomains} 
        />
      )}
    </div>
  )
}

export default App

import React, { useState, useEffect } from 'react'
import Landing from './components/Landing'
import DomainSelection from './components/DomainSelection'
import Consultation from './components/Consultation'

function App() {
  const [view, setView] = useState(() => {
    try {
      const saved = localStorage.getItem('deepchain_view')
      if (['landing', 'domains', 'consultation'].includes(saved)) return saved
    } catch (e) {}
    return 'landing'
  })

  const [selectedDomain, setSelectedDomain] = useState(() => {
    try {
      const saved = localStorage.getItem('deepchain_domain')
      return saved ? JSON.parse(saved) : null
    } catch (e) { return null }
  })

  useEffect(() => {
    localStorage.setItem('deepchain_view', view)
  }, [view])

  // VALIDATION: Version check to clear old incompatible data
  useEffect(() => {
    try {
      const CURRENT_VERSION = "2.1";
      const savedVersion = localStorage.getItem('deepchain_version');
      if (savedVersion !== CURRENT_VERSION) {
        localStorage.removeItem('deepchain_view');
        localStorage.removeItem('deepchain_domain');
        localStorage.removeItem('deepchain_viewmode');
        localStorage.setItem('deepchain_version', CURRENT_VERSION);
        // No hard reload here to avoid loops
      }
    } catch (e) {
      console.error("Storage error:", e);
    }
  }, [])

  // VALIDATION: If we are in consultation without a domain, rescue to domains
  useEffect(() => {
    if (view === 'consultation' && !selectedDomain) {
      console.warn("Rescue: No domain found for consultation. Redirecting...")
      setView('domains')
    }
  }, [view, selectedDomain])

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
    <div className="relative z-50 min-h-screen bg-red-900/20">
      <h1 className="fixed top-0 left-0 z-[100] bg-white text-black p-2 text-xs font-mono">App.jsx Mounted: {view}</h1>
      
      {view === 'landing' && <Landing onGetConsulted={navigateToDomains} />}
      {view === 'domains' && (
        <DomainSelection 
          onBack={navigateToLanding} 
          onSelectDomain={startConsultation} 
        />
      )}
      {view === 'consultation' && selectedDomain ? (
        <Consultation 
          domain={selectedDomain} 
          onBack={navigateToDomains} 
        />
      ) : view === 'consultation' ? (
        <DomainSelection 
          onBack={navigateToLanding} 
          onSelectDomain={startConsultation} 
        />
      ) : null}
    </div>
  )
}

export default App

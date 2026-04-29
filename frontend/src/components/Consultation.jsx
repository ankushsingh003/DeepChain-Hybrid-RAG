import React, { useState, useRef, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  Send, ArrowLeft, MoreHorizontal, Settings, 
  Info, Database, Share2, Sparkles, Network,
  BarChart2, FileText, ChevronRight, Activity,
  RefreshCw, CheckCircle2, Circle, PieChart, TrendingUp,
  ShieldCheck, Wallet, Briefcase, Calculator, Heart
} from 'lucide-react'
import { queryRAG, triggerIngestion, getPortfolioStrategy, runTradeTest } from '../services/api'

const Consultation = ({ domain, onBack }) => {
  const [messages, setMessages] = useState([
    { 
      role: 'assistant', 
      content: `Hello. I am your DeepChain consultant for ${domain.name}. How can I assist with your domain research today?`,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    }
  ])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [ingesting, setIngesting] = useState(false)
  const [method, setMethod] = useState('hybrid') // naive, graph, hybrid
  const [pipelineStep, setPipelineStep] = useState(0) // 0: Idle, 1: Extract, 2: Retrieve, 3: Fuse, 4: Generate
  const [viewMode, setViewMode] = useState('chat') // chat, portfolio, tradetest
  
  // Finance Pipeline State
  const [portfolioData, setPortfolioData] = useState(null)
  const [tradeTestData, setTradeTestData] = useState(null)
  const [financeLoading, setFinanceLoading] = useState(false)
  const [profileForm, setProfileForm] = useState({
    age: 30,
    monthly_income: 50000,
    monthly_expenses: 20000,
    pension: 0,
    govt_allowances: 0,
    additional_income: 0,
    dependents: 0,
    existing_savings: 0,
    emergency_fund_exists: false,
    amount_to_invest: 10000,
    liabilities: [],
    life_insurance: false,
    health_insurance: false,
    investment_horizon: "5yr",
    primary_goal: "Wealth Creation"
  })
  const [tradeForm, setTradeForm] = useState({
    symbol: 'RELIANCE.NS',
    strategy: 'SMA_Crossover',
    period: '1y'
  })
  const scrollRef = useRef(null)

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [messages])

  const handleSend = async () => {
    if (!input.trim() || loading) return

    const userMsg = { role: 'user', content: input, timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) }
    setMessages(prev => [...prev, userMsg])
    setInput('')
    setLoading(true)
    
    // Simulate pipeline steps for UI effect
    setPipelineStep(1)
    setTimeout(() => setPipelineStep(2), 800)
    setTimeout(() => setPipelineStep(3), 1600)
    setTimeout(() => setPipelineStep(4), 2400)

    try {
      const data = await queryRAG(input, method)
      
      const assistantMsg = { 
        role: 'assistant', 
        content: data.answer,
        method: data.method,
        fallback: data.fallback_reason,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        citations: ['SEC-2024-Q3', 'Risk-Protocol-v2'], // Simulated citations
        graph: [ // Simulated graph snippets
          { s: 'Entity A', p: 'OWNS', o: 'Entity B' },
          { s: 'Entity B', p: 'LOCATED_IN', o: 'Singapore' }
        ]
      }
      
      setMessages(prev => [...prev, assistantMsg])
      setPipelineStep(0)
    } catch (err) {
      setMessages(prev => [...prev, { role: 'assistant', content: "I'm sorry, I encountered an error connecting to the intelligence engine.", timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) }])
    } finally {
      setLoading(false)
    }
  }

  const handleIngestion = async () => {
    setIngesting(true)
    try {
      await triggerIngestion()
      alert('Ingestion completed successfully!')
    } catch (err) {
      alert('Ingestion failed. Check server logs.')
    } finally {
      setIngesting(false)
    }
  }

  const handlePortfolioRun = async () => {
    setFinanceLoading(true)
    try {
      const data = await getPortfolioStrategy(profileForm)
      setPortfolioData(data)
    } catch (err) {
      alert('Portfolio generation failed. Check API status.')
    } finally {
      setFinanceLoading(false)
    }
  }

  const handleTradeTestRun = async () => {
    setFinanceLoading(true)
    try {
      const data = await runTradeTest(tradeForm.symbol, tradeForm.strategy, tradeForm.period)
      setTradeTestData(data)
    } catch (err) {
      alert('Trade test failed. Check API status.')
    } finally {
      setFinanceLoading(false)
    }
  }

  return (
    <div className="flex h-[calc(100vh-65px)] overflow-hidden bg-bg">
      {/* LEFT SIDEBAR */}
      <aside className="w-[300px] flex-shrink-0 border-r border-border p-8 flex flex-col gap-5 bg-bg2 overflow-y-auto">
        <div className="flex items-center gap-3 p-4 rounded-2xl border border-border bg-bg3">
          <div className={`w-10 h-10 rounded-xl flex items-center justify-center text-lg text-${domain.color} bg-${domain.id}-bg border border-${domain.id}/20`}>
            {domain.icon}
          </div>
          <div>
            <div className="text-[14px] font-medium">{domain.name}</div>
            <div className="text-[11px] text-muted font-mono tracking-tight">{domain.tag}</div>
          </div>
        </div>

        <div className="mt-4">
          <h3 className="text-[10px] font-mono tracking-[0.1em] uppercase text-dim mb-3 px-1">Knowledge Sources</h3>
          <div className="flex flex-col gap-1.5">
            {[
              { name: 'SEC_FILINGS.PDF', count: '1,240' },
              { name: 'MARKET_DATA.CSV', count: '842' },
              { name: 'CORP_STATUTES.JSON', count: '5,100' }
            ].map((s, i) => (
              <div key={i} className="flex items-center gap-2 text-[12px] text-muted p-2 rounded-lg border border-border bg-bg hover:bg-bg3 transition-colors">
                <FileText size={14} className="text-dim" />
                <span className="flex-1 font-mono text-[11px] truncate">{s.name}</span>
                <span className="text-[10px] text-dim font-mono">{s.count}</span>
              </div>
            ))}
          </div>
        </div>

        <div className="mt-4 p-4 rounded-2xl border border-border bg-bg3">
          <h3 className="text-[10px] font-mono text-dim uppercase tracking-[0.08em] mb-3">System Controls</h3>
          <button 
            onClick={handleIngestion}
            disabled={ingesting}
            className="w-full flex items-center justify-center gap-2 py-2.5 px-4 rounded-xl bg-finance/10 border border-finance/20 text-finance text-[11px] font-mono hover:bg-finance/20 transition-all disabled:opacity-50"
          >
            {ingesting ? (
              <>
                <RefreshCw size={12} className="animate-spin" />
                Ingesting...
              </>
            ) : (
              <>
                <Database size={12} />
                Trigger Ingestion
              </>
            )}
          </button>
        </div>

        <div className="mt-4 p-4 rounded-2xl border border-border bg-bg3">
          <h3 className="text-[10px] font-mono text-dim uppercase tracking-[0.08em] mb-3">Live Pipeline</h3>
          <div className="flex flex-col gap-2.5">
            {[
              { label: 'Query Decomposition', id: 1 },
              { label: 'Graph Retrieval', id: 2 },
              { label: 'Vector Fusion', id: 3 },
              { label: 'Generation', id: 4 }
            ].map((step) => (
              <div key={step.id} className={`flex items-center gap-2.5 text-[11px] font-mono ${pipelineStep === step.id ? 'text-health' : pipelineStep > step.id ? 'text-finance' : 'text-muted'}`}>
                {pipelineStep === step.id ? (
                  <RefreshCw size={10} className="animate-spin" />
                ) : pipelineStep > step.id ? (
                  <CheckCircle2 size={10} />
                ) : (
                  <Circle size={10} className="opacity-20" />
                )}
                {step.label}
              </div>
            ))}
          </div>
        </div>

        {domain.id === 'finance' && (
          <div className="mt-4 p-4 rounded-2xl border border-border bg-bg3">
            <h3 className="text-[10px] font-mono text-dim uppercase tracking-[0.08em] mb-3">Financial Toolkit</h3>
            <div className="flex flex-col gap-2">
              <button 
                onClick={() => setViewMode('chat')}
                className={`w-full flex items-center gap-2.5 py-2 px-3 rounded-lg text-[11px] font-mono transition-all ${viewMode === 'chat' ? 'bg-finance/10 text-finance border border-finance/20' : 'text-muted hover:bg-bg'}`}
              >
                <Sparkles size={12} />
                Expert Consultation
              </button>
              <button 
                onClick={() => setViewMode('portfolio')}
                className={`w-full flex items-center gap-2.5 py-2 px-3 rounded-lg text-[11px] font-mono transition-all ${viewMode === 'portfolio' ? 'bg-finance/10 text-finance border border-finance/20' : 'text-muted hover:bg-bg'}`}
              >
                <PieChart size={12} />
                Portfolio Allocation
              </button>
              <button 
                onClick={() => setViewMode('tradetest')}
                className={`w-full flex items-center gap-2.5 py-2 px-3 rounded-lg text-[11px] font-mono transition-all ${viewMode === 'tradetest' ? 'bg-finance/10 text-finance border border-finance/20' : 'text-muted hover:bg-bg'}`}
              >
                <TrendingUp size={12} />
                Trade Simulator
              </button>
            </div>
          </div>
        )}
        
        <button onClick={onBack} className="mt-auto flex items-center justify-center gap-2 p-3 rounded-xl border border-border text-muted text-xs hover:text-text hover:border-border2 transition-all">
          <ArrowLeft size={14} /> Back to Domains
        </button>
      </aside>

      {/* CENTER CHAT */}
      <main className="flex-1 flex flex-col min-w-0">
        <header className="px-8 py-5 border-b border-border flex items-center justify-between bg-bg/85 backdrop-blur-md">
          <div className="flex flex-col gap-0.5">
            <div className="text-[13px] font-medium tracking-tight">Active Consultation</div>
            <div className="text-[11px] text-muted font-mono flex items-center gap-1.5">
              <div className="w-1.5 h-1.5 rounded-full bg-health animate-pulse" />
              Engine Online — Gemini 1.5 Pro
            </div>
          </div>
          <div className="flex gap-2">
            {viewMode === 'chat' && (
              <>
                <button 
                  onClick={() => setMethod('naive')} 
                  className={`text-[10px] font-mono border rounded-md px-2.5 py-1 transition-all ${method === 'naive' ? 'text-finance border-finance/30 bg-finance/5' : 'text-muted border-border hover:border-border2'}`}
                >
                  Naive RAG
                </button>
                <button 
                  onClick={() => setMethod('hybrid')} 
                  className={`text-[10px] font-mono border rounded-md px-2.5 py-1 transition-all ${method === 'hybrid' ? 'text-finance border-finance/30 bg-finance/5' : 'text-muted border-border hover:border-border2'}`}
                >
                  Hybrid RAG
                </button>
              </>
            )}
            {viewMode !== 'chat' && (
              <button 
                onClick={() => setViewMode('chat')}
                className="text-[10px] font-mono border border-border rounded-md px-2.5 py-1 text-muted hover:text-text transition-all flex items-center gap-1.5"
              >
                <ArrowLeft size={10} /> Back to Chat
              </button>
            )}
          </div>
        </header>

        <div ref={scrollRef} className="flex-1 overflow-y-auto p-8 flex flex-col gap-6 custom-scrollbar">
          {viewMode === 'chat' && (
            <>
              {messages.map((msg, i) => (
                <motion.div 
                  key={i}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className={`max-w-[640px] ${msg.role === 'user' ? 'self-end' : 'self-start'}`}
                >
                  <div className="flex items-center gap-2 mb-2">
                    <div className={`w-6 h-6 rounded-md flex items-center justify-center text-[11px] font-mono border ${msg.role === 'assistant' ? 'bg-bg3 border-border text-muted' : 'bg-white/10 border-border text-text'}`}>
                      {msg.role === 'assistant' ? 'AI' : 'U'}
                    </div>
                    <span className="text-[11px] text-dim font-mono">{msg.role === 'assistant' ? 'DeepChain' : 'You'} — {msg.timestamp}</span>
                  </div>
                  
                  <div className={`${msg.role === 'user' ? 'bg-bg3 border border-border rounded-2xl rounded-tr-sm p-4' : ''}`}>
                    <p className="text-[14px] leading-[1.7] text-text whitespace-pre-wrap">{msg.content}</p>
                    
                    {msg.role === 'assistant' && msg.citations && (
                      <div className="flex flex-wrap gap-1.5 mt-4">
                        {msg.citations.map((c, j) => (
                          <span key={j} className="inline-flex items-center gap-1.5 text-[10px] font-mono text-muted border border-border rounded-full px-2.5 py-1 bg-bg2">
                            <FileText size={10} /> {c}
                          </span>
                        ))}
                      </div>
                    )}

                    {msg.role === 'assistant' && msg.graph && (
                      <div className="mt-4 border border-border rounded-xl overflow-hidden bg-bg3">
                        <div className="px-4 py-2 border-b border-border text-[10px] font-mono text-dim flex items-center justify-between">
                          <span>Graph Extraction</span>
                          <Share2 size={10} />
                        </div>
                        <div className="p-4 flex flex-col gap-2">
                          {msg.graph.map((triplet, j) => (
                            <div key={j} className="flex items-center gap-2 text-[10px] font-mono">
                              <span className="px-2 py-1 rounded border border-finance/30 text-finance bg-finance/5">{triplet.s}</span>
                              <span className="text-dim">→ {triplet.p} →</span>
                              <span className="px-2 py-1 rounded border border-legal/30 text-legal bg-legal/5">{triplet.o}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </motion.div>
              ))}
              {loading && (
                <div className="self-start max-w-[640px]">
                  <div className="flex items-center gap-2 mb-2">
                    <div className="w-6 h-6 rounded-md bg-bg3 border border-border flex items-center justify-center animate-pulse">
                      <Activity size={12} className="text-muted" />
                    </div>
                    <span className="text-[11px] text-dim font-mono">Consultant is thinking...</span>
                  </div>
                </div>
              )}
            </>
          )}

          {viewMode === 'portfolio' && (
            <div className="max-w-[800px] mx-auto w-full pb-20">
              <div className="mb-8 border-b border-border pb-6">
                <h2 className="text-2xl font-serif mb-2">Portfolio Allocation Strategy</h2>
                <p className="text-sm text-muted">Configure your profile to generate a risk-aware investment strategy.</p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-10">
                <div className="flex flex-col gap-5">
                  <div>
                    <label className="text-[11px] font-mono text-dim uppercase mb-2 block">Personal Info</label>
                    <div className="grid grid-cols-2 gap-3">
                      <div className="p-3 rounded-xl border border-border bg-bg2">
                        <div className="text-[10px] text-muted mb-1">Age</div>
                        <input 
                          type="number" 
                          value={profileForm.age}
                          onChange={(e) => setProfileForm({...profileForm, age: parseInt(e.target.value)})}
                          className="w-full bg-transparent border-none outline-none text-sm font-mono" 
                        />
                      </div>
                      <div className="p-3 rounded-xl border border-border bg-bg2">
                        <div className="text-[10px] text-muted mb-1">Dependents</div>
                        <input 
                          type="number" 
                          value={profileForm.dependents}
                          onChange={(e) => setProfileForm({...profileForm, dependents: parseInt(e.target.value)})}
                          className="w-full bg-transparent border-none outline-none text-sm font-mono" 
                        />
                      </div>
                    </div>
                  </div>

                  <div>
                    <label className="text-[11px] font-mono text-dim uppercase mb-2 block">Monthly Cashflow</label>
                    <div className="grid grid-cols-2 gap-3">
                      <div className="p-3 rounded-xl border border-border bg-bg2">
                        <div className="text-[10px] text-muted mb-1">Primary Income (₹)</div>
                        <input 
                          type="number" 
                          value={profileForm.monthly_income}
                          onChange={(e) => setProfileForm({...profileForm, monthly_income: parseFloat(e.target.value)})}
                          className="w-full bg-transparent border-none outline-none text-sm font-mono" 
                        />
                      </div>
                      <div className="p-3 rounded-xl border border-border bg-bg2">
                        <div className="text-[10px] text-muted mb-1">Expenses (₹)</div>
                        <input 
                          type="number" 
                          value={profileForm.monthly_expenses}
                          onChange={(e) => setProfileForm({...profileForm, monthly_expenses: parseFloat(e.target.value)})}
                          className="w-full bg-transparent border-none outline-none text-sm font-mono" 
                        />
                      </div>
                      <div className="p-3 rounded-xl border border-border bg-bg2">
                        <div className="text-[10px] text-muted mb-1">Pension (₹)</div>
                        <input 
                          type="number" 
                          value={profileForm.pension}
                          onChange={(e) => setProfileForm({...profileForm, pension: parseFloat(e.target.value)})}
                          className="w-full bg-transparent border-none outline-none text-sm font-mono" 
                        />
                      </div>
                      <div className="p-3 rounded-xl border border-border bg-bg2">
                        <div className="text-[10px] text-muted mb-1">Other Income (₹)</div>
                        <input 
                          type="number" 
                          value={profileForm.additional_income}
                          onChange={(e) => setProfileForm({...profileForm, additional_income: parseFloat(e.target.value)})}
                          className="w-full bg-transparent border-none outline-none text-sm font-mono" 
                        />
                      </div>
                    </div>
                  </div>

                  <div>
                    <label className="text-[11px] font-mono text-dim uppercase mb-2 block">Current Assets & Goal</label>
                    <div className="grid grid-cols-2 gap-3">
                      <div className="p-3 rounded-xl border border-border bg-bg2">
                        <div className="text-[10px] text-muted mb-1">Existing Savings (₹)</div>
                        <input 
                          type="number" 
                          value={profileForm.existing_savings}
                          onChange={(e) => setProfileForm({...profileForm, existing_savings: parseFloat(e.target.value)})}
                          className="w-full bg-transparent border-none outline-none text-sm font-mono" 
                        />
                      </div>
                      <div className="p-3 rounded-xl border border-border bg-bg2">
                        <div className="text-[10px] text-muted mb-1">Target Investment (₹)</div>
                        <input 
                          type="number" 
                          value={profileForm.amount_to_invest}
                          onChange={(e) => setProfileForm({...profileForm, amount_to_invest: parseFloat(e.target.value)})}
                          className="w-full bg-transparent border-none outline-none text-sm font-mono" 
                        />
                      </div>
                    </div>
                  </div>
                </div>

                <div className="flex flex-col gap-5">
                   <div>
                    <label className="text-[11px] font-mono text-dim uppercase mb-2 block">Safety Nets</label>
                    <div className="grid grid-cols-1 gap-2">
                      <button 
                        onClick={() => setProfileForm({...profileForm, health_insurance: !profileForm.health_insurance})}
                        className={`flex items-center justify-between p-3 rounded-xl border transition-all ${profileForm.health_insurance ? 'border-finance/50 bg-finance/5 text-finance' : 'border-border bg-bg2 text-muted'}`}
                      >
                        <div className="flex items-center gap-2 text-xs">
                          <ShieldCheck size={14} /> Health Insurance
                        </div>
                        {profileForm.health_insurance ? 'YES' : 'NO'}
                      </button>
                      <button 
                         onClick={() => setProfileForm({...profileForm, life_insurance: !profileForm.life_insurance})}
                        className={`flex items-center justify-between p-3 rounded-xl border transition-all ${profileForm.life_insurance ? 'border-finance/50 bg-finance/5 text-finance' : 'border-border bg-bg2 text-muted'}`}
                      >
                        <div className="flex items-center gap-2 text-xs">
                          <Heart size={14} /> Life Insurance
                        </div>
                        {profileForm.life_insurance ? 'YES' : 'NO'}
                      </button>
                    </div>
                  </div>

                  <div>
                    <label className="text-[11px] font-mono text-dim uppercase mb-2 block">Strategy Controls</label>
                    <div className="flex flex-col gap-3">
                      <select 
                        value={profileForm.investment_horizon}
                        onChange={(e) => setProfileForm({...profileForm, investment_horizon: e.target.value})}
                        className="p-3 rounded-xl border border-border bg-bg2 text-sm outline-none"
                      >
                        <option value="1yr">Short Term (1 Yr)</option>
                        <option value="3yr">Medium Term (3 Yrs)</option>
                        <option value="5yr">Long Term (5+ Yrs)</option>
                      </select>
                      <select 
                        value={profileForm.primary_goal}
                        onChange={(e) => setProfileForm({...profileForm, primary_goal: e.target.value})}
                        className="p-3 rounded-xl border border-border bg-bg2 text-sm outline-none"
                      >
                        <option value="Wealth Creation">Wealth Creation</option>
                        <option value="Capital Preservation">Capital Preservation</option>
                        <option value="Retirement">Retirement Planning</option>
                      </select>
                    </div>
                  </div>

                  <button 
                    onClick={handlePortfolioRun}
                    disabled={financeLoading}
                    className="mt-auto py-4 rounded-xl bg-text text-bg font-medium flex items-center justify-center gap-2 hover:scale-[1.02] active:scale-[0.98] transition-all disabled:opacity-50"
                  >
                    {financeLoading ? <RefreshCw size={18} className="animate-spin" /> : <Calculator size={18} />}
                    Generate Strategy
                  </button>
                </div>
              </div>

              {portfolioData && (
                <motion.div 
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="p-8 rounded-3xl border border-border bg-bg3"
                >
                  <div className="flex items-center justify-between mb-8">
                    <div>
                      <div className="text-[10px] font-mono text-finance uppercase tracking-widest mb-1">
                        {portfolioData.is_fallback ? 'Analysis (Fallback Mode)' : 'Analysis Complete'}
                      </div>
                      <h3 className="text-xl font-medium">Financial Health: <span className={portfolioData.status === 'CRITICAL' ? 'text-health' : 'text-finance'}>{portfolioData.status || 'ANALYZING'}</span></h3>
                      {portfolioData.is_fallback && (
                        <p className="text-[10px] text-health mt-1 font-mono uppercase">! Knowledge Graph Offline - Using Market Baseline</p>
                      )}
                    </div>
                    <div className="w-12 h-12 rounded-2xl bg-finance/10 flex items-center justify-center text-finance border border-finance/20">
                      <PieChart size={24} />
                    </div>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-8">
                    <div className="p-6 rounded-2xl border border-border bg-bg/50">
                      <h4 className="text-[11px] font-mono text-dim uppercase mb-4">Recommended Allocations</h4>
                      <div className="flex flex-col gap-4">
                        {portfolioData.allocations && Object.entries(portfolioData.allocations).map(([sector, percent], i) => (
                          <div key={i}>
                            <div className="flex justify-between text-xs mb-1.5">
                              <span>{sector}</span>
                              <span className="font-mono">{percent}%</span>
                            </div>
                            <div className="h-1 bg-white/5 rounded-full overflow-hidden">
                              <div className="h-full bg-finance" style={{ width: `${percent}%` }} />
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                    <div className="p-6 rounded-2xl border border-border bg-bg/50">
                      <h4 className="text-[11px] font-mono text-dim uppercase mb-4">Risk Profile</h4>
                      <div className="text-3xl font-serif mb-2">{portfolioData.risk_profile || 'Moderate'}</div>
                      <p className="text-[12px] text-muted leading-relaxed">
                        Based on your age ({profileForm.age}) and surplus income of ₹{portfolioData.surplus_income || 0}, 
                        the engine has calibrated a {(portfolioData.risk_profile || 'Moderate').toLowerCase()} strategy for your {profileForm.investment_horizon} horizon.
                      </p>
                    </div>
                  </div>

                  <div className="p-6 rounded-2xl border border-finance/20 bg-finance/5">
                    <div className="flex items-center gap-2 mb-3 text-finance">
                      <Sparkles size={16} />
                      <span className="text-[11px] font-mono uppercase tracking-wider">AI Strategic Insights</span>
                    </div>
                    <p className="text-sm text-text leading-relaxed whitespace-pre-wrap italic">
                      "{portfolioData.explanation || 'No explanation available.'}"
                    </p>
                  </div>
                </motion.div>
              )}
            </div>
          )}

          {viewMode === 'tradetest' && (
            <div className="max-w-[800px] mx-auto w-full pb-20">
               <div className="mb-8 border-b border-border pb-6">
                <h2 className="text-2xl font-serif mb-2">Trade Simulator & Backtester</h2>
                <p className="text-sm text-muted">Validate trading strategies against historical NSE market data.</p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-10">
                <div className="p-4 rounded-xl border border-border bg-bg2">
                  <label className="text-[10px] font-mono text-dim uppercase mb-2 block">Ticker Symbol</label>
                  <input 
                    type="text" 
                    value={tradeForm.symbol}
                    onChange={(e) => setTradeForm({...tradeForm, symbol: e.target.value.toUpperCase()})}
                    className="w-full bg-transparent border-none outline-none text-lg font-mono text-finance" 
                  />
                  <div className="text-[10px] text-dim mt-1">e.g. RELIANCE.NS, TCS.NS</div>
                </div>
                <div className="p-4 rounded-xl border border-border bg-bg2">
                  <label className="text-[10px] font-mono text-dim uppercase mb-2 block">Strategy</label>
                  <select 
                    value={tradeForm.strategy}
                    onChange={(e) => setTradeForm({...tradeForm, strategy: e.target.value})}
                    className="w-full bg-transparent border-none outline-none text-sm font-sans mt-1"
                  >
                    <option value="SMA_Crossover">SMA Crossover (50/200)</option>
                    <option value="RSI_Standard">RSI Mean Reversion</option>
                  </select>
                </div>
                <div className="p-4 rounded-xl border border-border bg-bg2">
                  <label className="text-[10px] font-mono text-dim uppercase mb-2 block">Time Period</label>
                  <select 
                    value={tradeForm.period}
                    onChange={(e) => setTradeForm({...tradeForm, period: e.target.value})}
                    className="w-full bg-transparent border-none outline-none text-sm font-sans mt-1"
                  >
                    <option value="1mo">1 Month</option>
                    <option value="6mo">6 Months</option>
                    <option value="1y">1 Year</option>
                    <option value="2y">2 Years</option>
                  </select>
                </div>
              </div>

              <button 
                onClick={handleTradeTestRun}
                disabled={financeLoading}
                className="w-full py-4 rounded-xl bg-finance text-bg font-medium flex items-center justify-center gap-2 hover:brightness-110 active:scale-[0.99] transition-all disabled:opacity-50 mb-10"
              >
                {financeLoading ? <RefreshCw size={18} className="animate-spin" /> : <TrendingUp size={18} />}
                Run Simulation
              </button>

              {tradeTestData && (
                <motion.div 
                  initial={{ opacity: 0, scale: 0.98 }}
                  animate={{ opacity: 1, scale: 1 }}
                  className="grid grid-cols-4 gap-4"
                >
                  <div className="col-span-4 p-6 rounded-3xl border border-border bg-bg3 flex items-center justify-between mb-2">
                    <div className="flex items-center gap-4">
                      <div className="w-12 h-12 rounded-2xl bg-finance/10 flex items-center justify-center text-finance">
                        <Activity size={24} />
                      </div>
                      <div>
                        <div className="text-[10px] font-mono text-dim uppercase tracking-wider">Strategy Score</div>
                        <div className="text-2xl font-serif">{tradeTestData.evaluation?.score || 0}/100 — {tradeTestData.evaluation?.grade || 'N/A'}</div>
                      </div>
                    </div>
                    <div className={`px-4 py-2 rounded-full border ${tradeTestData.results?.total_return >= 0 ? 'border-finance/30 bg-finance/5 text-finance' : 'border-health/30 bg-health/5 text-health'} font-mono text-sm`}>
                      {tradeTestData.results?.total_return > 0 ? '+' : ''}{tradeTestData.results?.total_return?.toFixed(2)}% Return
                    </div>
                  </div>

                  {[
                    { label: 'Win Rate', value: `${((tradeTestData.results?.win_rate || 0) * 100).toFixed(1)}%`, icon: <CheckCircle2 size={14}/> },
                    { label: 'Sharpe Ratio', value: (tradeTestData.results?.sharpe_ratio || 0).toFixed(2), icon: <Activity size={14}/> },
                    { label: 'Max Drawdown', value: `${((tradeTestData.results?.max_drawdown || 0) * 100).toFixed(1)}%`, icon: <ShieldCheck size={14}/> },
                    { label: 'Total Trades', value: tradeTestData.results?.total_trades || 0, icon: <Briefcase size={14}/> }
                  ].map((stat, i) => (
                    <div key={i} className="p-4 rounded-2xl border border-border bg-bg2">
                      <div className="flex items-center gap-2 text-dim mb-2">
                        {stat.icon}
                        <span className="text-[10px] font-mono uppercase">{stat.label}</span>
                      </div>
                      <div className="text-lg font-mono">{stat.value}</div>
                    </div>
                  ))}

                  <div className="col-span-4 p-6 rounded-3xl border border-border bg-bg3 mt-2">
                    <h4 className="text-[11px] font-mono text-dim uppercase mb-4 flex items-center gap-2">
                      <Sparkles size={14} className="text-finance" />
                      ML Evaluator Insight
                    </h4>
                    <p className="text-sm text-text leading-relaxed">
                      {tradeTestData.evaluation?.recommendation || 'No recommendation available.'}
                    </p>
                  </div>
                </motion.div>
              )}
            </div>
          )}
        </div>

        {viewMode === 'chat' && (
          <footer className="p-8 bg-bg border-t border-border">
            <div className="max-w-[800px] mx-auto">
              <div className="flex items-center gap-2 mb-4">
                 {['Explain capital ratios', 'How HIPAA affects ePHI?', 'Recent case precedents'].map((q, i) => (
                   <button key={i} onClick={() => setInput(q)} className="text-[11px] text-muted px-3.5 py-1.5 border border-border rounded-full bg-bg2 hover:bg-bg3 hover:text-text transition-all">
                     {q}
                   </button>
                 ))}
              </div>
              <div className="flex items-end gap-3 bg-bg2 border border-border2 rounded-2xl p-4 focus-within:border-white/25 transition-all">
                <textarea 
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && (e.preventDefault(), handleSend())}
                  placeholder="Ask your specialized query..."
                  className="flex-1 bg-transparent border-none outline-none font-sans text-[14px] text-text resize-none leading-relaxed min-h-[24px] max-h-32"
                />
                <button onClick={handleSend} disabled={!input.trim() || loading} className="w-10 h-10 rounded-xl bg-text text-bg flex items-center justify-center transition-all hover:scale-105 active:scale-95 disabled:opacity-50 disabled:scale-100">
                  <Send size={18} />
                </button>
              </div>
            </div>
          </footer>
        )}
      </main>

      {/* RIGHT PANEL */}
      <aside className="w-[340px] flex-shrink-0 border-l border-border bg-bg p-8 flex flex-col gap-6 overflow-y-auto">
        <div>
          <h3 className="text-[10px] font-mono tracking-[0.1em] uppercase text-dim mb-4">Quality Metrics</h3>
          <div className="flex flex-col gap-4">
            {[
              { label: 'Faithfulness', value: 92, color: 'text-health' },
              { label: 'Context Precision', value: 88, color: 'text-finance' },
              { label: 'Context Recall', value: 74, color: 'text-legal' }
            ].map((m, i) => (
              <div key={i}>
                <div className="flex justify-between items-center mb-2">
                  <span className="text-[12px] text-muted">{m.label}</span>
                  <span className={`text-[12px] font-mono ${m.color}`}>{m.value}%</span>
                </div>
                <div className="h-1 bg-white/5 rounded-full overflow-hidden">
                  <motion.div 
                    initial={{ width: 0 }}
                    animate={{ width: `${m.value}%` }}
                    className={`h-full ${m.color.replace('text-', 'bg-')}`}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="flex-1 mt-4">
          <h3 className="text-[10px] font-mono tracking-[0.1em] uppercase text-dim mb-4">Relevance Map</h3>
          <div className="h-48 flex items-end gap-2 border-b border-border pb-2 mb-4">
            {[40, 75, 92, 60, 45].map((h, i) => (
              <motion.div 
                key={i} 
                initial={{ height: 0 }}
                animate={{ height: `${h}%` }}
                className="flex-1 bg-white/10 rounded-t-sm hover:bg-finance/30 transition-colors cursor-help"
                title={`Source ${i+1}: ${h}% relevance`}
              />
            ))}
          </div>
          <div className="flex justify-between text-[9px] font-mono text-dim">
            <span>Naive Hits</span>
            <span>Graph Facts</span>
          </div>
        </div>

        <div className="p-4 rounded-2xl border border-border bg-bg2 flex flex-col gap-3">
          <div className="flex items-center gap-2 text-[12px] text-text">
            <Sparkles size={14} className="text-finance" />
            <span>Consultation Insights</span>
          </div>
          <p className="text-[11px] text-muted leading-relaxed">
            The hybrid retriever found a critical relationship between <strong>Entity A</strong> and <strong>Entity B</strong> in the Neo4j subgraph that was absent from vector search hits.
          </p>
        </div>
      </aside>
    </div>
  )
}

export default Consultation

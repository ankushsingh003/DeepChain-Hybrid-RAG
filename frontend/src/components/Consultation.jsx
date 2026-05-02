import React, { useState, useRef, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Send, ArrowLeft, Database, Sparkles, Network,
  BarChart2, FileText, Activity, RefreshCw, CheckCircle2,
  Circle, PieChart, TrendingUp, ShieldCheck, Wallet,
  Briefcase, Calculator, Heart, ChevronRight, ChevronDown,
  Zap, Target, AlertTriangle, Lock, Eye, Code2,
  AreaChart, CandlestickChart, Cpu
} from 'lucide-react'
import { queryRAG, triggerIngestion, getPortfolioStrategy, runTradeTest, getStrategyAdvice, getMarketStrategyAdvice } from '../services/api'

// ─── tiny helpers ─────────────────────────────────────────────────────────────
const Dot = ({ active }) => (
  <span className={`inline-block w-1.5 h-1.5 rounded-full ${active ? 'bg-emerald-400 animate-pulse' : 'bg-white/10'}`} />
)

const Tag = ({ children, color = 'finance' }) => (
  <span className={`inline-flex items-center gap-1 text-[9px] font-mono tracking-widest uppercase px-2 py-0.5 rounded border border-${color}/30 text-${color} bg-${color}/5`}>
    {children}
  </span>
)

const MetricBar = ({ label, value, color }) => (
  <div>
    <div className="flex justify-between items-center mb-1.5">
      <span className="text-[11px] text-white/40 font-mono">{label}</span>
      <span className={`text-[11px] font-mono font-semibold ${color}`}>{value}%</span>
    </div>
    <div className="h-[3px] bg-white/5 rounded-full overflow-hidden">
      <motion.div
        initial={{ width: 0 }}
        animate={{ width: `${value}%` }}
        transition={{ duration: 1.2, ease: 'easeOut', delay: 0.2 }}
        className={`h-full rounded-full ${color.replace('text-', 'bg-')}`}
      />
    </div>
  </div>
)

const PipelineStep = ({ label, id, current }) => {
  const done = current > id
  const active = current === id
  return (
    <div className={`flex items-center gap-2.5 transition-all duration-300 ${active ? 'opacity-100' : done ? 'opacity-60' : 'opacity-25'}`}>
      <div className={`w-4 h-4 rounded flex items-center justify-center flex-shrink-0 border ${active ? 'border-emerald-400/60 bg-emerald-400/10' : done ? 'border-white/20 bg-white/5' : 'border-white/10'}`}>
        {active ? <RefreshCw size={8} className="text-emerald-400 animate-spin" />
          : done ? <CheckCircle2 size={8} className="text-white/40" />
          : <Circle size={8} className="text-white/20" />}
      </div>
      <span className={`text-[10px] font-mono ${active ? 'text-emerald-400' : 'text-white/40'}`}>{label}</span>
      {active && <span className="ml-auto text-[9px] font-mono text-emerald-400/60 animate-pulse">RUNNING</span>}
    </div>
  )
}

// ─── Main Component ───────────────────────────────────────────────────────────
const Consultation = ({ domain, onBack }) => {
  if (!domain) return null

  const [messages, setMessages] = useState([{
    role: 'assistant',
    content: `Hello. I am your DeepChain consultant for ${domain.name}. How can I assist with your domain research today?`,
    timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
  }])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [ingesting, setIngesting] = useState(false)
  const [method, setMethod] = useState('hybrid')
  const [pipelineStep, setPipelineStep] = useState(0)
  const [viewMode, setViewMode] = useState(() => localStorage.getItem('deepchain_viewmode') || 'chat')
  const [portfolioData, setPortfolioData] = useState(null)
  const [tradeTestData, setTradeTestData] = useState(null)
  const [strategyAdvice, setStrategyAdvice] = useState(null)
  const [marketSymbol, setMarketSymbol] = useState('TSLA')
  const [financeLoading, setFinanceLoading] = useState(false)
  const [rightTab, setRightTab] = useState('metrics')
  const [profileForm, setProfileForm] = useState({
    age: 30, monthly_income: 50000, monthly_expenses: 20000,
    pension: 0, govt_allowances: 0, additional_income: 0,
    dependents: 0, existing_savings: 0, emergency_fund_exists: false,
    amount_to_invest: 10000, liabilities: [], life_insurance: false,
    health_insurance: false, investment_horizon: '5yr', primary_goal: 'Wealth Creation'
  })
  const [tradeForm, setTradeForm] = useState({ symbol: 'RELIANCE.NS', strategy: 'SMA_Crossover', period: '1y' })
  const scrollRef = useRef(null)

  useEffect(() => { localStorage.setItem('deepchain_viewmode', viewMode) }, [viewMode])
  useEffect(() => {
    if (domain.id !== 'finance' && ['strategy_advisor', 'portfolio', 'tradetest'].includes(viewMode)) setViewMode('chat')
  }, [domain.id])
  useEffect(() => {
    if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight
  }, [messages])

  // ── Handlers ─────────────────────────────────────────────────────────────
  const handleSend = async () => {
    if (!input.trim() || loading) return
    const userMsg = { role: 'user', content: input, timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) }
    setMessages(prev => [...prev, userMsg])
    setInput('')
    setLoading(true)
    setPipelineStep(1)
    setTimeout(() => setPipelineStep(2), 800)
    setTimeout(() => setPipelineStep(3), 1600)
    setTimeout(() => setPipelineStep(4), 2400)
    try {
      const data = await queryRAG(input, method)
      setMessages(prev => [...prev, {
        role: 'assistant', content: data.answer, method: data.method,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        citations: ['SEC-2024-Q3', 'Risk-Protocol-v2'],
        graph: [{ s: 'Entity A', p: 'OWNS', o: 'Entity B' }, { s: 'Entity B', p: 'LOCATED_IN', o: 'Singapore' }]
      }])
      setPipelineStep(0)
    } catch {
      setMessages(prev => [...prev, { role: 'assistant', content: "I'm sorry, I encountered an error connecting to the intelligence engine.", timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) }])
    } finally { setLoading(false) }
  }

  const handleIngestion = async () => {
    setIngesting(true)
    try { await triggerIngestion(); alert('Ingestion completed successfully!') }
    catch { alert('Ingestion failed. Check server logs.') }
    finally { setIngesting(false) }
  }

  const handlePortfolioRun = async () => {
    setFinanceLoading(true)
    try { setPortfolioData(await getPortfolioStrategy(profileForm)) }
    catch { alert('Portfolio generation failed.') }
    finally { setFinanceLoading(false) }
  }

  const handleTradeTestRun = async () => {
    setFinanceLoading(true)
    try { setTradeTestData(await runTradeTest(tradeForm.symbol, tradeForm.strategy, tradeForm.period)) }
    catch { alert('Trade test failed.') }
    finally { setFinanceLoading(false) }
  }

  const handleStrategyAdvisorRun = async () => {
    if (!input.trim()) return
    setFinanceLoading(true)
    try { setStrategyAdvice(await getStrategyAdvice(input)) }
    catch { alert('Strategy generation failed.') }
    finally { setFinanceLoading(false) }
  }

  const handleMarketAnalysisRun = async () => {
    if (!marketSymbol.trim()) return
    setFinanceLoading(true)
    try {
      const data = await getMarketStrategyAdvice(marketSymbol)
      setStrategyAdvice({ approach_report: data.dynamic_report, retrieved_context: data.retrieved_context })
    } catch { alert('Market analysis failed.') }
    finally { setFinanceLoading(false) }
  }

  // ── Sidebar nav items ─────────────────────────────────────────────────────
  const financeTools = [
    { id: 'chat',            label: 'Expert Consultation', icon: <Sparkles size={13} /> },
    { id: 'portfolio',       label: 'Portfolio Allocation', icon: <PieChart size={13} /> },
    { id: 'tradetest',       label: 'Trade Simulator',      icon: <CandlestickChart size={13} /> },
    { id: 'strategy_advisor',label: 'Strategy Advisor',     icon: <Target size={13} /> },
  ]

  // ─────────────────────────────────────────────────────────────────────────
  return (
    <div className="flex h-[calc(100vh-65px)] overflow-hidden" style={{ background: '#080810' }}>

      {/* ══════════════════ LEFT SIDEBAR ══════════════════ */}
      <aside className="w-[260px] flex-shrink-0 flex flex-col overflow-y-auto border-r"
        style={{ borderColor: 'rgba(255,255,255,0.06)', background: '#0b0b16' }}>

        {/* Back */}
        <div className="px-5 pt-5 pb-4 border-b" style={{ borderColor: 'rgba(255,255,255,0.05)' }}>
          <button onClick={onBack}
            className="flex items-center gap-2 text-[10px] font-mono text-white/30 hover:text-white/60 transition-colors uppercase tracking-[0.12em]">
            <ArrowLeft size={12} /> Back to Hub
          </button>
        </div>

        {/* Domain badge */}
        <div className="px-5 py-4 border-b" style={{ borderColor: 'rgba(255,255,255,0.05)' }}>
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl flex items-center justify-center text-base"
              style={{ background: 'rgba(232,209,122,0.08)', border: '1px solid rgba(232,209,122,0.15)' }}>
              {domain.icon}
            </div>
            <div>
              <div className="text-[13px] font-medium text-white/90 leading-tight">{domain.name}</div>
              <div className="text-[10px] font-mono text-white/30 mt-0.5">{domain.tag}</div>
            </div>
          </div>
        </div>

        {/* Knowledge Sources */}
        <div className="px-5 py-4 border-b" style={{ borderColor: 'rgba(255,255,255,0.05)' }}>
          <div className="text-[9px] font-mono tracking-[0.14em] uppercase text-white/25 mb-3">Knowledge Sources</div>
          <div className="flex flex-col gap-1.5">
            {[
              { name: 'SEC_FILINGS.PDF', count: '1,240' },
              { name: 'MARKET_DATA.CSV', count: '842' },
              { name: 'CORP_STATUTES.JSON', count: '5,100' }
            ].map((s, i) => (
              <div key={i} className="flex items-center gap-2 px-3 py-2 rounded-lg transition-colors cursor-default"
                style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.04)' }}>
                <FileText size={11} className="text-white/20 flex-shrink-0" />
                <span className="text-[10px] font-mono text-white/40 flex-1 truncate">{s.name}</span>
                <span className="text-[9px] font-mono text-white/20">{s.count}</span>
              </div>
            ))}
          </div>
        </div>

        {/* System Controls */}
        <div className="px-5 py-4 border-b" style={{ borderColor: 'rgba(255,255,255,0.05)' }}>
          <div className="text-[9px] font-mono tracking-[0.14em] uppercase text-white/25 mb-3">System Controls</div>
          <button onClick={handleIngestion} disabled={ingesting}
            className="w-full flex items-center justify-center gap-2 py-2.5 rounded-lg text-[10px] font-mono transition-all disabled:opacity-40"
            style={{ background: ingesting ? 'rgba(232,209,122,0.06)' : 'rgba(232,209,122,0.07)', border: '1px solid rgba(232,209,122,0.15)', color: '#E8D17A' }}>
            {ingesting ? <><RefreshCw size={10} className="animate-spin" /> Ingesting...</> : <><Database size={10} /> Trigger Ingestion</>}
          </button>
        </div>

        {/* Live Pipeline */}
        <div className="px-5 py-4 border-b" style={{ borderColor: 'rgba(255,255,255,0.05)' }}>
          <div className="text-[9px] font-mono tracking-[0.14em] uppercase text-white/25 mb-3">Live Pipeline</div>
          <div className="flex flex-col gap-2.5">
            {[
              { label: 'Query Decomposition', id: 1 },
              { label: 'Graph Retrieval',     id: 2 },
              { label: 'Vector Fusion',       id: 3 },
              { label: 'Generation',          id: 4 }
            ].map(s => <PipelineStep key={s.id} {...s} current={pipelineStep} />)}
          </div>
        </div>

        {/* Financial Toolkit */}
        {domain.id === 'finance' && (
          <div className="px-5 py-4 flex-1">
            <div className="text-[9px] font-mono tracking-[0.14em] uppercase text-white/25 mb-3">Financial Toolkit</div>
            <div className="flex flex-col gap-1">
              {financeTools.map(tool => (
                <button key={tool.id} onClick={() => setViewMode(tool.id)}
                  className="w-full flex items-center gap-2.5 py-2.5 px-3 rounded-lg text-[11px] font-mono transition-all text-left"
                  style={viewMode === tool.id
                    ? { background: 'rgba(232,209,122,0.08)', border: '1px solid rgba(232,209,122,0.18)', color: '#E8D17A' }
                    : { background: 'transparent', border: '1px solid transparent', color: 'rgba(255,255,255,0.35)' }}>
                  <span className={viewMode === tool.id ? 'text-[#E8D17A]' : 'text-white/25'}>{tool.icon}</span>
                  {tool.label}
                  {viewMode === tool.id && <ChevronRight size={10} className="ml-auto" />}
                </button>
              ))}
            </div>
          </div>
        )}
      </aside>

      {/* ══════════════════ CENTER MAIN ══════════════════ */}
      <main className="flex-1 flex flex-col min-w-0 overflow-hidden">

        {/* Top bar */}
        <header className="px-7 py-4 flex items-center justify-between flex-shrink-0"
          style={{ borderBottom: '1px solid rgba(255,255,255,0.06)', background: 'rgba(8,8,16,0.95)', backdropFilter: 'blur(12px)' }}>
          <div className="flex items-center gap-4">
            <div>
              <div className="flex items-center gap-2">
                <span className="text-[13px] font-medium text-white/90">Active Consultation</span>
                {viewMode !== 'chat' && (
                  <span className="text-[10px] font-mono text-white/30 flex items-center gap-1">
                    <ChevronRight size={10} />
                    {financeTools.find(t => t.id === viewMode)?.label}
                  </span>
                )}
              </div>
              <div className="flex items-center gap-1.5 mt-0.5">
                <Dot active />
                <span className="text-[10px] font-mono text-white/30">Engine Online — Gemini 2.0 Flash</span>
              </div>
            </div>
          </div>

          <div className="flex items-center gap-2">
            {viewMode === 'chat' && (
              <>
                {['naive', 'hybrid'].map(m => (
                  <button key={m} onClick={() => setMethod(m)}
                    className="text-[10px] font-mono px-3 py-1.5 rounded-md transition-all"
                    style={method === m
                      ? { background: 'rgba(232,209,122,0.08)', border: '1px solid rgba(232,209,122,0.25)', color: '#E8D17A' }
                      : { background: 'transparent', border: '1px solid rgba(255,255,255,0.08)', color: 'rgba(255,255,255,0.35)' }}>
                    {m === 'naive' ? 'Naive RAG' : 'Hybrid RAG'}
                  </button>
                ))}
              </>
            )}
            {viewMode !== 'chat' && (
              <button onClick={() => setViewMode('chat')}
                className="flex items-center gap-1.5 text-[10px] font-mono px-3 py-1.5 rounded-md transition-all"
                style={{ border: '1px solid rgba(255,255,255,0.08)', color: 'rgba(255,255,255,0.35)' }}>
                <ArrowLeft size={10} /> Back to Chat
              </button>
            )}
          </div>
        </header>

        {/* Scrollable content */}
        <div ref={scrollRef} className="flex-1 overflow-y-auto p-7 flex flex-col gap-5 custom-scrollbar">

          {/* ── CHAT VIEW ─────────────────────────────── */}
          {viewMode === 'chat' && (
            <>
              {messages.map((msg, i) => (
                <motion.div key={i} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.25 }}
                  className={`flex flex-col gap-2 max-w-[620px] ${msg.role === 'user' ? 'self-end items-end' : 'self-start items-start'}`}>
                  <div className="flex items-center gap-2">
                    <div className={`w-5 h-5 rounded flex items-center justify-center text-[9px] font-mono flex-shrink-0 ${msg.role === 'assistant' ? 'text-white/30' : 'text-white/50'}`}
                      style={{ background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.08)' }}>
                      {msg.role === 'assistant' ? 'AI' : 'U'}
                    </div>
                    <span className="text-[10px] font-mono text-white/25">
                      {msg.role === 'assistant' ? 'DeepChain' : 'You'} · {msg.timestamp}
                    </span>
                    {msg.method && <Tag>{msg.method}</Tag>}
                  </div>

                  <div className={msg.role === 'user'
                    ? 'px-4 py-3 rounded-2xl rounded-tr-sm text-[13px] leading-[1.7] text-white/80'
                    : 'text-[13px] leading-[1.8] text-white/75'}
                    style={msg.role === 'user' ? { background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.07)' } : {}}>
                    <p className="whitespace-pre-wrap">{msg.content}</p>

                    {msg.role === 'assistant' && msg.citations && (
                      <div className="flex flex-wrap gap-1.5 mt-3 pt-3" style={{ borderTop: '1px solid rgba(255,255,255,0.05)' }}>
                        {msg.citations.map((c, j) => (
                          <span key={j} className="inline-flex items-center gap-1 text-[9px] font-mono text-white/30 px-2 py-1 rounded"
                            style={{ border: '1px solid rgba(255,255,255,0.07)', background: 'rgba(255,255,255,0.02)' }}>
                            <FileText size={8} /> {c}
                          </span>
                        ))}
                      </div>
                    )}

                    {msg.role === 'assistant' && msg.graph && (
                      <div className="mt-3 rounded-xl overflow-hidden"
                        style={{ border: '1px solid rgba(255,255,255,0.06)', background: 'rgba(255,255,255,0.02)' }}>
                        <div className="px-3 py-2 flex items-center gap-2 text-[9px] font-mono text-white/25"
                          style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                          <Network size={9} /> Graph Extraction
                        </div>
                        <div className="p-3 flex flex-col gap-1.5">
                          {msg.graph.map((t, j) => (
                            <div key={j} className="flex items-center gap-2 text-[10px] font-mono">
                              <span className="px-2 py-0.5 rounded text-[#E8D17A]"
                                style={{ border: '1px solid rgba(232,209,122,0.2)', background: 'rgba(232,209,122,0.05)' }}>{t.s}</span>
                              <span className="text-white/20">→ {t.p} →</span>
                              <span className="px-2 py-0.5 rounded text-[#A78BFA]"
                                style={{ border: '1px solid rgba(167,139,250,0.2)', background: 'rgba(167,139,250,0.05)' }}>{t.o}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </motion.div>
              ))}

              {loading && (
                <div className="self-start flex items-center gap-2.5 max-w-[620px]">
                  <div className="w-5 h-5 rounded flex items-center justify-center"
                    style={{ background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.08)' }}>
                    <Activity size={9} className="text-white/30 animate-pulse" />
                  </div>
                  <div className="flex gap-1">
                    {[0,1,2].map(i => (
                      <motion.div key={i} className="w-1.5 h-1.5 rounded-full bg-white/20"
                        animate={{ opacity: [0.2, 0.8, 0.2] }}
                        transition={{ duration: 1.2, repeat: Infinity, delay: i * 0.2 }} />
                    ))}
                  </div>
                </div>
              )}
            </>
          )}

          {/* ── PORTFOLIO VIEW ──────────────────────────── */}
          {viewMode === 'portfolio' && domain.id === 'finance' && (
            <div className="max-w-[820px] mx-auto w-full pb-10">
              <div className="mb-7">
                <div className="text-[9px] font-mono tracking-[0.14em] uppercase text-white/25 mb-1.5 flex items-center gap-2">
                  <PieChart size={9} /> Portfolio Engine
                </div>
                <h2 className="text-[22px] font-serif text-white/90 leading-tight">Portfolio Allocation Strategy</h2>
                <p className="text-[12px] text-white/35 mt-1">Configure your financial profile for a risk-calibrated investment strategy.</p>
              </div>

              <div className="grid grid-cols-2 gap-5 mb-6">
                {/* Left col */}
                <div className="flex flex-col gap-4">
                  <SectionCard title="Personal Info">
                    <div className="grid grid-cols-2 gap-2.5">
                      <InputField label="Age" type="number" value={profileForm.age}
                        onChange={v => setProfileForm({ ...profileForm, age: +v })} />
                      <InputField label="Dependents" type="number" value={profileForm.dependents}
                        onChange={v => setProfileForm({ ...profileForm, dependents: +v })} />
                    </div>
                  </SectionCard>

                  <SectionCard title="Monthly Cashflow">
                    <div className="grid grid-cols-2 gap-2.5">
                      <InputField label="Income (₹)" type="number" value={profileForm.monthly_income}
                        onChange={v => setProfileForm({ ...profileForm, monthly_income: +v })} />
                      <InputField label="Expenses (₹)" type="number" value={profileForm.monthly_expenses}
                        onChange={v => setProfileForm({ ...profileForm, monthly_expenses: +v })} />
                      <InputField label="Pension (₹)" type="number" value={profileForm.pension}
                        onChange={v => setProfileForm({ ...profileForm, pension: +v })} />
                      <InputField label="Other Income (₹)" type="number" value={profileForm.additional_income}
                        onChange={v => setProfileForm({ ...profileForm, additional_income: +v })} />
                    </div>
                  </SectionCard>

                  <SectionCard title="Assets & Target">
                    <div className="grid grid-cols-2 gap-2.5">
                      <InputField label="Savings (₹)" type="number" value={profileForm.existing_savings}
                        onChange={v => setProfileForm({ ...profileForm, existing_savings: +v })} />
                      <InputField label="Invest Amount (₹)" type="number" value={profileForm.amount_to_invest}
                        onChange={v => setProfileForm({ ...profileForm, amount_to_invest: +v })} />
                    </div>
                  </SectionCard>
                </div>

                {/* Right col */}
                <div className="flex flex-col gap-4">
                  <SectionCard title="Safety Nets">
                    <div className="flex flex-col gap-2">
                      <ToggleBtn active={profileForm.health_insurance} icon={<ShieldCheck size={12} />}
                        label="Health Insurance" onToggle={() => setProfileForm({ ...profileForm, health_insurance: !profileForm.health_insurance })} />
                      <ToggleBtn active={profileForm.life_insurance} icon={<Heart size={12} />}
                        label="Life Insurance" onToggle={() => setProfileForm({ ...profileForm, life_insurance: !profileForm.life_insurance })} />
                    </div>
                  </SectionCard>

                  <SectionCard title="Strategy Controls">
                    <div className="flex flex-col gap-2.5">
                      <SelectField label="Investment Horizon" value={profileForm.investment_horizon}
                        onChange={v => setProfileForm({ ...profileForm, investment_horizon: v })}
                        options={[{ value: '1yr', label: 'Short Term (1 Yr)' }, { value: '3yr', label: 'Medium Term (3 Yrs)' }, { value: '5yr', label: 'Long Term (5+ Yrs)' }]} />
                      <SelectField label="Primary Goal" value={profileForm.primary_goal}
                        onChange={v => setProfileForm({ ...profileForm, primary_goal: v })}
                        options={[{ value: 'Wealth Creation', label: 'Wealth Creation' }, { value: 'Capital Preservation', label: 'Capital Preservation' }, { value: 'Retirement', label: 'Retirement Planning' }]} />
                    </div>
                  </SectionCard>

                  <button onClick={handlePortfolioRun} disabled={financeLoading}
                    className="mt-auto py-3.5 rounded-xl font-medium flex items-center justify-center gap-2 text-[13px] transition-all hover:brightness-110 active:scale-[0.99] disabled:opacity-40"
                    style={{ background: '#E8D17A', color: '#08080f' }}>
                    {financeLoading ? <RefreshCw size={15} className="animate-spin" /> : <Calculator size={15} />}
                    Generate Strategy
                  </button>
                </div>
              </div>

              <AnimatePresence>
                {portfolioData && (
                  <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
                    className="rounded-2xl overflow-hidden"
                    style={{ border: '1px solid rgba(255,255,255,0.08)', background: 'rgba(255,255,255,0.02)' }}>
                    <div className="px-6 py-4 flex items-center justify-between"
                      style={{ borderBottom: '1px solid rgba(255,255,255,0.06)', background: 'rgba(232,209,122,0.04)' }}>
                      <div>
                        <div className="text-[9px] font-mono tracking-[0.12em] uppercase text-[#E8D17A]/60 mb-1">Analysis Complete</div>
                        <div className="text-[16px] font-medium text-white/90">
                          Financial Health: <span style={{ color: portfolioData.status === 'CRITICAL' ? '#6ECFA0' : '#E8D17A' }}>{portfolioData.status || 'ANALYZING'}</span>
                        </div>
                      </div>
                      <div className="w-10 h-10 rounded-xl flex items-center justify-center"
                        style={{ background: 'rgba(232,209,122,0.08)', border: '1px solid rgba(232,209,122,0.15)' }}>
                        <PieChart size={18} style={{ color: '#E8D17A' }} />
                      </div>
                    </div>
                    <div className="p-6 grid grid-cols-2 gap-6">
                      <div>
                        <div className="text-[9px] font-mono text-white/25 uppercase tracking-widest mb-4">Allocations</div>
                        <div className="flex flex-col gap-3">
                          {portfolioData.allocations && Object.entries(portfolioData.allocations).map(([sector, pct], i) => (
                            <div key={i}>
                              <div className="flex justify-between text-[11px] mb-1.5">
                                <span className="text-white/50">{sector}</span>
                                <span className="font-mono text-white/70">{pct}%</span>
                              </div>
                              <div className="h-[3px] rounded-full overflow-hidden" style={{ background: 'rgba(255,255,255,0.04)' }}>
                                <motion.div initial={{ width: 0 }} animate={{ width: `${pct}%` }}
                                  transition={{ duration: 0.8, delay: i * 0.1 }}
                                  className="h-full rounded-full" style={{ background: '#E8D17A' }} />
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                      <div>
                        <div className="text-[9px] font-mono text-white/25 uppercase tracking-widest mb-4">AI Insight</div>
                        <p className="text-[12px] text-white/50 leading-relaxed italic">"{portfolioData.explanation || 'No explanation available.'}"</p>
                      </div>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          )}

          {/* ── TRADE SIMULATOR VIEW ─────────────────────── */}
          {viewMode === 'tradetest' && domain.id === 'finance' && (
            <div className="max-w-[820px] mx-auto w-full pb-10">
              <div className="mb-7">
                <div className="text-[9px] font-mono tracking-[0.14em] uppercase text-white/25 mb-1.5 flex items-center gap-2">
                  <CandlestickChart size={9} /> Backtester
                </div>
                <h2 className="text-[22px] font-serif text-white/90 leading-tight">Trade Simulator</h2>
                <p className="text-[12px] text-white/35 mt-1">Validate trading strategies against historical NSE market data.</p>
              </div>

              <div className="grid grid-cols-3 gap-3 mb-5">
                <div className="p-4 rounded-xl" style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.07)' }}>
                  <div className="text-[9px] font-mono text-white/25 uppercase tracking-widest mb-2">Ticker Symbol</div>
                  <input type="text" value={tradeForm.symbol}
                    onChange={e => setTradeForm({ ...tradeForm, symbol: e.target.value.toUpperCase() })}
                    className="w-full bg-transparent border-none outline-none text-[16px] font-mono font-semibold"
                    style={{ color: '#E8D17A' }} />
                  <div className="text-[9px] text-white/20 mt-1">e.g. RELIANCE.NS</div>
                </div>
                <div className="p-4 rounded-xl" style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.07)' }}>
                  <div className="text-[9px] font-mono text-white/25 uppercase tracking-widest mb-2">Strategy</div>
                  <select value={tradeForm.strategy} onChange={e => setTradeForm({ ...tradeForm, strategy: e.target.value })}
                    className="w-full bg-transparent border-none outline-none text-[12px] font-mono text-white/70">
                    <option value="SMA_Crossover" style={{ background: '#0b0b16' }}>SMA Crossover (50/200)</option>
                    <option value="RSI_Standard" style={{ background: '#0b0b16' }}>RSI Mean Reversion</option>
                  </select>
                </div>
                <div className="p-4 rounded-xl" style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.07)' }}>
                  <div className="text-[9px] font-mono text-white/25 uppercase tracking-widest mb-2">Time Period</div>
                  <select value={tradeForm.period} onChange={e => setTradeForm({ ...tradeForm, period: e.target.value })}
                    className="w-full bg-transparent border-none outline-none text-[12px] font-mono text-white/70">
                    {[['1mo','1 Month'],['6mo','6 Months'],['1y','1 Year'],['2y','2 Years']].map(([v,l]) => (
                      <option key={v} value={v} style={{ background: '#0b0b16' }}>{l}</option>
                    ))}
                  </select>
                </div>
              </div>

              <button onClick={handleTradeTestRun} disabled={financeLoading}
                className="w-full py-3.5 rounded-xl font-medium flex items-center justify-center gap-2 text-[13px] transition-all mb-6 disabled:opacity-40"
                style={{ background: '#E8D17A', color: '#08080f' }}>
                {financeLoading ? <RefreshCw size={15} className="animate-spin" /> : <TrendingUp size={15} />}
                Run Simulation
              </button>

              <AnimatePresence>
                {tradeTestData && (
                  <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} className="flex flex-col gap-3">
                    {/* Score banner */}
                    <div className="p-5 rounded-xl flex items-center justify-between"
                      style={{ background: 'rgba(232,209,122,0.04)', border: '1px solid rgba(232,209,122,0.12)' }}>
                      <div className="flex items-center gap-4">
                        <div className="w-10 h-10 rounded-xl flex items-center justify-center"
                          style={{ background: 'rgba(232,209,122,0.08)', border: '1px solid rgba(232,209,122,0.15)' }}>
                          <Activity size={18} style={{ color: '#E8D17A' }} />
                        </div>
                        <div>
                          <div className="text-[9px] font-mono text-white/25 uppercase tracking-widest">Strategy Score</div>
                          <div className="text-[18px] font-mono font-semibold text-white/90">
                            {tradeTestData.evaluation?.score || 0}<span className="text-white/30 text-[13px]">/100</span>
                            <span className="text-[13px] text-white/50 ml-2">— {tradeTestData.evaluation?.grade || 'N/A'}</span>
                          </div>
                        </div>
                      </div>
                      <div className={`px-4 py-2 rounded-full text-[12px] font-mono font-semibold`}
                        style={tradeTestData.results?.total_return >= 0
                          ? { background: 'rgba(110,207,160,0.08)', border: '1px solid rgba(110,207,160,0.2)', color: '#6ECFA0' }
                          : { background: 'rgba(239,68,68,0.08)', border: '1px solid rgba(239,68,68,0.2)', color: '#f87171' }}>
                        {tradeTestData.results?.total_return > 0 ? '+' : ''}{tradeTestData.results?.total_return?.toFixed(2)}%
                      </div>
                    </div>

                    {/* Stats grid */}
                    <div className="grid grid-cols-4 gap-3">
                      {[
                        { label: 'Win Rate',     value: `${((tradeTestData.results?.win_rate || 0)*100).toFixed(1)}%`, icon: <CheckCircle2 size={12}/> },
                        { label: 'Sharpe Ratio', value: (tradeTestData.results?.sharpe_ratio || 0).toFixed(2),        icon: <Activity size={12}/> },
                        { label: 'Max Drawdown', value: `${((tradeTestData.results?.max_drawdown || 0)*100).toFixed(1)}%`, icon: <ShieldCheck size={12}/> },
                        { label: 'Total Trades', value: tradeTestData.results?.total_trades || 0,                     icon: <Briefcase size={12}/> },
                      ].map((s, i) => (
                        <div key={i} className="p-4 rounded-xl" style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.06)' }}>
                          <div className="flex items-center gap-1.5 text-white/25 mb-2">{s.icon}<span className="text-[9px] font-mono uppercase tracking-wider">{s.label}</span></div>
                          <div className="text-[18px] font-mono font-semibold text-white/80">{s.value}</div>
                        </div>
                      ))}
                    </div>

                    {/* ML Insight */}
                    <div className="p-5 rounded-xl" style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.06)' }}>
                      <div className="flex items-center gap-2 mb-3">
                        <Cpu size={12} style={{ color: '#E8D17A' }} />
                        <span className="text-[9px] font-mono uppercase tracking-widest text-white/35">ML Evaluator Insight</span>
                      </div>
                      <p className="text-[12px] text-white/55 leading-relaxed">{tradeTestData.evaluation?.recommendation || 'No recommendation available.'}</p>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          )}

          {/* ── STRATEGY ADVISOR VIEW ────────────────────── */}
          {viewMode === 'strategy_advisor' && domain.id === 'finance' && (
            <div className="max-w-[820px] mx-auto w-full pb-10">
              <div className="mb-7">
                <div className="text-[9px] font-mono tracking-[0.14em] uppercase text-white/25 mb-1.5 flex items-center gap-2">
                  <Target size={9} /> Quant Engine
                </div>
                <h2 className="text-[22px] font-serif text-white/90 leading-tight">Strategy Advisor</h2>
                <p className="text-[12px] text-white/35 mt-1">Generate full strategy approaches grounded in institutional knowledge.</p>
              </div>

              <div className="flex flex-col gap-4 mb-6">
                {/* Intent textarea */}
                <div className="rounded-xl overflow-hidden transition-all"
                  style={{ border: '1px solid rgba(255,255,255,0.08)', background: 'rgba(255,255,255,0.02)' }}>
                  <div className="px-5 py-3" style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                    <span className="text-[9px] font-mono text-white/25 uppercase tracking-[0.12em]">Describe your strategy intent</span>
                  </div>
                  <div className="p-5">
                    <textarea value={input} onChange={e => setInput(e.target.value)}
                      placeholder="e.g. I want a trend-following strategy that uses volatility bands to identify breakouts and includes a trailing stop-loss."
                      className="w-full bg-transparent border-none outline-none text-[14px] font-serif text-white/70 resize-none leading-relaxed min-h-[90px] placeholder:text-white/20" />
                    <div className="flex justify-end mt-3">
                      <button onClick={handleStrategyAdvisorRun} disabled={financeLoading || !input.trim()}
                        className="flex items-center gap-2 py-2.5 px-6 rounded-lg text-[12px] font-mono font-medium transition-all disabled:opacity-40"
                        style={{ background: '#E8D17A', color: '#08080f' }}>
                        {financeLoading ? <RefreshCw size={13} className="animate-spin" /> : <Sparkles size={13} />}
                        Build Strategy
                      </button>
                    </div>
                  </div>
                </div>

                {/* Live market ticker */}
                <div className="p-4 rounded-xl flex items-center justify-between gap-4"
                  style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.07)' }}>
                  <div>
                    <div className="text-[9px] font-mono text-white/25 uppercase tracking-widest mb-1">Dynamic Market Analysis</div>
                    <div className="text-[11px] text-white/40">Generate a strategy based on current LIVE OHLC conditions for a ticker.</div>
                  </div>
                  <div className="flex items-center gap-2 flex-shrink-0">
                    <input type="text" value={marketSymbol} onChange={e => setMarketSymbol(e.target.value.toUpperCase())}
                      placeholder="TICKER"
                      className="w-24 px-3 py-2.5 rounded-lg bg-transparent text-[13px] font-mono text-center focus:outline-none transition-all"
                      style={{ border: '1px solid rgba(255,255,255,0.1)', color: '#E8D17A' }} />
                    <button onClick={handleMarketAnalysisRun} disabled={financeLoading || !marketSymbol.trim()}
                      className="flex items-center gap-2 py-2.5 px-5 rounded-lg text-[11px] font-mono transition-all disabled:opacity-40"
                      style={{ border: '1px solid rgba(232,209,122,0.25)', color: '#E8D17A', background: 'rgba(232,209,122,0.05)' }}>
                      {financeLoading ? <RefreshCw size={12} className="animate-spin" /> : <Activity size={12} />}
                      Analyze
                    </button>
                  </div>
                </div>
              </div>

              <AnimatePresence>
                {strategyAdvice && (
                  <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}
                    className="rounded-xl overflow-hidden"
                    style={{ border: '1px solid rgba(255,255,255,0.07)', background: 'rgba(255,255,255,0.015)' }}>
                    <div className="px-6 py-4 flex items-center justify-between"
                      style={{ borderBottom: '1px solid rgba(255,255,255,0.05)', background: 'rgba(232,209,122,0.03)' }}>
                      <div className="flex items-center gap-3">
                        <div className="w-8 h-8 rounded-lg flex items-center justify-center"
                          style={{ background: 'rgba(232,209,122,0.08)', border: '1px solid rgba(232,209,122,0.15)' }}>
                          <Briefcase size={14} style={{ color: '#E8D17A' }} />
                        </div>
                        <span className="text-[13px] font-medium text-white/80">Strategic Report</span>
                      </div>
                      <div className="flex items-center gap-1.5 text-[9px] font-mono text-white/20">
                        <Lock size={9} /> RAG SECURED
                      </div>
                    </div>
                    <div className="p-6">
                      <p className="text-[13px] leading-[1.85] text-white/60 whitespace-pre-wrap">
                        {typeof strategyAdvice.approach_report === 'string' ? strategyAdvice.approach_report : JSON.stringify(strategyAdvice.approach_report)}
                      </p>
                      {strategyAdvice.retrieved_context && (
                        <div className="mt-6 pt-5" style={{ borderTop: '1px solid rgba(255,255,255,0.05)' }}>
                          <div className="flex items-center gap-2 mb-3">
                            <Database size={10} style={{ color: '#E8D17A' }} />
                            <span className="text-[9px] font-mono uppercase tracking-widest text-white/25">Retrieved Knowledge Context</span>
                          </div>
                          <p className="text-[11px] text-white/30 italic leading-relaxed">"{strategyAdvice.retrieved_context}"</p>
                        </div>
                      )}
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          )}
        </div>

        {/* ── Chat input ─────────────────────────────── */}
        {viewMode === 'chat' && (
          <footer className="px-7 py-5 flex-shrink-0"
            style={{ borderTop: '1px solid rgba(255,255,255,0.06)', background: 'rgba(8,8,16,0.95)' }}>
            <div className="max-w-[760px] mx-auto">
              <div className="flex items-center gap-2 mb-3 flex-wrap">
                {['Explain capital ratios', 'How HIPAA affects ePHI?', 'Recent case precedents'].map((q, i) => (
                  <button key={i} onClick={() => setInput(q)}
                    className="text-[10px] font-mono text-white/30 px-3 py-1.5 rounded-full transition-all hover:text-white/60"
                    style={{ border: '1px solid rgba(255,255,255,0.07)', background: 'rgba(255,255,255,0.02)' }}>
                    {q}
                  </button>
                ))}
              </div>
              <div className="flex items-end gap-3 rounded-xl p-4 transition-all"
                style={{ border: '1px solid rgba(255,255,255,0.09)', background: 'rgba(255,255,255,0.025)' }}>
                <textarea value={input} onChange={e => setInput(e.target.value)}
                  onKeyDown={e => e.key === 'Enter' && !e.shiftKey && (e.preventDefault(), handleSend())}
                  placeholder="Ask your specialized query..."
                  className="flex-1 bg-transparent border-none outline-none text-[13px] text-white/70 resize-none leading-relaxed min-h-[22px] max-h-32 placeholder:text-white/20" />
                <button onClick={handleSend} disabled={!input.trim() || loading}
                  className="w-8 h-8 rounded-lg flex items-center justify-center transition-all hover:scale-105 active:scale-95 disabled:opacity-30 flex-shrink-0"
                  style={{ background: '#E8D17A' }}>
                  <Send size={14} style={{ color: '#08080f' }} />
                </button>
              </div>
            </div>
          </footer>
        )}
      </main>

      {/* ══════════════════ RIGHT PANEL ══════════════════ */}
      <aside className="w-[300px] flex-shrink-0 flex flex-col overflow-y-auto"
        style={{ borderLeft: '1px solid rgba(255,255,255,0.06)', background: '#0b0b16' }}>

        {/* Tab switcher */}
        <div className="flex border-b" style={{ borderColor: 'rgba(255,255,255,0.06)' }}>
          {[['metrics', 'Metrics'], ['graph', 'Graph'], ['insight', 'Insight']].map(([tab, label]) => (
            <button key={tab} onClick={() => setRightTab(tab)}
              className="flex-1 py-3.5 text-[10px] font-mono uppercase tracking-[0.1em] transition-all"
              style={rightTab === tab
                ? { color: '#E8D17A', borderBottom: '1px solid #E8D17A', background: 'rgba(232,209,122,0.03)' }
                : { color: 'rgba(255,255,255,0.25)', borderBottom: '1px solid transparent' }}>
              {label}
            </button>
          ))}
        </div>

        <div className="flex-1 p-5 flex flex-col gap-5 overflow-y-auto">

          {/* METRICS TAB */}
          {rightTab === 'metrics' && (
            <>
              <div>
                <div className="text-[9px] font-mono tracking-[0.12em] uppercase text-white/25 mb-4">Quality Metrics</div>
                <div className="flex flex-col gap-4">
                  <MetricBar label="Faithfulness"      value={92} color="text-emerald-400" />
                  <MetricBar label="Context Precision" value={88} color="text-[#E8D17A]" />
                  <MetricBar label="Context Recall"    value={74} color="text-[#A78BFA]" />
                </div>
              </div>

              <div>
                <div className="text-[9px] font-mono tracking-[0.12em] uppercase text-white/25 mb-3">Relevance Map</div>
                <div className="h-36 flex items-end gap-1.5 pb-2 mb-2">
                  {[40, 75, 92, 60, 45, 82, 55].map((h, i) => (
                    <motion.div key={i} initial={{ height: 0 }} animate={{ height: `${h}%` }}
                      transition={{ duration: 0.6, delay: i * 0.07, ease: 'easeOut' }}
                      className="flex-1 rounded-t-sm cursor-help transition-colors"
                      style={{ background: h > 70 ? 'rgba(232,209,122,0.3)' : 'rgba(255,255,255,0.07)' }}
                      title={`Source ${i+1}: ${h}% relevance`} />
                  ))}
                </div>
                <div className="flex justify-between text-[9px] font-mono text-white/20">
                  <span>Naive Hits</span>
                  <span>Graph Facts</span>
                </div>
              </div>

              <div className="p-4 rounded-xl" style={{ background: 'rgba(232,209,122,0.04)', border: '1px solid rgba(232,209,122,0.1)' }}>
                <div className="flex items-center justify-between mb-3">
                  <span className="text-[9px] font-mono uppercase tracking-widest text-white/25">RAG Method</span>
                  <Tag>{method}</Tag>
                </div>
                <div className="flex flex-col gap-2">
                  {[['Vector Hits', '12'], ['Graph Edges', '4'], ['Fusion Score', '0.87']].map(([k, v]) => (
                    <div key={k} className="flex justify-between text-[11px]">
                      <span className="text-white/30 font-mono">{k}</span>
                      <span className="text-white/60 font-mono">{v}</span>
                    </div>
                  ))}
                </div>
              </div>
            </>
          )}

          {/* GRAPH TAB */}
          {rightTab === 'graph' && (
            <>
              <div>
                <div className="text-[9px] font-mono tracking-[0.12em] uppercase text-white/25 mb-3">Graph Statistics</div>
                <div className="grid grid-cols-2 gap-2">
                  {[['Nodes', '284'], ['Edges', '1,042'], ['Clusters', '17'], ['Depth', '4']].map(([k, v]) => (
                    <div key={k} className="p-3 rounded-lg" style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.06)' }}>
                      <div className="text-[9px] font-mono text-white/25 mb-1">{k}</div>
                      <div className="text-[16px] font-mono font-semibold text-white/70">{v}</div>
                    </div>
                  ))}
                </div>
              </div>
              <div>
                <div className="text-[9px] font-mono tracking-[0.12em] uppercase text-white/25 mb-3">Recent Triplets</div>
                <div className="flex flex-col gap-2">
                  {[
                    { s: 'Entity A', p: 'OWNS', o: 'Entity B' },
                    { s: 'Entity B', p: 'LOCATED_IN', o: 'Singapore' },
                    { s: 'Policy X', p: 'GOVERNS', o: 'Market A' },
                  ].map((t, i) => (
                    <div key={i} className="p-3 rounded-lg flex flex-col gap-1.5" style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.05)' }}>
                      <div className="flex items-center gap-1.5 text-[10px] font-mono flex-wrap">
                        <span className="px-2 py-0.5 rounded text-[#E8D17A]" style={{ border: '1px solid rgba(232,209,122,0.2)', background: 'rgba(232,209,122,0.05)' }}>{t.s}</span>
                        <span className="text-white/20">→</span>
                        <span className="text-white/35">{t.p}</span>
                        <span className="text-white/20">→</span>
                        <span className="px-2 py-0.5 rounded text-[#A78BFA]" style={{ border: '1px solid rgba(167,139,250,0.2)', background: 'rgba(167,139,250,0.05)' }}>{t.o}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </>
          )}

          {/* INSIGHT TAB */}
          {rightTab === 'insight' && (
            <>
              <div>
                <div className="text-[9px] font-mono tracking-[0.12em] uppercase text-white/25 mb-3">Consultation Insights</div>
                <div className="p-4 rounded-xl" style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.06)' }}>
                  <div className="flex items-center gap-2 mb-2.5">
                    <Sparkles size={11} style={{ color: '#E8D17A' }} />
                    <span className="text-[10px] font-mono text-white/45">Hybrid Retrieval Edge</span>
                  </div>
                  <p className="text-[11px] text-white/40 leading-relaxed">
                    The hybrid retriever found a critical relationship between <strong className="text-white/65">Entity A</strong> and <strong className="text-white/65">Entity B</strong> in the Neo4j subgraph that was absent from vector search hits.
                  </p>
                </div>
              </div>
              <div>
                <div className="text-[9px] font-mono tracking-[0.12em] uppercase text-white/25 mb-3">Session Stats</div>
                <div className="flex flex-col gap-2">
                  {[['Queries', messages.filter(m => m.role==='user').length], ['Avg Latency', '1.2s'], ['Model', 'gemini-2.0-flash'], ['Method', method.toUpperCase()]].map(([k, v]) => (
                    <div key={k} className="flex justify-between py-2" style={{ borderBottom: '1px solid rgba(255,255,255,0.04)' }}>
                      <span className="text-[11px] font-mono text-white/25">{k}</span>
                      <span className="text-[11px] font-mono text-white/55">{v}</span>
                    </div>
                  ))}
                </div>
              </div>
            </>
          )}
        </div>
      </aside>
    </div>
  )
}

// ─── Sub-components ───────────────────────────────────────────────────────────
const SectionCard = ({ title, children }) => (
  <div className="rounded-xl overflow-hidden" style={{ border: '1px solid rgba(255,255,255,0.07)' }}>
    <div className="px-4 py-2.5" style={{ borderBottom: '1px solid rgba(255,255,255,0.05)', background: 'rgba(255,255,255,0.02)' }}>
      <span className="text-[9px] font-mono uppercase tracking-[0.12em] text-white/30">{title}</span>
    </div>
    <div className="p-4" style={{ background: 'rgba(255,255,255,0.01)' }}>{children}</div>
  </div>
)

const InputField = ({ label, type = 'text', value, onChange }) => (
  <div className="p-3 rounded-lg" style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.06)' }}>
    <div className="text-[9px] font-mono text-white/25 mb-1.5">{label}</div>
    <input type={type} value={value} onChange={e => onChange(e.target.value)}
      className="w-full bg-transparent border-none outline-none text-[13px] font-mono text-white/70" />
  </div>
)

const SelectField = ({ label, value, onChange, options }) => (
  <div className="p-3 rounded-lg" style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.06)' }}>
    <div className="text-[9px] font-mono text-white/25 mb-1.5">{label}</div>
    <select value={value} onChange={e => onChange(e.target.value)}
      className="w-full bg-transparent border-none outline-none text-[12px] font-mono text-white/60">
      {options.map(o => <option key={o.value} value={o.value} style={{ background: '#0b0b16' }}>{o.label}</option>)}
    </select>
  </div>
)

const ToggleBtn = ({ active, icon, label, onToggle }) => (
  <button onClick={onToggle}
    className="flex items-center justify-between p-3 rounded-lg text-[11px] font-mono transition-all w-full"
    style={active
      ? { background: 'rgba(232,209,122,0.06)', border: '1px solid rgba(232,209,122,0.2)', color: '#E8D17A' }
      : { background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.06)', color: 'rgba(255,255,255,0.35)' }}>
    <div className="flex items-center gap-2">{icon} {label}</div>
    <span className="text-[9px] tracking-widest">{active ? 'ON' : 'OFF'}</span>
  </button>
)

export default Consultation

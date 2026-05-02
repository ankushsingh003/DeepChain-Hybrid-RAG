import React, { useState, useRef, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Send, ArrowLeft, Database, Sparkles, Network,
  BarChart2, FileText, Activity, RefreshCw, CheckCircle2,
  Circle, PieChart, TrendingUp, ShieldCheck, Wallet,
  Briefcase, Calculator, Heart, ChevronRight, ChevronDown,
  Zap, Target, AlertTriangle, Lock, Eye, Code2,
  AreaChart, CandlestickChart, Cpu, BrainCircuit, Microscope
} from 'lucide-react'
import { 
  queryRAG, triggerIngestion, getPortfolioStrategy, 
  runTradeTest, getStrategyAdvice, getMarketStrategyAdvice,
  getMLAdvisory, triggerMLTraining, getMLModelStatus
} from '../services/api'

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
  const [mlAdvisoryData, setMlAdvisoryData] = useState(null)
  const [mlStatus, setMlStatus] = useState(null)
  const [marketSymbol, setMarketSymbol] = useState('RELIANCE.NS')
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
    if (domain.id !== 'finance' && ['strategy_advisor', 'portfolio', 'tradetest', 'ml_strategist'].includes(viewMode)) {
      setViewMode('chat')
    }
    if (domain.id === 'finance') {
      fetchMLStatus()
    }
  }, [domain.id])

  useEffect(() => {
    if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight
  }, [messages])

  const fetchMLStatus = async () => {
    try { setMlStatus(await getMLModelStatus()) } catch (e) { console.error(e) }
  }

  // ── Handlers ─────────────────────────────────────────────────────────────
  const handleSend = async () => {
    if (!input.trim() || loading) return
    const userMsg = { role: 'user', content: input, timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) }
    setMessages(prev => [...prev, userMsg])
    const currentInput = input
    setInput('')
    setLoading(true)
    setPipelineStep(1)
    
    try {
      setTimeout(() => setPipelineStep(2), 500)
      setTimeout(() => setPipelineStep(3), 1200)
      
      const data = await queryRAG(currentInput, method)
      setPipelineStep(4)
      
      setMessages(prev => [...prev, {
        role: 'assistant', 
        content: data.answer, 
        method: data.method,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        citations: data.citations || ['SYSTEM_GEN'],
        graph: data.graph || []
      }])
    } catch {
      setMessages(prev => [...prev, { role: 'assistant', content: "Error connecting to the intelligence engine.", timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) }])
    } finally { 
      setLoading(false)
      setTimeout(() => setPipelineStep(0), 1000)
    }
  }

  const handleIngestion = async () => {
    setIngesting(true)
    try { await triggerIngestion(); alert('Ingestion completed successfully!') }
    catch { alert('Ingestion failed.') }
    finally { setIngesting(false) }
  }

  const handleMLTrain = async (quick = true) => {
    setFinanceLoading(true)
    try { 
      await triggerMLTraining(quick)
      alert('ML Training cycle triggered in background.')
      fetchMLStatus()
    } catch { alert('ML Training failed.') }
    finally { setFinanceLoading(false) }
  }

  const handleMLAdvisoryRun = async () => {
    if (!marketSymbol.trim()) return
    setFinanceLoading(true)
    try { setMlAdvisoryData(await getMLAdvisory(marketSymbol)) }
    catch { alert('Neural advisory failed.') }
    finally { setFinanceLoading(false) }
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

  // ── Sidebar nav items ─────────────────────────────────────────────────────
  const financeTools = [
    { id: 'chat',            label: 'Expert Consultation', icon: <Sparkles size={13} /> },
    { id: 'ml_strategist',   label: 'Neural ML Advisor',    icon: <BrainCircuit size={13} /> },
    { id: 'portfolio',       label: 'Portfolio Allocation', icon: <PieChart size={13} /> },
    { id: 'tradetest',       label: 'Trade Simulator',      icon: <CandlestickChart size={13} /> },
    { id: 'strategy_advisor',label: 'Strategy Advisor',     icon: <Target size={13} /> },
  ]

  // ─────────────────────────────────────────────────────────────────────────
  return (
    <div className="flex h-[calc(100vh-65px)] overflow-hidden" style={{ background: '#080810' }}>

      {/* ══════════════════ LEFT SIDEBAR ══════════════════ */}
      <aside className="w-[280px] flex-shrink-0 flex flex-col overflow-y-auto border-r"
        style={{ borderColor: 'rgba(255,255,255,0.06)', background: '#0b0b16' }}>

        {/* Back */}
        <div className="px-5 pt-5 pb-4 border-b" style={{ borderColor: 'rgba(255,255,255,0.05)' }}>
          <button onClick={onBack}
            className="flex items-center gap-2 text-[10px] font-mono text-white/30 hover:text-white/60 transition-colors uppercase tracking-[0.12em]">
            <ArrowLeft size={12} /> Back to Hub
          </button>
        </div>

        {/* Domain Profile */}
        <div className="px-5 py-5 border-b" style={{ borderColor: 'rgba(255,255,255,0.05)' }}>
          <div className="flex items-center gap-4">
            <div className="w-11 h-11 rounded-2xl flex items-center justify-center text-xl shadow-inner shadow-white/5"
              style={{ background: 'rgba(232,209,122,0.06)', border: '1px solid rgba(232,209,122,0.12)' }}>
              {domain.icon}
            </div>
            <div>
              <div className="text-[14px] font-medium text-white/95 leading-tight">{domain.name}</div>
              <div className="text-[10px] font-mono text-[#E8D17A]/60 mt-0.5 tracking-wider">{domain.tag}</div>
            </div>
          </div>
        </div>

        {/* Knowledge Sources */}
        <div className="px-5 py-5 border-b" style={{ borderColor: 'rgba(255,255,255,0.05)' }}>
          <div className="text-[9px] font-mono tracking-[0.14em] uppercase text-white/20 mb-4 flex items-center gap-2">
            <Database size={9} /> Ingested Intelligence
          </div>
          <div className="flex flex-col gap-2">
            {[
              { name: 'SEC_Q3_SUMMARY.PDF', type: 'PDF' },
              { name: 'NSE_OHLCV_HISTORY', type: 'CSV' },
              { name: 'CORP_STATUTES_KB', type: 'Graph' }
            ].map((s, i) => (
              <div key={i} className="flex items-center gap-2.5 px-3 py-2 rounded-xl transition-all"
                style={{ background: 'rgba(255,255,255,0.01)', border: '1px solid rgba(255,255,255,0.03)' }}>
                <FileText size={11} className="text-white/20" />
                <span className="text-[10px] font-mono text-white/40 flex-1 truncate">{s.name}</span>
                <span className="text-[8px] font-mono text-[#E8D17A]/40">{s.type}</span>
              </div>
            ))}
          </div>
          <button onClick={handleIngestion} disabled={ingesting}
            className="w-full mt-4 flex items-center justify-center gap-2 py-2.5 rounded-xl text-[10px] font-mono transition-all disabled:opacity-40 hover:brightness-125"
            style={{ background: 'rgba(232,209,122,0.08)', border: '1px solid rgba(232,209,122,0.15)', color: '#E8D17A' }}>
            {ingesting ? <RefreshCw size={10} className="animate-spin" /> : <Zap size={10} />}
            Trigger Pipeline Ingestion
          </button>
        </div>

        {/* Financial Toolkit */}
        {domain.id === 'finance' && (
          <div className="px-5 py-5 border-b" style={{ borderColor: 'rgba(255,255,255,0.05)' }}>
            <div className="text-[9px] font-mono tracking-[0.14em] uppercase text-white/20 mb-4">Quant Command Center</div>
            <div className="flex flex-col gap-1.5">
              {financeTools.map(tool => (
                <button key={tool.id} onClick={() => setViewMode(tool.id)}
                  className="w-full flex items-center gap-3 py-3 px-4 rounded-xl text-[11px] font-mono transition-all text-left group"
                  style={viewMode === tool.id
                    ? { background: 'rgba(232,209,122,0.08)', border: '1px solid rgba(232,209,122,0.18)', color: '#E8D17A' }
                    : { background: 'transparent', border: '1px solid transparent', color: 'rgba(255,255,255,0.3)' }}>
                  <span className={viewMode === tool.id ? 'text-[#E8D17A]' : 'text-white/20 group-hover:text-white/40'}>{tool.icon}</span>
                  {tool.label}
                  {viewMode === tool.id && <motion.div layoutId="activeDot" className="w-1 h-1 rounded-full bg-[#E8D17A] ml-auto" />}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Real-time Pipeline Step */}
        <div className="px-5 py-5">
          <div className="text-[9px] font-mono tracking-[0.14em] uppercase text-white/20 mb-4">Fusion Processing</div>
          <div className="flex flex-col gap-3">
            {[
              { label: 'Decomposition', id: 1 },
              { label: 'Graph Retrieval', id: 2 },
              { label: 'Vector Similarity', id: 3 },
              { label: 'Synthesized Reponse', id: 4 }
            ].map(s => <PipelineStep key={s.id} {...s} current={pipelineStep} />)}
          </div>
        </div>
      </aside>

      {/* ══════════════════ CENTER MAIN ══════════════════ */}
      <main className="flex-1 flex flex-col min-w-0 overflow-hidden relative" style={{ background: 'radial-gradient(circle at 50% 0%, rgba(232,209,122,0.03) 0%, rgba(8,8,16,1) 100%)' }}>

        {/* Header */}
        <header className="px-8 py-5 flex items-center justify-between flex-shrink-0"
          style={{ borderBottom: '1px solid rgba(255,255,255,0.04)', background: 'rgba(8,8,16,0.8)', backdropFilter: 'blur(20px)' }}>
          <div className="flex items-center gap-5">
            <div>
              <div className="flex items-center gap-2.5">
                <span className="text-[14px] font-medium text-white/90">DeepChain Terminal</span>
                <span className="px-2 py-0.5 rounded text-[8px] font-mono bg-white/5 border border-white/10 text-white/30 tracking-tighter uppercase">v2.1 Stable</span>
              </div>
              <div className="flex items-center gap-1.5 mt-1">
                <Dot active />
                <span className="text-[10px] font-mono text-white/30">Active Intelligence Protocol — {method.toUpperCase()} Fusion</span>
              </div>
            </div>
          </div>

          <div className="flex items-center gap-3">
            {viewMode === 'chat' && (
              <div className="flex bg-white/5 p-1 rounded-xl border border-white/10">
                {['naive', 'hybrid'].map(m => (
                  <button key={m} onClick={() => setMethod(m)}
                    className="text-[10px] font-mono px-4 py-1.5 rounded-lg transition-all"
                    style={method === m
                      ? { background: 'rgba(232,209,122,0.1)', color: '#E8D17A' }
                      : { color: 'rgba(255,255,255,0.3)' }}>
                    {m.toUpperCase()}
                  </button>
                ))}
              </div>
            )}
            {viewMode !== 'chat' && (
              <button onClick={() => setViewMode('chat')}
                className="flex items-center gap-2 text-[10px] font-mono px-4 py-2 rounded-xl border border-white/10 text-white/40 hover:bg-white/5 transition-all">
                <ArrowLeft size={12} /> Return to Terminal
              </button>
            )}
          </div>
        </header>

        {/* Content Area */}
        <div ref={scrollRef} className="flex-1 overflow-y-auto p-8 flex flex-col gap-6 custom-scrollbar">

          {/* ── 💬 CHAT VIEW ─────────────────────────────── */}
          {viewMode === 'chat' && (
            <>
              {messages.map((msg, i) => (
                <motion.div key={i} initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}
                  className={`flex flex-col gap-3 max-w-[700px] ${msg.role === 'user' ? 'self-end items-end' : 'self-start items-start'}`}>
                  
                  <div className="flex items-center gap-2.5">
                    <div className={`w-6 h-6 rounded-lg flex items-center justify-center text-[10px] font-mono flex-shrink-0 ${msg.role === 'assistant' ? 'text-white/20' : 'text-white/40'}`}
                      style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.06)' }}>
                      {msg.role === 'assistant' ? <Sparkles size={11} /> : 'U'}
                    </div>
                    <span className="text-[10px] font-mono text-white/20 tracking-wider">
                      {msg.role === 'assistant' ? 'INTELLIGENCE_ENGINE' : 'USER_CLIENT'} · {msg.timestamp}
                    </span>
                    {msg.method && <Tag color={msg.method === 'hybrid' ? 'emerald-400' : 'white/20'}>{msg.method}</Tag>}
                  </div>

                  <div className={`text-[14px] leading-[1.8] ${msg.role === 'user' ? 'px-5 py-4 rounded-2xl rounded-tr-sm text-white/90 shadow-2xl' : 'text-white/70'}`}
                    style={msg.role === 'user' ? { background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.08)' } : {}}>
                    <p className="whitespace-pre-wrap">{msg.content}</p>

                    {msg.role === 'assistant' && msg.citations && (
                      <div className="flex flex-wrap gap-2 mt-5 pt-4" style={{ borderTop: '1px solid rgba(255,255,255,0.05)' }}>
                        {msg.citations.map((c, j) => (
                          <div key={j} className="flex items-center gap-1.5 px-2.5 py-1 rounded-md text-[9px] font-mono text-white/30 border border-white/5 bg-white/[0.01]">
                            <FileText size={9} /> {c}
                          </div>
                        ))}
                      </div>
                    )}

                    {msg.role === 'assistant' && msg.graph && msg.graph.length > 0 && (
                      <div className="mt-5 rounded-2xl overflow-hidden border border-white/5 bg-white/[0.01]">
                        <div className="px-4 py-2 flex items-center gap-2 text-[9px] font-mono text-white/20 border-b border-white/5">
                          <Network size={10} /> Neural Relational Map
                        </div>
                        <div className="p-4 flex flex-col gap-2">
                          {msg.graph.map((t, j) => (
                            <div key={j} className="flex items-center gap-3 text-[11px] font-mono">
                              <span className="px-3 py-1 rounded-lg text-[#E8D17A] bg-[#E8D17A]/5 border border-[#E8D17A]/10">{t.s}</span>
                              <span className="text-white/10 tracking-widest">─ {t.p} ─▶</span>
                              <span className="px-3 py-1 rounded-lg text-emerald-400 bg-emerald-400/5 border border-emerald-400/10">{t.o}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </motion.div>
              ))}

              {loading && (
                <div className="self-start flex items-center gap-4 py-4">
                  <div className="flex gap-1.5">
                    {[0,1,2].map(i => (
                      <motion.div key={i} className="w-1.5 h-1.5 rounded-full bg-[#E8D17A]/40"
                        animate={{ opacity: [0.3, 1, 0.3], scale: [0.8, 1.2, 0.8] }}
                        transition={{ duration: 1.5, repeat: Infinity, delay: i * 0.3 }} />
                    ))}
                  </div>
                  <span className="text-[10px] font-mono text-white/20 uppercase tracking-widest">Synthesizing Domain Response...</span>
                </div>
              )}
            </>
          )}

          {/* ── 🤖 NEURAL ML ADVISOR ──────────────────────────── */}
          {viewMode === 'ml_strategist' && domain.id === 'finance' && (
            <div className="max-w-[900px] mx-auto w-full pb-10">
              <div className="mb-10">
                <div className="text-[10px] font-mono tracking-[0.2em] uppercase text-[#E8D17A]/40 mb-3 flex items-center gap-2">
                  <BrainCircuit size={12} /> Neural Stock Engine
                </div>
                <h2 className="text-[28px] font-serif text-white/95 tracking-tight">Neural Strategy Advisor</h2>
                <p className="text-[14px] text-white/30 mt-2 max-w-xl leading-relaxed">
                  Predictive ML engine that scans fundamentals and technicals to determine the optimal mathematical strategy.
                </p>
              </div>

              <div className="grid grid-cols-12 gap-6">
                <div className="col-span-12 p-6 rounded-3xl border border-white/5 bg-white/[0.02]">
                  <div className="flex items-center justify-between mb-6">
                    <div className="flex items-center gap-3">
                      <div className="w-10 h-10 rounded-2xl flex items-center justify-center bg-[#E8D17A]/10 border border-[#E8D17A]/20">
                        <Microscope size={18} className="text-[#E8D17A]" />
                      </div>
                      <div>
                        <div className="text-[13px] font-medium text-white/90">DeepScan Analysis</div>
                        <div className="text-[10px] font-mono text-white/20 uppercase tracking-tighter">Enter NSE Ticker Symbol</div>
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <input type="text" value={marketSymbol} onChange={e => setMarketSymbol(e.target.value.toUpperCase())}
                        className="w-40 px-5 py-3 rounded-2xl bg-white/5 border border-white/10 text-white/80 font-mono text-[14px] focus:outline-none focus:border-[#E8D17A]/30 transition-all text-center"
                        placeholder="e.g. RELIANCE.NS" />
                      <button onClick={handleMLAdvisoryRun} disabled={financeLoading}
                        className="px-6 py-3 rounded-2xl bg-[#E8D17A] text-[#08080f] font-mono font-bold text-[12px] hover:brightness-110 active:scale-95 transition-all disabled:opacity-30">
                        {financeLoading ? <RefreshCw size={14} className="animate-spin" /> : "RUN ANALYSIS"}
                      </button>
                    </div>
                  </div>

                  {mlAdvisoryData && (
                    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="grid grid-cols-2 gap-6">
                      <div className="p-5 rounded-2xl bg-white/[0.02] border border-white/5">
                        <div className="flex items-center justify-between mb-4">
                          <span className="text-[10px] font-mono text-white/25 uppercase tracking-widest">ML Prediction</span>
                          <span className="px-3 py-1 rounded-full text-[10px] font-mono bg-emerald-400/10 text-emerald-400 border border-emerald-400/20">CONFIDENT</span>
                        </div>
                        <div className="text-[20px] font-medium text-[#E8D17A] mb-1">{mlAdvisoryData.recommended_strategy}</div>
                        <div className="text-[11px] text-white/30">Target Sharpe Ratio: <span className="text-white/60 font-mono">{(mlAdvisoryData.expected_sharpe || 0).toFixed(2)}</span></div>
                        
                        <div className="mt-6 flex flex-col gap-3">
                          <div className="flex justify-between items-center py-2 border-b border-white/5">
                            <span className="text-[12px] text-white/40">Backtest Return</span>
                            <span className="text-[14px] font-mono text-emerald-400">+{mlAdvisoryData.backtest_return?.toFixed(2)}%</span>
                          </div>
                          <div className="flex justify-between items-center py-2 border-b border-white/5">
                            <span className="text-[12px] text-white/40">Market Trend</span>
                            <span className="text-[12px] font-mono text-white/70">{mlAdvisoryData.market_trend}</span>
                          </div>
                        </div>
                      </div>
                      
                      <div className="p-5 rounded-2xl bg-white/[0.02] border border-white/5">
                        <div className="text-[10px] font-mono text-white/25 uppercase tracking-widest mb-4">Neural Logic</div>
                        <p className="text-[13px] text-white/60 leading-relaxed italic">
                          "{mlAdvisoryData.logic_explanation || "Analyzing market microstructure and volatility regimes to deliver optimal entry signals."}"
                        </p>
                      </div>
                    </motion.div>
                  )}
                </div>

                <div className="col-span-12 p-6 rounded-3xl border border-white/5 bg-white/[0.02]">
                  <div className="flex items-center justify-between mb-6">
                    <div>
                      <div className="text-[13px] font-medium text-white/90">Brain Status</div>
                      <div className="text-[10px] font-mono text-white/20 uppercase tracking-tighter">Model Last Trained: {mlStatus?.trained_at || "NEVER"}</div>
                    </div>
                    <div className="flex gap-2">
                      <button onClick={() => handleMLTrain(true)} disabled={financeLoading}
                        className="px-4 py-2 rounded-xl border border-white/10 text-[10px] font-mono text-white/50 hover:bg-white/5 transition-all">
                        QUICK TRAIN (10m)
                      </button>
                      <button onClick={() => handleMLTrain(false)} disabled={financeLoading}
                        className="px-4 py-2 rounded-xl border border-[#E8D17A]/20 text-[10px] font-mono text-[#E8D17A] bg-[#E8D17A]/5 hover:bg-[#E8D17A]/10 transition-all">
                        FULL RETRAIN (30m)
                      </button>
                    </div>
                  </div>
                  <div className="grid grid-cols-3 gap-4">
                    <div className="p-4 rounded-2xl bg-white/[0.01] border border-white/5">
                      <div className="text-[9px] font-mono text-white/20 uppercase tracking-widest mb-1">Classifier</div>
                      <div className={`text-[12px] font-mono ${mlStatus?.classifier_exists ? 'text-emerald-400' : 'text-red-400'}`}>
                        {mlStatus?.classifier_exists ? 'ONLINE' : 'MISSING'}
                      </div>
                    </div>
                    <div className="p-4 rounded-2xl bg-white/[0.01] border border-white/5">
                      <div className="text-[9px] font-mono text-white/20 uppercase tracking-widest mb-1">Regressor</div>
                      <div className={`text-[12px] font-mono ${mlStatus?.regressor_exists ? 'text-emerald-400' : 'text-red-400'}`}>
                        {mlStatus?.regressor_exists ? 'ONLINE' : 'MISSING'}
                      </div>
                    </div>
                    <div className="p-4 rounded-2xl bg-white/[0.01] border border-white/5">
                      <div className="text-[9px] font-mono text-white/20 uppercase tracking-widest mb-1">Nifty-50 Ready</div>
                      <div className="text-[12px] font-mono text-emerald-400">YES</div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* ── 📊 PORTFOLIO VIEW ──────────────────────────── */}
          {viewMode === 'portfolio' && domain.id === 'finance' && (
            <div className="max-w-[850px] mx-auto w-full pb-10">
              <div className="mb-10">
                <div className="text-[10px] font-mono tracking-[0.2em] uppercase text-white/25 mb-3 flex items-center gap-2">
                  <PieChart size={12} /> Wealth Architecture
                </div>
                <h2 className="text-[28px] font-serif text-white/95 tracking-tight">Portfolio Allocation</h2>
                <p className="text-[14px] text-white/30 mt-2 max-w-lg leading-relaxed">
                  Design a mathematically optimized asset distribution based on your risk profile and surplus income.
                </p>
              </div>

              <div className="grid grid-cols-2 gap-8 mb-10">
                {/* Inputs */}
                <div className="space-y-6">
                  <SectionCard title="CASH FLOW & ASSETS">
                    <div className="grid grid-cols-2 gap-4">
                      <InputField label="Monthly Income" value={profileForm.monthly_income} onChange={v => setProfileForm({ ...profileForm, monthly_income: +v })} />
                      <InputField label="Monthly Expenses" value={profileForm.monthly_expenses} onChange={v => setProfileForm({ ...profileForm, monthly_expenses: +v })} />
                      <InputField label="Current Savings" value={profileForm.existing_savings} onChange={v => setProfileForm({ ...profileForm, existing_savings: +v })} />
                      <InputField label="Target Investment" value={profileForm.amount_to_invest} onChange={v => setProfileForm({ ...profileForm, amount_to_invest: +v })} />
                    </div>
                  </SectionCard>
                  
                  <SectionCard title="RISK PARAMETERS">
                    <div className="grid grid-cols-2 gap-4">
                      <InputField label="Age" type="number" value={profileForm.age} onChange={v => setProfileForm({ ...profileForm, age: +v })} />
                      <SelectField label="Horizon" value={profileForm.investment_horizon}
                        onChange={v => setProfileForm({ ...profileForm, investment_horizon: v })}
                        options={[{ value: '1yr', label: 'Short' }, { value: '3yr', label: 'Medium' }, { value: '5yr', label: 'Long' }]} />
                    </div>
                  </SectionCard>
                </div>

                <div className="space-y-6 flex flex-col justify-between">
                  <SectionCard title="SECURITY PROTOCOLS">
                    <div className="space-y-3">
                      <ToggleBtn active={profileForm.health_insurance} icon={<ShieldCheck size={12} />} label="Health Insurance" onToggle={() => setProfileForm({ ...profileForm, health_insurance: !profileForm.health_insurance })} />
                      <ToggleBtn active={profileForm.life_insurance} icon={<Heart size={12} />} label="Life Insurance" onToggle={() => setProfileForm({ ...profileForm, life_insurance: !profileForm.life_insurance })} />
                    </div>
                  </SectionCard>

                  <button onClick={handlePortfolioRun} disabled={financeLoading}
                    className="w-full py-5 rounded-3xl bg-[#E8D17A] text-[#08080f] font-mono font-bold text-[14px] hover:brightness-110 active:scale-[0.98] transition-all shadow-xl shadow-[#E8D17A]/5 flex items-center justify-center gap-3">
                    {financeLoading ? <RefreshCw size={18} className="animate-spin" /> : <Calculator size={18} />}
                    GENERATE ALLOCATION
                  </button>
                </div>
              </div>

              {portfolioData && (
                <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
                  className="rounded-3xl border border-white/5 bg-white/[0.02] overflow-hidden shadow-2xl">
                  <div className="px-8 py-6 flex items-center justify-between bg-[#E8D17A]/5 border-b border-white/5">
                    <div>
                      <div className="text-[10px] font-mono text-[#E8D17A]/50 uppercase tracking-widest mb-1">STRATEGIC SUMMARY</div>
                      <div className="text-[18px] font-medium text-white/90">Portfolio Status: <span className="text-[#E8D17A]">{portfolioData.status}</span></div>
                    </div>
                    <div className="text-right">
                      <div className="text-[10px] font-mono text-white/20 uppercase tracking-widest mb-1">Risk Profile</div>
                      <div className="text-[14px] font-mono text-emerald-400 uppercase">{portfolioData.risk_profile}</div>
                    </div>
                  </div>
                  <div className="p-8 grid grid-cols-2 gap-10">
                    <div className="space-y-6">
                      <div className="text-[10px] font-mono text-white/20 uppercase tracking-widest">Target Allocations</div>
                      <div className="space-y-4">
                        {portfolioData.allocations && Object.entries(portfolioData.allocations).map(([k, v]) => (
                          <div key={k}>
                            <div className="flex justify-between text-[12px] mb-2 font-mono">
                              <span className="text-white/40">{k}</span>
                              <span className="text-white/80">{v}%</span>
                            </div>
                            <div className="h-1 rounded-full bg-white/5 overflow-hidden">
                              <motion.div initial={{ width: 0 }} animate={{ width: `${v}%` }} className="h-full bg-[#E8D17A]" />
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                    <div>
                      <div className="text-[10px] font-mono text-white/20 uppercase tracking-widest mb-4">Neural Advisor Insight</div>
                      <p className="text-[14px] text-white/60 leading-[1.8] italic font-serif">
                        "{portfolioData.explanation}"
                      </p>
                    </div>
                  </div>
                </motion.div>
              )}
            </div>
          )}

          {/* ── 📈 TRADE SIMULATOR VIEW ─────────────────────── */}
          {viewMode === 'tradetest' && domain.id === 'finance' && (
            <div className="max-w-[850px] mx-auto w-full pb-10">
              <div className="mb-10">
                <div className="text-[10px] font-mono tracking-[0.2em] uppercase text-white/25 mb-3 flex items-center gap-2">
                  <CandlestickChart size={12} /> Algo Sandbox
                </div>
                <h2 className="text-[28px] font-serif text-white/95 tracking-tight">Strategy Backtester</h2>
                <p className="text-[14px] text-white/30 mt-2 max-w-lg leading-relaxed">
                  Run high-frequency simulations of technical strategies against multi-year historical data.
                </p>
              </div>

              <div className="grid grid-cols-3 gap-4 mb-6">
                <div className="p-5 rounded-2xl bg-white/[0.02] border border-white/5">
                  <div className="text-[9px] font-mono text-white/25 uppercase tracking-widest mb-2">SYMBOL</div>
                  <input type="text" value={tradeForm.symbol} onChange={e => setTradeForm({ ...tradeForm, symbol: e.target.value.toUpperCase() })}
                    className="w-full bg-transparent border-none outline-none text-[18px] font-mono text-[#E8D17A]" />
                </div>
                <div className="p-5 rounded-2xl bg-white/[0.02] border border-white/5">
                  <div className="text-[9px] font-mono text-white/25 uppercase tracking-widest mb-2">ALGO</div>
                  <select value={tradeForm.strategy} onChange={e => setTradeForm({ ...tradeForm, strategy: e.target.value })}
                    className="w-full bg-transparent border-none outline-none text-[12px] font-mono text-white/60">
                    <option value="SMA_Crossover" style={{ background: '#0b0b16' }}>SMA CROSSOVER</option>
                    <option value="RSI_Standard" style={{ background: '#0b0b16' }}>RSI MEAN REVERSION</option>
                  </select>
                </div>
                <div className="p-5 rounded-2xl bg-white/[0.02] border border-white/5">
                  <div className="text-[9px] font-mono text-white/25 uppercase tracking-widest mb-2">PERIOD</div>
                  <select value={tradeForm.period} onChange={e => setTradeForm({ ...tradeForm, period: e.target.value })}
                    className="w-full bg-transparent border-none outline-none text-[12px] font-mono text-white/60">
                    {[['1mo','1 MONTH'],['1y','1 YEAR'],['2y','2 YEARS']].map(([v,l]) => <option key={v} value={v} style={{ background: '#0b0b16' }}>{l}</option>)}
                  </select>
                </div>
              </div>

              <button onClick={handleTradeTestRun} disabled={financeLoading}
                className="w-full py-5 rounded-3xl bg-[#E8D17A] text-[#08080f] font-mono font-bold text-[14px] hover:brightness-110 mb-10 transition-all">
                {financeLoading ? <RefreshCw size={18} className="animate-spin" /> : "EXECUTE SIMULATION"}
              </button>

              {tradeTestData && (
                <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-6">
                  <div className="p-8 rounded-3xl border border-white/5 bg-white/[0.02] flex items-center justify-between">
                    <div>
                      <div className="text-[10px] font-mono text-white/20 uppercase tracking-[0.2em] mb-2">QUANT EVALUATION</div>
                      <div className="text-[32px] font-mono font-bold text-white/90">
                        {tradeTestData.evaluation?.score}<span className="text-white/20 text-[20px]">/100</span>
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="text-[10px] font-mono text-white/20 uppercase tracking-[0.2em] mb-2">TOTAL RETURN</div>
                      <div className={`text-[32px] font-mono font-bold ${tradeTestData.results?.total_return >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                        {tradeTestData.results?.total_return > 0 ? '+' : ''}{tradeTestData.results?.total_return?.toFixed(2)}%
                      </div>
                    </div>
                  </div>
                  
                  <div className="grid grid-cols-4 gap-4">
                    {[
                      { l: 'WIN RATE', v: `${((tradeTestData.results?.win_rate || 0)*100).toFixed(1)}%` },
                      { l: 'SHARPE', v: (tradeTestData.results?.sharpe_ratio || 0).toFixed(2) },
                      { l: 'MAX DRAWDOWN', v: `${((tradeTestData.results?.max_drawdown || 0)*100).toFixed(1)}%` },
                      { l: 'TRADES', v: tradeTestData.results?.total_trades || 0 }
                    ].map((s, i) => (
                      <div key={i} className="p-5 rounded-2xl bg-white/[0.02] border border-white/5">
                        <div className="text-[9px] font-mono text-white/20 uppercase tracking-widest mb-2">{s.l}</div>
                        <div className="text-[18px] font-mono text-white/80">{s.v}</div>
                      </div>
                    ))}
                  </div>
                </motion.div>
              )}
            </div>
          )}

          {/* ── 🎯 STRATEGY ADVISOR VIEW ────────────────────── */}
          {viewMode === 'strategy_advisor' && domain.id === 'finance' && (
            <div className="max-w-[850px] mx-auto w-full pb-10">
              <div className="mb-10">
                <div className="text-[10px] font-mono tracking-[0.2em] uppercase text-white/25 mb-3 flex items-center gap-2">
                  <Target size={12} /> Institutional Alpha
                </div>
                <h2 className="text-[28px] font-serif text-white/95 tracking-tight">Quant Strategy Builder</h2>
                <p className="text-[14px] text-white/30 mt-2 max-w-lg leading-relaxed">
                  Synthesize complex trading logic by bridging institutional knowledge with RAG-fused generation.
                </p>
              </div>

              <div className="p-8 rounded-3xl bg-white/[0.02] border border-white/5 mb-8">
                <div className="text-[10px] font-mono text-white/20 uppercase tracking-widest mb-4">Strategic Intent</div>
                <textarea value={input} onChange={e => setInput(e.target.value)}
                  placeholder="Describe the alpha you want to capture..."
                  className="w-full bg-transparent border-none outline-none text-[18px] font-serif text-white/80 resize-none h-32 leading-relaxed" />
                <div className="flex justify-between items-center mt-6">
                  <div className="flex gap-2">
                    <input type="text" value={marketSymbol} onChange={e => setMarketSymbol(e.target.value.toUpperCase())}
                      className="w-32 bg-white/5 border border-white/10 rounded-xl px-4 py-2.5 text-[11px] font-mono text-[#E8D17A]" placeholder="TICKER" />
                    <button onClick={handleMarketAnalysisRun} disabled={financeLoading}
                      className="px-5 py-2.5 rounded-xl border border-[#E8D17A]/30 text-[#E8D17A] text-[10px] font-mono hover:bg-[#E8D17A]/5 transition-all">
                      LIVE SCAN
                    </button>
                  </div>
                  <button onClick={handleStrategyAdvisorRun} disabled={financeLoading || !input.trim()}
                    className="px-8 py-3 rounded-2xl bg-[#E8D17A] text-[#08080f] font-mono font-bold text-[12px] shadow-lg shadow-[#E8D17A]/10 transition-all hover:brightness-110 active:scale-95">
                    {financeLoading ? <RefreshCw size={14} className="animate-spin" /> : "BUILD STRATEGY"}
                  </button>
                </div>
              </div>

              {strategyAdvice && (
                <motion.div initial={{ opacity: 0, scale: 0.98 }} animate={{ opacity: 1, scale: 1 }}
                  className="p-10 rounded-3xl border border-white/5 bg-white/[0.015] shadow-2xl relative overflow-hidden">
                  <div className="absolute top-0 right-0 p-6 opacity-10">
                    <Target size={120} />
                  </div>
                  <div className="relative z-10">
                    <div className="text-[10px] font-mono text-[#E8D17A] uppercase tracking-[0.3em] mb-6">STRATEGIC_ALPHA_REPORT</div>
                    <div className="text-[15px] leading-[2.2] text-white/70 font-serif whitespace-pre-wrap">
                      {typeof strategyAdvice.approach_report === 'string' ? strategyAdvice.approach_report : JSON.stringify(strategyAdvice.approach_report, null, 2)}
                    </div>
                  </div>
                </motion.div>
              )}
            </div>
          )}
        </div>

        {/* Chat input footer */}
        {viewMode === 'chat' && (
          <footer className="px-8 py-6 flex-shrink-0"
            style={{ borderTop: '1px solid rgba(255,255,255,0.04)', background: 'rgba(8,8,16,0.95)' }}>
            <div className="max-w-[800px] mx-auto">
              <div className="flex items-end gap-4 p-5 rounded-2xl transition-all shadow-2xl group focus-within:border-white/20"
                style={{ border: '1px solid rgba(255,255,255,0.07)', background: 'rgba(255,255,255,0.02)' }}>
                <textarea value={input} onChange={e => setInput(e.target.value)}
                  onKeyDown={e => e.key === 'Enter' && !e.shiftKey && (e.preventDefault(), handleSend())}
                  placeholder="Enter domain query or strategic requirement..."
                  className="flex-1 bg-transparent border-none outline-none text-[14px] text-white/70 resize-none leading-relaxed min-h-[24px] max-h-48 placeholder:text-white/20" />
                <button onClick={handleSend} disabled={!input.trim() || loading}
                  className="w-10 h-10 rounded-xl flex items-center justify-center transition-all hover:scale-110 active:scale-90 disabled:opacity-20 flex-shrink-0"
                  style={{ background: '#E8D17A' }}>
                  <Send size={18} style={{ color: '#08080f' }} />
                </button>
              </div>
            </div>
          </footer>
        )}
      </main>

      {/* ══════════════════ RIGHT PANEL ══════════════════ */}
      <aside className="w-[320px] flex-shrink-0 flex flex-col overflow-y-auto border-l"
        style={{ borderColor: 'rgba(255,255,255,0.06)', background: '#0b0b16' }}>

        {/* Tab switch */}
        <div className="flex border-b" style={{ borderColor: 'rgba(255,255,255,0.05)' }}>
          {[['metrics', 'Telemetry'], ['graph', 'Topology'], ['insight', 'Neural']].map(([tab, label]) => (
            <button key={tab} onClick={() => setRightTab(tab)}
              className="flex-1 py-4 text-[10px] font-mono uppercase tracking-[0.15em] transition-all"
              style={rightTab === tab
                ? { color: '#E8D17A', borderBottom: '1px solid #E8D17A', background: 'rgba(232,209,122,0.03)' }
                : { color: 'rgba(255,255,255,0.2)', borderBottom: '1px solid transparent' }}>
              {label}
            </button>
          ))}
        </div>

        <div className="flex-1 p-6 flex flex-col gap-8 overflow-y-auto custom-scrollbar">

          {rightTab === 'metrics' && (
            <>
              <div>
                <div className="text-[10px] font-mono tracking-[0.2em] uppercase text-white/20 mb-6 flex items-center gap-2">
                  <Activity size={10} /> RAG TELEMETRY
                </div>
                <div className="space-y-6">
                  <MetricBar label="Faithfulness Score" value={94} color="text-emerald-400" />
                  <MetricBar label="Context Precision"  value={89} color="text-[#E8D17A]" />
                  <MetricBar label="Syntactic Coherence" value={91} color="text-[#A78BFA]" />
                </div>
              </div>

              <div className="p-5 rounded-2xl bg-white/[0.02] border border-white/5 mt-4">
                <div className="flex items-center justify-between mb-4">
                  <span className="text-[9px] font-mono uppercase tracking-[0.2em] text-white/20">RETRIEVAL LATENCY</span>
                  <Tag>1.2s</Tag>
                </div>
                <div className="h-24 flex items-end gap-1 px-1">
                  {[30, 50, 80, 45, 95, 60, 40, 70, 55, 85].map((h, i) => (
                    <motion.div key={i} initial={{ height: 0 }} animate={{ height: `${h}%` }}
                      transition={{ delay: i * 0.05 }} className="flex-1 rounded-t-sm bg-white/5" />
                  ))}
                </div>
              </div>
            </>
          )}

          {rightTab === 'graph' && (
            <>
              <div className="p-5 rounded-2xl bg-white/[0.02] border border-white/5">
                <div className="text-[10px] font-mono tracking-[0.2em] uppercase text-white/20 mb-4">Topology Health</div>
                <div className="grid grid-cols-2 gap-3">
                  {[['Entities', '4.2k'], ['Edges', '12.8k'], ['Depth', '5'], ['Clustering', '0.42']].map(([k,v]) => (
                    <div key={k} className="p-3 bg-white/5 rounded-xl border border-white/5">
                      <div className="text-[9px] font-mono text-white/20 mb-1">{k}</div>
                      <div className="text-[14px] font-mono font-bold text-white/70">{v}</div>
                    </div>
                  ))}
                </div>
              </div>
              <div>
                <div className="text-[10px] font-mono tracking-[0.2em] uppercase text-white/20 mb-4">Recent Clusters</div>
                <div className="space-y-3">
                  {[
                    { n: 'EQUITY_MARKETS', e: 142, c: '#E8D17A' },
                    { n: 'REGULATORY_STATUTES', e: 89, c: '#A78BFA' },
                    { n: 'CORP_STRUCTURE', e: 214, c: 'emerald-400' }
                  ].map((c, i) => (
                    <div key={i} className="flex items-center gap-3 p-3 rounded-xl hover:bg-white/5 transition-all cursor-pointer">
                      <div className="w-1.5 h-1.5 rounded-full" style={{ background: c.c.startsWith('#') ? c.c : undefined }} />
                      <span className="text-[11px] font-mono text-white/40 flex-1">{c.n}</span>
                      <span className="text-[10px] font-mono text-white/20">{c.e}</span>
                    </div>
                  ))}
                </div>
              </div>
            </>
          )}

          {rightTab === 'insight' && (
            <div className="space-y-6">
              <div className="p-5 rounded-2xl bg-[#E8D17A]/5 border border-[#E8D17A]/10">
                <div className="flex items-center gap-2 mb-3">
                  <Sparkles size={12} className="text-[#E8D17A]" />
                  <span className="text-[10px] font-mono uppercase text-[#E8D17A]/80 tracking-[0.2em]">NEURAL EDGE</span>
                </div>
                <p className="text-[12px] text-white/50 leading-[1.8] italic font-serif">
                  The hybrid protocol successfully resolved a multi-hop dependency between Section 144 of the Companies Act and the provided SEC Q3 filing, yielding 14% higher coherence than standard vector search.
                </p>
              </div>
              <div className="space-y-4">
                <div className="text-[10px] font-mono tracking-[0.2em] uppercase text-white/20">Session Intel</div>
                {[['Queries Ingested', messages.filter(m=>m.role==='user').length], ['API Uptime', '99.9%'], ['Active Agent', 'Dominion-7']].map(([k,v]) => (
                  <div key={k} className="flex justify-between items-center py-2 border-b border-white/5">
                    <span className="text-[11px] font-mono text-white/30">{k}</span>
                    <span className="text-[11px] font-mono text-white/60">{v}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </aside>
    </div>
  )
}

// ─── Sub-components ───────────────────────────────────────────────────────────
const SectionCard = ({ title, children }) => (
  <div className="rounded-3xl overflow-hidden shadow-sm" style={{ border: '1px solid rgba(255,255,255,0.06)' }}>
    <div className="px-5 py-3" style={{ borderBottom: '1px solid rgba(255,255,255,0.04)', background: 'rgba(255,255,255,0.02)' }}>
      <span className="text-[10px] font-mono uppercase tracking-[0.2em] text-white/20">{title}</span>
    </div>
    <div className="p-5" style={{ background: 'rgba(255,255,255,0.01)' }}>{children}</div>
  </div>
)

const InputField = ({ label, type = 'text', value, onChange }) => (
  <div className="p-4 rounded-2xl transition-all focus-within:border-white/20" style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.06)' }}>
    <div className="text-[10px] font-mono text-white/20 mb-2 tracking-widest">{label}</div>
    <input type={type} value={value} onChange={e => onChange(e.target.value)}
      className="w-full bg-transparent border-none outline-none text-[14px] font-mono text-white/80" />
  </div>
)

const SelectField = ({ label, value, onChange, options }) => (
  <div className="p-4 rounded-2xl" style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.06)' }}>
    <div className="text-[10px] font-mono text-white/20 mb-2 tracking-widest">{label}</div>
    <select value={value} onChange={e => onChange(e.target.value)}
      className="w-full bg-transparent border-none outline-none text-[13px] font-mono text-white/60 cursor-pointer">
      {options.map(o => <option key={o.value} value={o.value} style={{ background: '#0b0b16' }}>{o.label}</option>)}
    </select>
  </div>
)

const ToggleBtn = ({ active, icon, label, onToggle }) => (
  <button onClick={onToggle}
    className="flex items-center justify-between p-4 rounded-2xl text-[12px] font-mono transition-all w-full group"
    style={active
      ? { background: 'rgba(232,209,122,0.08)', border: '1px solid rgba(232,209,122,0.2)', color: '#E8D17A' }
      : { background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.06)', color: 'rgba(255,255,255,0.3)' }}>
    <div className="flex items-center gap-3">{icon} {label}</div>
    <span className="text-[10px] tracking-[0.2em]">{active ? 'PROTOCOL_ON' : 'PROTOCOL_OFF'}</span>
  </button>
)

export default Consultation

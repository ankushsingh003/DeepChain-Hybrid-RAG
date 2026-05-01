import React from 'react'
import { ArrowRight, Zap, Database, Shield, Cpu, BarChart3, Network } from 'lucide-react'
import { motion } from 'framer-motion'

const Landing = ({ onGetConsulted }) => {
  return (
    <div className="flex flex-col items-center">
      {/* NAV */}
      <nav className="w-full flex items-center justify-between px-12 py-5 backdrop-blur-xl sticky top-0 z-[100] bg-bg/85 border-b border-border">
        <a href="#" className="flex items-center gap-2.5 font-serif text-xl tracking-tight text-text">
          <div className="w-2 h-2 rounded-full bg-finance" />
          DeepChain
        </a>
        <div className="flex gap-2">
          <button className="nav-pill">Architecture</button>
          <button className="nav-pill">Benchmarks</button>
          <button className="nav-pill">Stack</button>
        </div>
      </nav>

      {/* HERO */}
      <section className="min-h-[calc(100vh-65px)] flex flex-col items-center justify-center text-center px-12 py-20 relative">
        <div className="hero-badge">
          v2.1 Enterprise Release
        </div>
        
        <h1 className="font-serif text-[clamp(32px,5vw,72px)] leading-none tracking-[-2px] text-text mb-7 w-full">
          Enterprise Knowledge Intelligence.
        </h1>

        <p className="text-[17px] leading-[1.65] text-muted max-w-[520px] mb-14 font-light">
          Combine Knowledge Graphs with Vector Embeddings to eliminate hallucinations in high-stakes domains.
        </p>

        <button 
          onClick={onGetConsulted}
          className="inline-flex items-center gap-2.5 text-[15px] font-medium text-bg bg-text rounded-full px-9 py-4 transition-all hover:-translate-y-0.5 hover:shadow-[0_12px_40px_rgba(240,237,232,0.15)] group"
        >
          Get Consulted
          <ArrowRight className="w-4 h-4 transition-transform group-hover:translate-x-1" />
        </button>

        <div className="mt-20 flex gap-16 border-t border-border pt-10">
          <div className="text-center">
            <span className="font-serif text-3xl text-text block leading-none mb-1.5">98.4%</span>
            <span className="text-[12px] text-dim font-mono tracking-wider uppercase">Accuracy</span>
          </div>
          <div className="text-center">
            <span className="font-serif text-3xl text-text block leading-none mb-1.5">&lt; 2s</span>
            <span className="text-[12px] text-dim font-mono tracking-wider uppercase">Latency</span>
          </div>
          <div className="text-center">
            <span className="font-serif text-3xl text-text block leading-none mb-1.5">10M+</span>
            <span className="text-[12px] text-dim font-mono tracking-wider uppercase">Nodes</span>
          </div>
        </div>
      </section>

      {/* DIFFERENTIATORS */}
      <section className="py-20 px-12 max-w-[1100px] w-full mx-auto">
        <div className="flex items-baseline gap-4 mb-12">
          <span className="font-mono text-[11px] text-dim tracking-[0.1em] uppercase">Why DeepChain</span>
          <h2 className="font-serif text-3xl tracking-tight text-text">Hybrid intelligence vs Vanilla RAG</h2>
        </div>

        <div className="grid grid-cols-3 gap-[1px] bg-border border border-border rounded-2xl overflow-hidden">
          {[
            { icon: <Zap />, title: "Fact Grounding", desc: "Verifies every claim against structured graph triplets before generation." },
            { icon: <Network />, title: "Relationship Mapping", desc: "Understands multi-hop connections that vector search often misses." },
            { icon: <Shield />, title: "Domain Specificity", desc: "Tailored extraction engines for Legal, Finance, and Healthcare." },
            { icon: <Cpu />, title: "Hybrid Retrieval", desc: "Fuses Neo4j graph context with Weaviate vector hits in real-time." },
            { icon: <BarChart3 />, title: "Ragas Benchmarked", desc: "Automated faithfulness and relevancy scoring on every response." },
            { icon: <Database />, title: "Scalable Ingestion", desc: "Handles millions of documents with sub-second hybrid retrieval." },
          ].map((item, i) => (
            <div key={i} className="bg-bg p-8 transition-colors hover:bg-bg2">
              <div className="w-9 h-9 rounded-lg flex items-center justify-center mb-5 text-finance bg-finance-bg border border-finance/20">
                {React.cloneElement(item.icon, { size: 18 })}
              </div>
              <h3 className="text-[15px] font-medium mb-2 text-text">{item.title}</h3>
              <p className="text-[13px] text-muted leading-relaxed font-light">{item.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* TECH STRIP */}
      <div className="w-full py-10 px-12 border-y border-border flex items-center gap-8 overflow-x-auto mb-20 no-scrollbar">
        <span className="text-[11px] text-dim font-mono whitespace-nowrap tracking-widest uppercase">Tech Stack</span>
        {[
          { label: "LangChain", color: "bg-finance" },
          { label: "Neo4j", color: "bg-legal" },
          { label: "Weaviate", color: "bg-health" },
          { label: "FastAPI", color: "bg-mental" },
          { label: "React", color: "bg-finance" },
          { label: "Google Gemini", color: "bg-legal" },
          { label: "MLflow", color: "bg-health" },
        ].map((tech, i) => (
          <div key={i} className="inline-flex items-center gap-1.5 text-[12px] font-mono text-muted border border-border rounded-full px-3.5 py-1.5 whitespace-nowrap bg-bg2">
            <div className={`w-1.5 h-1.5 rounded-full ${tech.color}`} />
            {tech.label}
          </div>
        ))}
      </div>
    </div>
  )
}

export default Landing

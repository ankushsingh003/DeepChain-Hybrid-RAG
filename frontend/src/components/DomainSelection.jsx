import React from 'react'
import { motion } from 'framer-motion'
import { ArrowLeft, Landmark, Stethoscope, Scale, Heart, ArrowRight } from 'lucide-react'

const domains = [
  {
    id: 'finance',
    name: 'Finance & Markets',
    desc: 'Analyze quarterly reports, market trends, and regulatory compliance with precision.',
    icon: <Landmark />,
    color: 'finance',
    tag: 'Fin-RAG',
    features: ['SEC Filings', 'Portfolio Analysis', 'Risk Assessment'],
    sources: '8.4k Documents'
  },
  {
    id: 'health',
    name: 'Healthcare & Research',
    desc: 'Navigate medical journals, clinical trials, and patient protocols with cross-verified facts.',
    icon: <Stethoscope />,
    color: 'health',
    tag: 'Med-Graph',
    features: ['PubMed Integration', 'Clinical Trials', 'Drug Interaction'],
    sources: '12k Articles'
  },
  {
    id: 'legal',
    name: 'Legal Proceedings',
    desc: 'Sift through case law, statutes, and contracts with multi-hop relationship extraction.',
    icon: <Scale />,
    color: 'legal',
    tag: 'Law-Chain',
    features: ['Case Precedents', 'Statutory Analysis', 'Contract Review'],
    sources: '5.2k Statutes'
  },
  {
    id: 'mental',
    name: 'Mental Health & Psych',
    desc: 'Access psychological research and therapy frameworks with high empathy and accuracy.',
    icon: <Heart />,
    color: 'mental',
    tag: 'Psy-Assist',
    features: ['CBT Frameworks', 'Peer Reviews', 'Behavioral Data'],
    sources: '3.1k Papers'
  }
]

const DomainSelection = ({ onBack, onSelectDomain }) => {
  return (
    <div className="py-16 px-12 max-w-[1100px] mx-auto min-h-screen">
      <div className="mb-14">
        <div className="flex items-center gap-3 mb-4">
          <button onClick={onBack} className="back-btn group">
            <ArrowLeft className="w-3.5 h-3.5 transition-transform group-hover:-translate-x-0.5" />
            Back
          </button>
        </div>
        <h1 className="font-serif text-4xl tracking-tight mb-3 leading-tight">Select your domain of <em>Consultation</em></h1>
        <p className="text-[15px] text-muted font-light">Choose a specialized knowledge base for your intelligence requirements.</p>
      </div>

      <div className="grid grid-cols-2 gap-5">
        {domains.map((domain, i) => (
          <motion.div
            key={domain.id}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.1 }}
            onClick={() => onSelectDomain(domain)}
            className={`domain-card group ${domain.id}`}
            style={{ '--accent-color': `var(--${domain.color})` }}
          >
            <div className="flex justify-between items-start mb-7">
              <div className={`w-[52px] h-[52px] rounded-xl flex items-center justify-center text-22 border bg-${domain.id}-bg border-${domain.id}/20 text-${domain.color}`}>
                {React.cloneElement(domain.icon, { size: 22 })}
              </div>
              <span className={`domain-tag text-[10px] font-mono tracking-wider uppercase px-2.5 py-1 rounded-full border border-${domain.id}/30 bg-${domain.id}-bg text-${domain.color}`}>
                {domain.tag}
              </span>
            </div>
            
            <h2 className="font-serif text-2xl mb-2.5 tracking-tight leading-[1.15] text-text">{domain.name}</h2>
            <p className="text-[13px] text-muted leading-[1.65] mb-7 font-light">{domain.desc}</p>
            
            <div className="flex flex-col gap-2.5 mb-7">
              {domain.features.map((f, j) => (
                <div key={j} className="flex items-center gap-2 text-[12px] text-muted">
                  <div className={`w-1 h-1 rounded-full bg-${domain.color}`} />
                  {f}
                </div>
              ))}
            </div>

            <div className="flex items-center justify-between pt-5 border-t border-border">
              <span className="text-[11px] text-dim font-mono">{domain.sources}</span>
              <button className={`domain-enter group-hover:bg-${domain.id}-bg text-${domain.color} border-${domain.id}/35 text-[12px] font-medium px-5 py-2.5 rounded-full border transition-all flex items-center gap-2`}>
                Enter
                <ArrowRight className="w-3.5 h-3.5 transition-transform group-hover:translate-x-1" />
              </button>
            </div>
          </motion.div>
        ))}
      </div>
    </div>
  )
}

export default DomainSelection

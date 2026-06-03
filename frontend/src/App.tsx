import { useState } from 'react'
import { analyzeFile } from './api'
import type { AnalysisResult } from './types'
import FileUpload from './components/FileUpload'
import NarrativeAnalysis from './components/NarrativeAnalysis'
import MoodFlowChart from './components/MoodFlowChart'
import CharacterImpactChart from './components/CharacterImpactChart'
import CharacterScatterChart from './components/CharacterScatterChart'
import PNGViewer from './components/PNGViewer'

export default function App() {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<AnalysisResult | null>(null)

  async function handleSubmit(file: File, domain: string) {
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const data = await analyzeFile(file, domain)
      setResult(data)
    } catch (e: any) {
      setError(e.message ?? 'Unknown error')
    } finally {
      setLoading(false)
    }
  }

  return (
    <>
      <header className="header">
        <h1>Narrative Analysis</h1>
        <p>NLP + LLM literary intelligence</p>
      </header>

      <FileUpload onSubmit={handleSubmit} loading={loading} />

      {error && (
        <div className="error-card" style={{ marginTop: 20 }}>
          {error}
        </div>
      )}

      {result && (
        <div className="results" style={{ marginTop: 28 }}>
          <NarrativeAnalysis title={result.title} text={result.narrative_analysis} />

          <div className="grid-2">
            <MoodFlowChart data={result.mood_flow} turningPointScene={result.turning_point_scene} />
            <CharacterImpactChart data={result.characters} />
          </div>

          <div className="grid-2">
            <CharacterScatterChart data={result.characters} />
            <PNGViewer src={result.charts.articulation_network} title="Character Network — Structural Bridges" />
          </div>

          <div className="grid-2">
            <PNGViewer src={result.charts.entity_flow} title="Entity Flow Across Scenes" />
            <PNGViewer src={result.charts.emotion_distribution} title="Emotion Distribution" />
          </div>

          <PNGViewer src={result.charts.character_impact_scatter} title="Character Impact — Full Scatter (static)" />
        </div>
      )}
    </>
  )
}

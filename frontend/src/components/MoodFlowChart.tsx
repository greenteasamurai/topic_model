import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ReferenceLine, ReferenceArea, ResponsiveContainer,
} from 'recharts'
import type { MoodPoint } from '../types'

const ARC_COLORS = ['#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3', '#fdb462', '#b3de69']

interface ArcSpan { arc: number; x1: number; x2: number }

function buildArcSpans(data: MoodPoint[]): ArcSpan[] {
  const spans: ArcSpan[] = []
  if (!data.length) return spans
  let current = data[0].arc
  let start = data[0].scene
  for (const pt of data) {
    if (pt.arc !== current) {
      spans.push({ arc: current, x1: start, x2: pt.scene - 1 })
      current = pt.arc
      start = pt.scene
    }
  }
  spans.push({ arc: current, x1: start, x2: data[data.length - 1].scene })
  return spans
}

interface Props {
  data: MoodPoint[]
  turningPointScene: number | null
}

export default function MoodFlowChart({ data, turningPointScene }: Props) {
  const spans = buildArcSpans(data)

  return (
    <section className="card chart-card">
      <h3>Mood Flow</h3>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={data} margin={{ top: 8, right: 24, left: 0, bottom: 8 }}>
          <CartesianGrid strokeDasharray="3 3" opacity={0.3} />

          {spans.filter(s => s.arc >= 0).map(s => (
            <ReferenceArea
              key={`arc-${s.arc}-${s.x1}`}
              x1={s.x1} x2={s.x2}
              fill={ARC_COLORS[s.arc % ARC_COLORS.length]}
              fillOpacity={0.18}
            />
          ))}

          {turningPointScene != null && (
            <ReferenceLine
              x={turningPointScene}
              stroke="#e74c3c"
              strokeDasharray="5 3"
              label={{ value: 'Turning point', fill: '#e74c3c', fontSize: 11, position: 'insideTopLeft' }}
            />
          )}

          <XAxis dataKey="scene" label={{ value: 'Scene', position: 'insideBottom', offset: -4, fontSize: 11 }} />
          <YAxis domain={[-1.1, 1.1]} tickCount={5} width={36}
            label={{ value: 'Sentiment', angle: -90, position: 'insideLeft', fontSize: 11 }} />
          <Tooltip
            content={({ active, payload }) => {
              if (!active || !payload?.length) return null
              const pt = payload[0].payload as MoodPoint
              return (
                <div className="tt">
                  <strong>Scene {pt.scene}</strong>
                  <div>Score: {pt.score.toFixed(3)}</div>
                  <div>Emotion: {pt.dominant_emotion}</div>
                  {pt.entities.length > 0 && <div>{pt.entities.join(', ')}</div>}
                  {pt.arc >= 0 && <div>Arc {pt.arc}</div>}
                </div>
              )
            }}
          />
          <Line type="monotone" dataKey="score" dot={{ r: 3 }} stroke="#2c3e50" strokeWidth={2} />
        </LineChart>
      </ResponsiveContainer>
    </section>
  )
}

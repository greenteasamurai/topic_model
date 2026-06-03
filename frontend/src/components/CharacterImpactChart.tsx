import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ReferenceLine, ResponsiveContainer, Cell,
} from 'recharts'
import type { CharacterData } from '../types'

function quadrant(c: CharacterData): string {
  if (c.cohens_d > 0 && c.lagged_delta < 0) return 'Active catalyst'
  if (c.cohens_d > 0 && c.lagged_delta >= 0) return 'Tragic presence'
  if (c.cohens_d <= 0 && c.lagged_delta < 0) return 'Hidden driver'
  return 'Stabiliser'
}

const QUAD_COLOR: Record<string, string> = {
  'Active catalyst': '#e74c3c',
  'Tragic presence': '#e67e22',
  'Hidden driver':   '#8e44ad',
  'Stabiliser':      '#27ae60',
}

interface Props { data: CharacterData[] }

export default function CharacterImpactChart({ data }: Props) {
  const sorted = [...data].sort((a, b) => b.cohens_d - a.cohens_d)

  return (
    <section className="card chart-card">
      <h3>Character Mood Impact</h3>
      <div className="quad-legend">
        {Object.entries(QUAD_COLOR).map(([label, color]) => (
          <span key={label} style={{ color }} className="quad-item">● {label}</span>
        ))}
      </div>
      <ResponsiveContainer width="100%" height={Math.max(240, sorted.length * 36)}>
        <BarChart layout="vertical" data={sorted} margin={{ top: 4, right: 80, left: 80, bottom: 4 }}>
          <CartesianGrid strokeDasharray="3 3" horizontal={false} opacity={0.3} />
          <XAxis type="number" domain={['auto', 'auto']} tickCount={5}
            label={{ value: "Cohen's d", position: 'insideBottom', offset: -2, fontSize: 11 }} />
          <YAxis type="category" dataKey="name" width={76} tick={{ fontSize: 12 }} />
          <ReferenceLine x={0} stroke="#555" strokeWidth={1} />
          <Tooltip
            content={({ active, payload }) => {
              if (!active || !payload?.length) return null
              const c = payload[0].payload as CharacterData
              return (
                <div className="tt">
                  <strong>{c.name}</strong>
                  <div>Cohen's d: {c.cohens_d.toFixed(3)}</div>
                  <div>Lagged Δ: {c.lagged_delta.toFixed(3)}</div>
                  <div>Scenes: {c.scene_count} ({c.crisis_scene_count} crisis)</div>
                  <div>{quadrant(c)}</div>
                </div>
              )
            }}
          />
          <Bar dataKey="cohens_d" name="Cohen's d" radius={[0, 3, 3, 0]}>
            {sorted.map((c, i) => (
              <Cell key={i} fill={QUAD_COLOR[quadrant(c)]} fillOpacity={0.85} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </section>
  )
}

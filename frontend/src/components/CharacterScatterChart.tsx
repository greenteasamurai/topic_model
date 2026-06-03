import {
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip,
  ReferenceLine, ResponsiveContainer,
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


const CustomDot = (props: any) => {
  const { cx, cy, payload } = props
  const color = QUAD_COLOR[quadrant(payload as CharacterData)] ?? '#999'
  return (
    <g>
      <circle cx={cx} cy={cy} r={6} fill={color} fillOpacity={0.85} />
      <text x={cx + 9} y={cy + 4} fontSize={10} fill="#333">{payload.name}</text>
    </g>
  )
}

interface Props { data: CharacterData[] }

export default function CharacterScatterChart({ data }: Props) {
  return (
    <section className="card chart-card">
      <h3>Associative vs Causal Impact</h3>
      <p className="chart-subtitle">
        X: Cohen's d (presence at dark scenes) · Y: Lagged Δ (sentiment change after appearance)
      </p>
      <ResponsiveContainer width="100%" height={340}>
        <ScatterChart margin={{ top: 16, right: 40, left: 16, bottom: 32 }}>
          <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
          <XAxis type="number" dataKey="cohens_d" name="Cohen's d"
            label={{ value: "Cohen's d →", position: 'insideBottom', offset: -8, fontSize: 11 }} />
          <YAxis type="number" dataKey="lagged_delta" name="Lagged Δ" width={44}
            label={{ value: '← Lagged Δ', angle: -90, position: 'insideLeft', fontSize: 11 }} />
          <ReferenceLine x={0} stroke="#aaa" strokeDasharray="4 3" />
          <ReferenceLine y={0} stroke="#aaa" strokeDasharray="4 3" />
          <Tooltip
            content={({ active, payload }) => {
              if (!active || !payload?.length) return null
              const c = payload[0].payload as CharacterData
              return (
                <div className="tt">
                  <strong>{c.name}</strong>
                  <div>Cohen's d: {c.cohens_d.toFixed(3)}</div>
                  <div>Lagged Δ: {c.lagged_delta.toFixed(3)}</div>
                  <div>{quadrant(c)}</div>
                </div>
              )
            }}
          />
          <Scatter data={data} shape={<CustomDot />} />
        </ScatterChart>
      </ResponsiveContainer>
    </section>
  )
}

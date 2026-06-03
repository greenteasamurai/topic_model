import type { AnalysisResult } from './types'

export async function analyzeFile(file: File, domain: string): Promise<AnalysisResult> {
  const body = new FormData()
  body.append('file', file)
  body.append('domain', domain)

  const res = await fetch('/analyze', { method: 'POST', body })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail ?? 'Analysis failed')
  }
  return res.json()
}

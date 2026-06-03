import { useState, useCallback } from 'react'

interface Props {
  onSubmit: (file: File, domain: string) => void
  loading: boolean
}

const DOMAINS = [
  { value: 'book', label: 'Literary work' },
  { value: 'court', label: 'Legal / court transcript' },
  { value: 'meeting', label: 'Meeting transcript' },
]

export default function FileUpload({ onSubmit, loading }: Props) {
  const [file, setFile] = useState<File | null>(null)
  const [domain, setDomain] = useState('book')
  const [dragging, setDragging] = useState(false)

  const handleFile = (f: File) => {
    if (f.name.endsWith('.txt')) setFile(f)
  }

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setDragging(false)
    const f = e.dataTransfer.files[0]
    if (f) handleFile(f)
  }, [])

  return (
    <div className="upload-card">
      <div
        className={`drop-zone${dragging ? ' dragging' : ''}`}
        onDragOver={e => { e.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        onClick={() => document.getElementById('file-input')?.click()}
      >
        <input
          id="file-input"
          type="file"
          accept=".txt"
          style={{ display: 'none' }}
          onChange={e => { const f = e.target.files?.[0]; if (f) handleFile(f) }}
        />
        {file
          ? <p className="file-name">📄 {file.name}</p>
          : <p className="drop-hint">Drop a .txt file here or click to browse</p>
        }
      </div>

      <div className="upload-controls">
        <select value={domain} onChange={e => setDomain(e.target.value)} className="domain-select">
          {DOMAINS.map(d => <option key={d.value} value={d.value}>{d.label}</option>)}
        </select>

        <button
          className="analyze-btn"
          disabled={!file || loading}
          onClick={() => file && onSubmit(file, domain)}
        >
          {loading ? 'Analysing…' : 'Analyse'}
        </button>
      </div>

      {loading && (
        <p className="loading-hint">
          Running NLP pipeline + LLM analysis — typically 30–90 s for a novel.
        </p>
      )}
    </div>
  )
}
